"""Tiny invariants for the new encoder adapter; these are not accuracy evidence."""

import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from fedlora_edge.data import INTENTS
from fedlora_edge.encoder_adapter import (
    EncoderAdapterConfig, EncoderAdapterTrainer, _a_state, load_encoder_state,
    masked_mean_normalize, save_encoder_state,
)
from fedlora_edge.state import TensorData, hash_state, state_schema

try:
    import torch
    import psutil  # noqa: F401
except ImportError:
    torch = None


class EncoderConfigTests(unittest.TestCase):
    def test_rejects_nonfinite_and_invalid_work_limits(self):
        for settings in ({"adapter_lr": float("nan")}, {"head_lr": 0},
                         {"microbatch_size": True}, {"max_length": 385},
                         {"gradient_accumulation": 0}, {"weight_decay": -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                EncoderAdapterConfig(**settings)

    def test_state_serialization_rejects_changed_values_or_identity(self):
        state = {"head.bias": TensorData((2,), (0.1, 0.2))}
        identity = {"contract_hash": "contract", "state_schema": state_schema(state)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            save_encoder_state(path, state, identity)
            self.assertEqual(state, load_encoder_state(path, identity))
            with self.assertRaisesRegex(ValueError, "contract"):
                load_encoder_state(path, {**identity, "contract_hash": "different"})
            payload = json.loads(path.read_text())
            payload["state"]["head.bias"]["values"][0] = 5
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "hash/schema"):
                load_encoder_state(path, identity)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [1] + [ord(char) % 25 + 2 for char in text] + [2]

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        rows = [self.encode(text)[:max_length] for text in texts]
        width = max(map(len, rows))
        return {"input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows]),
                "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows])}


@unittest.skipIf(torch is None, "Optional torch/psutil not installed")
class EncoderTrainingTests(unittest.TestCase):
    def make_trainer(self):
        class Adapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(8, 3, bias=False)})
                self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 8, bias=False)})
                self.dropout = torch.nn.Dropout(0.05)
                torch.nn.init.zeros_(self.lora_B["default"].weight)

            def forward(self, hidden):
                return self.lora_B["default"](self.lora_A["default"](self.dropout(hidden)))

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(30, 8)
                self.adapter = Adapter()
                for name, parameter in self.named_parameters():
                    parameter.requires_grad_(".lora_B." in name)

            def forward(self, input_ids, attention_mask):
                hidden = self.embedding(input_ids)
                return SimpleNamespace(last_hidden_state=hidden + self.adapter(hidden))

        torch.manual_seed(13)
        torch.set_num_threads(1)
        trainer = object.__new__(EncoderAdapterTrainer)
        trainer.config = EncoderAdapterConfig(microbatch_size=2, max_length=32, head_lr=0.05)
        trainer.device = torch.device("cpu")
        trainer.encoder = Encoder()
        trainer.head = torch.nn.Linear(8, len(INTENTS))
        trainer.tokenizer = TinyTokenizer()
        trainer.identity = {"contract_hash": "tiny-fixture", "common_a_hash": hash_state(_a_state(trainer.encoder)),
                            "state_schema": state_schema(trainer.get_state())}
        return trainer

    def test_masked_pooling_ignores_padding_then_normalizes(self):
        hidden = torch.tensor([[[3.0, 4.0], [999.0, 999.0]]])
        result = masked_mean_normalize(hidden, torch.tensor([[1, 0]]))
        self.assertTrue(torch.allclose(result, torch.tensor([[0.6, 0.8]])))

    def test_only_b_and_head_change_with_real_gradients(self):
        trainer = self.make_trainer()
        fixed = trainer.frozen_hashes()
        initial = trainer.get_state()
        result = trainer.train([{"query": "my card", "label": INTENTS[0]}], 2, 42)
        current = trainer.get_state()
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["telemetry"]["nonzero_b_gradient"])
        self.assertEqual(trainer.frozen_hashes(), fixed)
        self.assertNotEqual(initial["head.weight"], current["head.weight"])
        self.assertTrue(any(initial[name] != current[name] for name in initial if ".lora_B." in name))
        self.assertEqual(result["telemetry"]["examples_processed"], 4)

    def test_reset_incoming_state_and_optimizer_reproduces_result(self):
        trainer = self.make_trainer()
        initial = trainer.get_state()
        records = [{"query": "card request", "label": INTENTS[0]},
                   {"query": "transfer please", "label": INTENTS[8]}]
        first = trainer.train(records, 2, 123)
        trainer.set_state(initial)
        second = trainer.train(records, 2, 123)
        self.assertEqual(first["state_hash"], second["state_hash"])
        self.assertEqual(first["telemetry"]["task_loss"], second["telemetry"]["task_loss"])

    def test_expired_deadline_preserves_state_and_reports_partial(self):
        trainer = self.make_trainer()
        initial = hash_state(trainer.get_state())
        result = trainer.train([{"query": "card", "label": INTENTS[0]}], 2, 42, deadline_unix=time.time() - 1)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["state_hash"], initial)
        self.assertEqual(result["telemetry"]["optimizer_steps"], 0)

    def test_cross_entropy_updates_even_locally_absent_class_biases(self):
        trainer = self.make_trainer()
        initial = trainer.head.bias.detach().clone()
        trainer.train([{"query": "card", "label": INTENTS[0]}], 1, 42)
        self.assertTrue(torch.all(trainer.head.bias.detach()[1:] != initial[1:]))

    def test_predict_shape_and_truncation_are_explicit(self):
        trainer = self.make_trainer()
        rows = [{"query": "x" * 100, "label": INTENTS[0]}]
        metrics = trainer.evaluate(rows)
        self.assertEqual(trainer.predict_scores(rows).shape, (1, 20))
        self.assertEqual(metrics["truncated_queries"], 1)
        self.assertEqual(metrics["count"], 1)
        self.assertEqual(trainer.predict_scores([]).shape, (0, 20))

    def test_tiny_huggingface_mpnet_peft_forward_and_backward(self):
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import MPNetConfig, MPNetModel
        except ImportError:
            self.skipTest("Optional transformers/peft not installed")
        trainer = self.make_trainer()
        base = MPNetModel(MPNetConfig(vocab_size=30, hidden_size=8, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=16))
        trainer.encoder = get_peft_model(base, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
            r=2, lora_alpha=4, lora_dropout=0.05, target_modules=["q", "v"], bias="none"))
        for name, parameter in trainer.encoder.named_parameters():
            parameter.requires_grad_(".lora_B." in name)
        trainer.identity = {"contract_hash": "tiny-mpnet-peft", "common_a_hash": hash_state(_a_state(trainer.encoder)),
                            "state_schema": state_schema(trainer.get_state())}
        fixed = trainer.frozen_hashes()
        result = trainer.train([{"query": "card", "label": INTENTS[0]}], 1, 42)
        self.assertTrue(result["telemetry"]["nonzero_b_gradient"])
        self.assertEqual(fixed, trainer.frozen_hashes())


if __name__ == "__main__":
    unittest.main()
