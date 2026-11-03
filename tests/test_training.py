"""Mask/stream/metric invariants; optional tiny-model tests are not ML evidence."""

import itertools
import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from fedlora_edge.data import INTENTS
from fedlora_edge.state import TensorData, hash_state, state_schema
from fedlora_edge.training import (
    CompletionEncoder,
    LocalTrainer,
    TrainingConfig,
    classification_metrics,
    derive_client_seed,
    example_indices,
    load_b_state,
    save_b_state,
)

try:
    import torch
except ImportError:
    torch = None


class TinyTokenizer:
    eos_token_id = 99
    pad_token_id = 99

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 80 + 1 for char in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"system: {messages[0]['content']}\nuser: {messages[1]['content']}\nassistant: "


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.encoder = CompletionEncoder(TinyTokenizer(), 128)

    def test_prompt_mask_preserves_true_eos(self):
        example = self.encoder.encode("Where is my card?", "card_arrival")
        boundary = example["prompt_length"]
        self.assertTrue(all(label == -100 for label in example["labels"][:boundary]))
        self.assertEqual(example["labels"][boundary:], self.encoder.completions["card_arrival"])
        self.assertEqual(example["labels"][-1], TinyTokenizer.eos_token_id)
        self.assertEqual(example["attention_mask"][-1], 1)

    def test_truncation_keeps_identical_candidate_prompt_and_all_completion(self):
        examples = [self.encoder.encode("a long request " * 100, label) for label in INTENTS]
        prompts = [example["input_ids"][:example["prompt_length"]] for example in examples]
        self.assertTrue(all(prompt == prompts[0] for prompt in prompts))
        self.assertTrue(all(example["truncated_query"] for example in examples))
        self.assertTrue(all(len(example["input_ids"]) <= 128 for example in examples))
        self.assertTrue(all(example["labels"][-1] == 99 for example in examples))

    def test_impossible_sequence_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "preserve"):
            CompletionEncoder(TinyTokenizer(), 3)

    @unittest.skipIf(torch is None, "Optional torch is not installed")
    def test_padding_eos_id_is_masked_but_true_eos_is_supervised(self):
        examples = [self.encoder.encode("a", "card_arrival"), self.encoder.encode("a longer question", "transfer_not_received_by_recipient")]
        batch = self.encoder.collate(examples)
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())
        for index, example in enumerate(examples):
            true_eos = len(example["input_ids"]) - 1
            self.assertEqual(batch["input_ids"][index, true_eos], 99)
            self.assertEqual(batch["labels"][index, true_eos], 99)


class StreamAndMetricTests(unittest.TestCase):
    def test_saved_b_is_bound_to_actual_adapter_identity_and_hash(self):
        state = {"layer.lora_B.default.weight": TensorData((1, 1), (0.25,))}
        identity = {"contract_hash": "contract", "common_a_hash": "a", "model_revision": "model", "b_schema": state_schema(state)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b.json"
            save_b_state(path, state, identity)
            self.assertEqual(load_b_state(path, identity), state)
            with self.assertRaisesRegex(ValueError, "common_a_hash mismatch"):
                load_b_state(path, {**identity, "common_a_hash": "different-a"})
            payload = json.loads(path.read_text())
            payload["state"]["layer.lora_B.default.weight"]["values"][0] = 1.0
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "B hash mismatch"):
                load_b_state(path, identity)

    def test_invalid_numerical_configuration_is_rejected(self):
        for value in (float("nan"), float("inf"), -0.1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TrainingConfig(learning_rate=value)
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TrainingConfig(gradient_accumulation=value)

    def test_streams_do_not_depend_on_omitted_clients(self):
        all_streams = {client: list(itertools.islice(example_indices(10, derive_client_seed(42, 3, client)), 24)) for client in range(5)}
        survivors = {client: list(itertools.islice(example_indices(10, derive_client_seed(42, 3, client)), 24)) for client in [0, 1, 2, 3]}
        self.assertEqual({client: all_streams[client] for client in survivors}, survivors)
        self.assertNotEqual(all_streams[0], all_streams[1])
        self.assertEqual(set(all_streams[0][:10]), set(range(10)))

    def test_metric_denominators_and_zero_f1(self):
        metrics = classification_metrics(list(INTENTS), [INTENTS[0]] * 20)
        self.assertAlmostEqual(metrics["accuracy"], 0.05)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 21) / 20)
        self.assertEqual(metrics["worst_domain_accuracy"], 0)
        self.assertEqual(metrics["domain_gap"], 0.25)

    def test_metrics_reject_unknown_labels(self):
        with self.assertRaises(ValueError):
            classification_metrics(["not_an_intent"], [INTENTS[0]])


@unittest.skipIf(torch is None, "Optional torch is not installed")
class TrainingLoopTests(unittest.TestCase):
    def make_trainer(self):
        class Adapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(8, 3, bias=False)})
                self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 100, bias=False)})
                torch.nn.init.zeros_(self.lora_B["default"].weight)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(100, 8)
                self.adapter = Adapter()
                for name, parameter in self.named_parameters():
                    parameter.requires_grad_(".lora_B." in name)

            def forward(self, input_ids, attention_mask, labels=None):
                hidden = self.embedding(input_ids)
                logits = self.adapter.lora_B["default"](self.adapter.lora_A["default"](hidden))
                loss = None if labels is None else torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 100), labels[:, 1:].reshape(-1), ignore_index=-100)
                return SimpleNamespace(loss=loss, logits=logits)

        torch.manual_seed(7)
        trainer = object.__new__(LocalTrainer)
        trainer.config = TrainingConfig(cpu_threads=1, eval_batch_size=4)
        trainer.device = torch.device("cpu")
        trainer.model = TinyModel()
        trainer.tokenizer = TinyTokenizer()
        trainer.encoder = CompletionEncoder(trainer.tokenizer, 128)
        trainer.b_parameters = {name: value for name, value in trainer.model.named_parameters() if value.requires_grad}
        a_state = {name: TensorData.from_torch(value) for name, value in trainer.model.named_parameters() if ".lora_A." in name}
        trainer.identity = {"common_a_hash": hash_state(a_state), "b_schema": state_schema(trainer.get_b_state())}
        return trainer

    def test_only_b_changes_and_every_attempt_resets_optimizer_and_incoming_b(self):
        trainer = self.make_trainer()
        initial = trainer.get_b_state()
        fixed_before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters() if not parameter.requires_grad}
        records = [{"query": "my card", "label": "card_arrival"}]
        first = trainer.train(records, initial, 2, 42, 0, 0)
        second = trainer.train(records, initial, 2, 42, 0, 0)
        self.assertNotEqual(first["b_hash"], hash_state(initial))
        self.assertEqual(first["b_hash"], second["b_hash"])
        self.assertEqual(first["telemetry"]["optimizer_steps"], 2)
        self.assertEqual(first["telemetry"]["microbatches"], 8)
        for name, parameter in trainer.model.named_parameters():
            if name in fixed_before:
                self.assertTrue(torch.equal(parameter, fixed_before[name]))

    def test_expired_deadline_cannot_claim_successful_work(self):
        trainer = self.make_trainer()
        initial = trainer.get_b_state()
        result = trainer.train([{"query": "my card", "label": "card_arrival"}], initial, 2, 42, 0, 0, deadline_unix=time.time() - 1)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["telemetry"]["optimizer_steps"], 0)
        self.assertEqual(result["b_hash"], hash_state(initial))

    def test_training_counts_ragged_rows_and_preserves_true_eos(self):
        trainer = self.make_trainer()
        trainer.config = TrainingConfig(microbatch_size=2, gradient_accumulation=1, max_length=256, cpu_threads=1)
        trainer.encoder = CompletionEncoder(trainer.tokenizer, trainer.config.max_length)
        records = [{"query": "a", "label": "card_arrival"}, {"query": "long banking query " * 100, "label": "transfer_not_received_by_recipient"}]
        encoded = [trainer.encoder.encode(record["query"], record["label"]) for record in records]
        batch = trainer.encoder.collate(encoded)
        self.assertTrue((batch["attention_mask"] == 0).any())
        self.assertEqual(sum(item["truncated_query"] for item in encoded), 1)
        result = trainer.train(records, trainer.get_b_state(), 1, 42, 0, 0)
        telemetry = result["telemetry"]
        self.assertEqual(telemetry["tokens_processed"], int(batch["attention_mask"].sum()))
        self.assertEqual(telemetry["supervised_tokens"], int((batch["labels"][:, 1:] != -100).sum()))
        self.assertEqual(telemetry["supervised_tokens"], sum(item["completion_length"] for item in encoded))
        self.assertEqual(telemetry["examples_processed"], 2)
        self.assertEqual(telemetry["truncated_examples_processed"], 1)

    def test_nonfinite_gradients_are_rejected_before_b_update(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                trainer = self.make_trainer()
                initial = trainer.get_b_state()
                parameter = next(iter(trainer.b_parameters.values()))
                handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, value))
                try:
                    with self.assertRaisesRegex(RuntimeError, "non-finite"):
                        trainer.train([{"query": "my card", "label": "card_arrival"}], initial, 1, 42, 0, 0)
                finally:
                    handle.remove()
                self.assertEqual(hash_state(trainer.get_b_state()), hash_state(initial))
                self.assertTrue(all(parameter.grad is None for parameter in trainer.b_parameters.values()))
                self.assertFalse(trainer.model.training)

    def test_nonfinite_task_loss_is_rejected_before_b_update(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                trainer = self.make_trainer()
                initial = trainer.get_b_state()

                def corrupt_loss(module, arguments, output):
                    output.loss = output.loss * value
                    return output

                handle = trainer.model.register_forward_hook(corrupt_loss)
                try:
                    with self.assertRaisesRegex(FloatingPointError, "Nonfinite task loss"):
                        trainer.train([{"query": "my card", "label": "card_arrival"}], initial, 1, 42, 0, 0)
                finally:
                    handle.remove()
                self.assertEqual(hash_state(trainer.get_b_state()), hash_state(initial))
                self.assertTrue(all(parameter.grad is None for parameter in trainer.b_parameters.values()))
                self.assertFalse(trainer.model.training)

    def test_completion_score_is_token_sum_including_eos_and_ignores_padding(self):
        trainer = self.make_trainer()
        examples = [trainer.encoder.encode("card", "card_arrival"), trainer.encoder.encode("a longer user query", "transfer_not_received_by_recipient")]
        scores = trainer._scores_for_examples(examples)
        # Initial B is zero, so this small test model has a uniform 100-token
        # distribution. Exactly the completion tokens, including EOS, count.
        for example, (score, count) in zip(examples, scores):
            self.assertEqual(count, example["completion_length"])
            self.assertAlmostEqual(score, -count * math.log(100), places=4)
        self.assertNotEqual(scores[0][0], scores[1][0])

    def test_supervised_row_scores_match_full_logit_cross_entropy(self):
        trainer = self.make_trainer()
        # Nonuniform logits catch incorrect row selection that a zero-B model's
        # uniform distribution cannot distinguish.
        with torch.no_grad():
            for parameter in trainer.b_parameters.values():
                parameter.copy_(torch.randn_like(parameter) * 0.2)
        examples = [trainer.encoder.encode("banking request " * (index + 1), label)
                    for index, label in enumerate(INTENTS[:7])]
        expected = []
        trainer.model.eval()
        with torch.inference_mode():
            for offset in range(0, len(examples), trainer.config.eval_batch_size):
                batch = trainer.encoder.collate(examples[offset:offset + trainer.config.eval_batch_size])
                logits = trainer.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1].float()
                labels = batch["labels"][:, 1:]
                mask = labels != -100
                losses = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="none").reshape(labels.shape)
                expected.extend((float(score), int(count)) for score, count in zip(-(losses * mask).sum(dim=1), mask.sum(dim=1)))
        self.assertEqual(trainer._scores_for_examples(examples), expected)

    def test_progress_callback_and_interrupted_work(self):
        trainer = self.make_trainer()
        observed = []
        result = trainer.train([{"query": "my card", "label": "card_arrival"}], trainer.get_b_state(), 4, 42, 0, 0, stop_after_steps=2, progress_callback=lambda step, telemetry: observed.append(step))
        self.assertEqual(observed, [1, 2])
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["telemetry"]["optimizer_steps"], 2)

    def test_centralized_replay_consumes_exact_per_client_work(self):
        trainer = self.make_trainer()
        clients = {client: [{"query": f"query {client}", "label": INTENTS[client * 4]}] for client in range(5)}
        result = trainer.train_centralized(clients, trainer.get_b_state(), [1, 2, 1, 1, 1], 42, 0)
        self.assertEqual(result["telemetry"]["completed_steps_by_client"], {0: 1, 1: 2, 2: 1, 3: 1, 4: 1})
        self.assertEqual(result["telemetry"]["microbatches"], 24)
        self.assertEqual(result["telemetry"]["examples_processed"], 24)


if __name__ == "__main__":
    unittest.main()
