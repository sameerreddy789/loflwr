"""Development-only representation study data boundary and freeze checks."""
import importlib.util
import unittest

from fedlora_edge.data import INTENTS
from tools.run_encoder_representation_study import select_top_layers, validate_records


class RepresentationDataTests(unittest.TestCase):
    def fixture(self):
        train = [{"source_id": f"train:{i}", "source_split": "train", "query": f"training phrase {i}", "label": label}
                 for i, label in enumerate(INTENTS)]
        validation = [{"source_id": "train:999", "source_split": "train", "query": "development phrase", "label": INTENTS[0]}]
        return train, validation

    def test_rejects_test_origin_and_normalized_overlap(self):
        train, validation = self.fixture()
        validate_records(train, validation)
        with self.assertRaises(ValueError):
            validate_records(train, [dict(validation[0], source_id="test:999", source_split="test")])
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_records(train, [dict(validation[0], query="  TRAINING phrase 0  ")])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional Torch dependency is absent")
    def test_only_selected_last_layers_receive_gradients(self):
        import torch
        class Fixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Linear(2, 2)
                self.encoder = torch.nn.Module()
                self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
        model = Fixture()
        parameters = select_top_layers(model, 2)
        self.assertEqual(len(parameters), 4)
        output = model.embedding(torch.ones(1, 2))
        for layer in model.encoder.layer:
            output = layer(output)
        output.sum().backward()
        self.assertTrue(all(p.grad is None for p in model.embedding.parameters()))
        self.assertTrue(all(p.grad is None for layer in model.encoder.layer[:2] for p in layer.parameters()))
        self.assertTrue(all(p.grad is not None for layer in model.encoder.layer[2:] for p in layer.parameters()))
        with self.assertRaises(ValueError):
            select_top_layers(model, True)


if __name__ == "__main__":
    unittest.main()
