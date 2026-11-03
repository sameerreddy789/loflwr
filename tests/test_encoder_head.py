"""Head optimizer/protocol invariants; these fixtures are not BANKING77 evidence."""

import time
import unittest

from fedlora_edge.encoder_head import HeadConfig, HeadTrainer
from fedlora_edge.state import TensorData, hash_state, state_from_dict, state_to_dict

try:
    import torch
except ImportError:
    torch = None


class HeadConfigTests(unittest.TestCase):
    def test_invalid_optimizer_configuration_is_rejected(self):
        for kwargs in ({"learning_rate": 0}, {"learning_rate": float("nan")},
                       {"weight_decay": -1}, {"gradient_clip": 0}, {"batch_size": True},
                       {"cpu_threads": 1.5}, {"optimizer": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HeadConfig(**kwargs)


@unittest.skipIf(torch is None, "Optional torch is not installed")
class HeadTrainerTests(unittest.TestCase):
    def test_real_gradient_steps_learn_separable_training_fixture(self):
        trainer = HeadTrainer(3, 3, 11, HeadConfig(learning_rate=0.7, batch_size=3))
        x, labels = torch.eye(3), torch.arange(3)
        before = hash_state(trainer.get_state())
        result = trainer.train(x, labels, steps=30, seed=19)
        self.assertEqual(trainer.predict_scores(x).argmax(axis=1).tolist(), labels.tolist())
        self.assertLess(result["training_loss_after"], result["training_loss_before"])
        self.assertNotEqual(before, result["state_hash_after"])
        self.assertTrue(result["nonzero_head_gradient"])
        self.assertEqual(result["examples_processed"], 90)
        self.assertEqual(result["parameter_count"], 12)

    def test_state_roundtrip_and_invalid_state_are_atomic(self):
        trainer = HeadTrainer(3, 4)
        original = trainer.get_state()
        trainer.set_state(state_from_dict(state_to_dict(original)))
        self.assertEqual(trainer.get_state(), original)
        invalid = [dict(original, **{"head.bias": TensorData((4,), (float("nan"), 0, 0, 0))}),
                   dict(original, **{"head.weight": TensorData((4, 2), [0] * 8)}),
                   {"head.weight": original["head.weight"]}]
        for state in invalid:
            with self.assertRaises(ValueError):
                trainer.set_state(state)
            self.assertEqual(trainer.get_state(), original)

    def test_optimizer_and_random_stream_reset_reproduce_interleaved_clients(self):
        config = HeadConfig(learning_rate=0.03, batch_size=2, optimizer="adamw", weight_decay=0.01)
        trainer = HeadTrainer(3, 3, 17, config)
        initial = trainer.get_state()
        x, y = torch.eye(3).repeat(2, 1), torch.arange(3).repeat(2)
        trainer.train(x, y, 7, seed=42)
        expected = trainer.get_state()
        trainer.train(x, y.flip(0), 5, seed=99)
        trainer.set_state(initial)
        torch.rand(100)  # Global RNG activity must not affect local minibatches.
        trainer.train(x, y, 7, seed=42)
        self.assertEqual(trainer.get_state(), expected)

    def test_absent_classes_still_participate_in_softmax_loss(self):
        trainer = HeadTrainer(2, 20, config=HeadConfig(learning_rate=0.1, batch_size=2))
        trainer.set_state({"head.weight": TensorData((20, 2), [0] * 40),
                           "head.bias": TensorData((20,), [0] * 20)})
        trainer.train(torch.eye(2), torch.zeros(2, dtype=torch.long), 1, seed=1)
        bias = trainer.get_state()["head.bias"].values
        self.assertGreater(bias[0], 0)
        self.assertTrue(all(value < 0 for value in bias[1:]))
        self.assertEqual(trainer.predict_scores(torch.eye(2)).shape, (2, 20))

    def test_feature_gradients_and_global_initialization_rng_are_untouched(self):
        torch.manual_seed(127)
        before = torch.get_rng_state().clone()
        trainer = HeadTrainer(2, 2)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        x = torch.eye(2, requires_grad=True)
        trainer.train(x, torch.arange(2), 1, seed=2)
        self.assertIsNone(x.grad)

    def test_expired_deadline_or_callback_failure_preserves_incoming_state(self):
        trainer = HeadTrainer(2, 2)
        original = trainer.get_state()
        with self.assertRaises(TimeoutError):
            trainer.train(torch.eye(2), torch.arange(2), 2, seed=2, deadline_unix=time.time() - 1)
        self.assertEqual(original, trainer.get_state())

        def fail_after_step(step, work):
            self.assertEqual(step, 1)
            self.assertEqual(work["examples_processed"], 2)
            raise TimeoutError("Simulated coordinator cancellation")

        with self.assertRaises(TimeoutError):
            trainer.train(torch.eye(2), torch.arange(2), 2, seed=2, progress_callback=fail_after_step)
        self.assertEqual(original, trainer.get_state())

    def test_bad_features_labels_and_work_budgets_are_rejected(self):
        trainer = HeadTrainer(2, 2)
        cases = [(torch.eye(2), torch.tensor([0.0, 1.0]), 1),
                 (torch.eye(2), torch.tensor([0, 2]), 1),
                 (torch.ones(2, 3), torch.arange(2), 1),
                 (torch.tensor([[float("nan"), 0], [0, 1]]), torch.arange(2), 1),
                 (torch.eye(2), torch.arange(2), 0)]
        for x, y, steps in cases:
            with self.assertRaises(ValueError):
                trainer.train(x, y, steps, seed=2)

    def test_tail_batches_report_actual_work(self):
        trainer = HeadTrainer(3, 3, config=HeadConfig(batch_size=2))
        result = trainer.train(torch.eye(3), torch.arange(3), steps=3, seed=2)
        self.assertEqual(result["examples_processed"], 5)  # 2 + 1 + 2
        self.assertEqual(result["unique_examples_processed"], 3)
        self.assertEqual(result["completed_optimizer_steps"], 3)


if __name__ == "__main__":
    unittest.main()
