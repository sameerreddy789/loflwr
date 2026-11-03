"""Lock integrity tests use temporary text files, never held-out data."""
import tempfile
from pathlib import Path
import unittest
import importlib.util
from unittest.mock import patch
from fedlora_edge.data import content_hash
from tools.confirm_encoder import file_hash, verify_lock, selective_result


class ConfirmationLockTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Optional numpy is absent")
    def test_abstention_denominator_and_empty_acceptance_are_explicit(self):
        result = selective_result([[5, 0], [1, 1.1], [0, 4]], [0, 0, 1], 2.)
        self.assertEqual(result["confirmation_answered"], 2)
        self.assertEqual(result["confirmation_abstained"], 1)
        self.assertEqual(result["confirmation_accuracy_on_answered"], 1.)
        self.assertAlmostEqual(result["confirmation_coverage"], 2/3)
        self.assertIsNone(selective_result([[5, 0]], [0], 6.)["confirmation_accuracy_on_answered"])
        with self.assertRaises(ValueError):
            selective_result([[5, 0]], [0], float("nan"))

    def test_model_metadata_does_not_substitute_for_tokenizer_hashes(self):
        from fedlora_edge.encoder_inference import verify_inference_assets, PINNED_ASSET_HASH
        with patch("fedlora_edge.encoder_inference.verify_encoder_assets", return_value={"asset_hash": "changed"}):
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                verify_inference_assets("unused-fixture")
        with patch("fedlora_edge.encoder_inference.verify_encoder_assets", return_value={"asset_hash": PINNED_ASSET_HASH}):
            self.assertEqual(verify_inference_assets("unused-fixture")["asset_hash"], PINNED_ASSET_HASH)

    def test_file_and_plan_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.txt"
            path.write_text("fixture")
            plan = {"locked_files": {str(path): file_hash(path)}, "reason": "test"}
            plan["lock_hash"] = content_hash(plan)
            verify_lock(plan)
            plan["reason"] = "modified"
            with self.assertRaisesRegex(ValueError, "lock was modified"):
                verify_lock(plan)
            plan["reason"] = "test"
            path.write_text("modified")
            with self.assertRaisesRegex(ValueError, "Frozen file changed"):
                verify_lock(plan)


if __name__ == "__main__":
    unittest.main()
