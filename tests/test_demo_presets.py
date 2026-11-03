"""Recorded views must preserve experiment identities and never invoke a model."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fedlora_edge.demo_presets import ROOT, load_comparisons, load_recovery, load_scope


class EvidencePresetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "research/results/remaining-work-v1"
        self.folder.mkdir(parents=True)
        self.result = load_recovery("five")

    def save(self, result):
        raw = json.dumps(result).encode()
        (self.folder / "five-recovery.json").write_bytes(raw)
        (self.folder / "verification.json").write_text(json.dumps({
            "runtime_exports": {"five-recovery.json": hashlib.sha256(raw).hexdigest()}}))

    def test_published_evidence_loads(self):
        self.assertEqual(load_recovery("three")["accepted_counts"], [3, 2, 3, 1, 3])
        self.assertEqual(self.result["accepted_counts"], [5, 4, 5, 2, 5])
        self.assertEqual(load_comparisons()["completed"], 33)
        self.assertEqual(load_scope()["comparisons"][-1]["with_scope_false_accepts"], 57)

    def test_changed_file_cannot_reuse_published_checksum(self):
        self.save(self.result)
        with (self.folder / "five-recovery.json").open("ab") as target:
            target.write(b" ")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_recovery(root=self.root)

    def test_committed_update_requires_quorum_equal_weights_and_current_state(self):
        cases = (("accepted", ["0"]), ("coefficients", {"0": 1.}),
                 ("all_accepted_clients_started_from_current_head", False),
                 ("version_after", 8), ("incoming_head_hash", "stale"))
        for field, value in cases:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.result)
                changed["rounds"][0][field] = value
                self.save(changed)
                with self.assertRaises(ValueError):
                    load_recovery(root=self.root)

    def test_skipped_attempt_must_retain_state_and_have_no_weights(self):
        for field, value in (("checkpoint_hash", "changed"), ("version_after", 4),
                             ("coefficients", {"0": .5}),
                             ("below_quorum_retention_passed", False)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.result)
                changed["rounds"][3][field] = value
                self.save(changed)
                with self.assertRaises(ValueError):
                    load_recovery(root=self.root)

    def test_duplicate_reply_and_wrong_final_state_rejected(self):
        changed = copy.deepcopy(self.result)
        changed["rounds"][0]["accepted"][1] = "0"
        self.save(changed)
        with self.assertRaises(ValueError):
            load_recovery(root=self.root)
        changed = copy.deepcopy(self.result)
        changed["final_head_hash"] = "changed"
        self.save(changed)
        with self.assertRaises(ValueError):
            load_recovery(root=self.root)


HAS_UI = all(importlib.util.find_spec(x) for x in ("torch", "streamlit"))


@unittest.skipUnless(HAS_UI, "Optional Torch/Streamlit dependencies are absent")
class DemoNavigationTests(unittest.TestCase):
    def test_recorded_views_work_without_checkpoint_or_inference_assets(self):
        from streamlit.testing.v1 import AppTest
        app_path = ROOT / "fedlora_edge/encoder_dashboard.py"
        with tempfile.TemporaryDirectory() as missing, patch.object(
                sys, "argv", [str(app_path), "--run-dir", missing]), patch(
                "fedlora_edge.encoder_inference.EncoderPredictor") as predictor:
            app = AppTest.from_file(str(app_path), default_timeout=30).run()
            self.assertEqual(next(r for r in app.radio if r.label == "Choose a demo view").value, "Custom run")
            self.assertEqual(len(app.error), 1)
            next(r for r in app.radio if r.label == "Choose a demo view").set_value("Recorded recovery").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            next(r for r in app.radio if r.label == "Choose a recorded attempt").set_value(4).run()
            self.assertTrue(any("unchanged" in x.value for x in app.warning))
            app.selectbox[0].set_value("Three clients").run()
            self.assertFalse(app.exception)
            next(r for r in app.radio if r.label == "Choose a demo view").set_value("Experiment comparisons").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertFalse(any(b.label == "Classify query" for b in app.button))
            predictor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
