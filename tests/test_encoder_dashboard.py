"""UI fixtures are checks of behavior, not experiment evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HAS_DEPS = all(importlib.util.find_spec(x) for x in ("torch", "streamlit"))
APP = Path(__file__).parents[1] / "fedlora_edge/encoder_dashboard.py"


@unittest.skipUnless(HAS_DEPS, "Optional Torch/Streamlit dependencies are absent")
class EncoderDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.argv = patch.object(sys, "argv", [str(APP), "--run-dir", str(self.root)])
        self.argv.start()
        self.addCleanup(self.argv.stop)

    def fixture(self):
        from fedlora_edge.checkpoint import CheckpointStore
        from fedlora_edge.data import INTENTS, content_hash
        from fedlora_edge.encoder_adapter import MODEL_REVISION
        from fedlora_edge.encoder_head import HeadTrainer
        from fedlora_edge.state import hash_state
        state = HeadTrainer().get_state()
        identity = {"kind": "frozen_mpnet_linear_head", "model_revision": MODEL_REVISION,
                    "labels": list(INTENTS), "feature_dim": 768}
        identity["contract_hash"] = content_hash(identity)
        CheckpointStore(self.root / "selected").save(run_id="UI-FIXTURE-NOT-AN-EXPERIMENT",
            attempt_id=1, version=1, state=state, identity=identity, config_hash="fixture")
        metrics = {"accuracy": 0.5, "count": 2, "macro_f1": 0.5,
                   "worst_domain_accuracy": 0.5, "per_domain_accuracy": {"Fixture": 0.5}}
        row = {"split": "validation", "version": 1, "attempt_id": 1,
               "state_hash": hash_state(state), "metrics": metrics}
        (self.root / "summary.json").write_text(json.dumps({"selected": row, "history": [row]}))

    def app(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
        return app

    def test_missing_evidence_does_not_offer_inference(self):
        app = self.app()
        self.assertEqual(len(app.error), 1)
        self.assertFalse(any(b.label == "Classify query" for b in app.button))

    def test_fixture_renders_and_empty_query_stays_local(self):
        self.fixture()
        app = self.app()
        self.assertEqual(len(app.tabs), 3)
        next(b for b in app.button if b.label == "Classify query").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Enter a banking query" in w.value for w in app.warning))

    def test_mismatched_evidence_rejected(self):
        self.fixture()
        p = self.root / "summary.json"
        result = json.loads(p.read_text())
        result["selected"]["state_hash"] = "wrong"
        p.write_text(json.dumps(result))
        app = self.app()
        self.assertTrue(any("disagree" in e.value for e in app.error))

    def test_contract_tampering_rejected(self):
        self.fixture()
        p = self.root / "selected/checkpoint.json"
        saved = json.loads(p.read_text())
        saved["identity"]["labels"].reverse()
        p.write_text(json.dumps(saved))
        app = self.app()
        self.assertTrue(any("contract" in e.value for e in app.error))


if __name__ == "__main__":
    unittest.main()
