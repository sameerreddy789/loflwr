"""Optional Streamlit AppTests. All generated evidence is a temporary UI fixture.

These fixtures exercise widgets and error states, never model performance.
Run with Streamlit installed; the dependency-free suite skips this module.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None
APP = Path(__file__).parents[1] / "fedlora_edge/dashboard.py"


@unittest.skipUnless(HAS_STREAMLIT, "Streamlit is an optional dashboard dependency")
class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fedlora-ui-unit-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        arguments = patch.object(sys, "argv", [str(APP), "--runs-dir", str(self.root)])
        arguments.start()
        self.addCleanup(arguments.stop)

    def app(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(APP), default_timeout=180).run()
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        return app

    def fixture(self):
        run = self.root / "ui-unit-fixture"
        run.mkdir()
        config = {"run_id": "UI-UNIT-TEST-NOT-AN-EXPERIMENT", "kind": "ui_test_fixture",
                  "strategy": "equal_clipped", "seed": 42, "quorum": 3,
                  "deadline_seconds": 10, "clients": {"0": 10, "1": 20, "2": 10}}
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        events = [
            {"event": "attempt_started", "attempt_id": 1, "version": 0,
             "selected": ["0", "1", "2"], "budgets": {"0": 2, "1": 2, "2": 2}},
            {"event": "evaluation", "split": "validation", "attempt_id": 0, "version": 0,
             "metrics": {"count": 4, "accuracy": 0.25, "macro_f1": 0.125, "task_loss": 1.5}},
            {"event": "reply_accepted", "attempt_id": 1, "client_id": "0",
             "telemetry": {"optimizer_steps": 2, "examples_processed": 8,
                           "tokens_processed": 64, "local_train_seconds": 1.5}},
            {"event": "round_result", "attempt_id": 1, "status": "skipped",
             "version_before": 0, "version_after": 0, "accepted": ["0"], "absent": ["1", "2"]},
            {"event": "evaluation", "split": "validation", "attempt_id": 1, "version": 0,
             "metrics": {"count": 4, "accuracy": 0.25, "macro_f1": 0.125, "task_loss": 1.5,
                         "per_domain_accuracy": {"Card lifecycle": 0.25}}},
        ]
        (run / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    def test_empty_directory_has_no_metrics_or_query_submission(self):
        app = self.app()
        self.assertEqual(len(app.metric), 0)
        self.assertTrue(any("No experiment evidence yet" in item.value for item in app.markdown))
        self.assertFalse(any(button.label == "Score query" for button in app.button))

    def test_fixture_tables_and_replay_widgets_render(self):
        self.fixture()
        app = self.app()
        self.assertEqual(app.metric[0].value, "1")
        self.assertEqual(len(app.tabs), 5)
        app.sidebar.radio[0].set_value("Replay recorded events").run()
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        app.slider[0].set_value(0).run()
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        self.assertEqual(app.metric[0].value, "0")
        self.assertFalse(any(button.label == "Score query" for button in app.button))

    def test_missing_model_assets_display_recovery_message(self):
        self.fixture()
        app = self.app()
        model_field = next(widget for widget in app.text_input if widget.label == "Cached model directory")
        adapter_field = next(widget for widget in app.text_input if widget.label == "Common adapter directory")
        self.assertEqual(Path(model_field.value), APP.parents[1] / "assets/model")
        self.assertEqual(Path(adapter_field.value), APP.parents[1] / "assets/adapter")
        for widget in app.text_input:
            if widget.label in {"Cached model directory", "Common adapter directory"}:
                widget.set_value(str(self.root / "intentionally-missing-assets"))
        app.text_area[0].set_value("My transfer is still pending")
        next(button for button in app.button if button.label == "Score query").click().run()
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        self.assertTrue(any("directory is missing" in error.value for error in app.error))

    def test_new_run_does_not_change_selected_evidence(self):
        self.fixture()
        app = self.app()
        selected = app.sidebar.selectbox[0].value
        newer = self.root / "newer-ui-fixture"
        newer.mkdir()
        (newer / "config.json").write_text(json.dumps({"run_id": "NEW-UI-FIXTURE"}))
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.sidebar.selectbox[0].value, selected)


if __name__ == "__main__":
    unittest.main()
