"""Readiness fixture tests; none of these measure model accuracy."""
import importlib.util
import io
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import demo_preflight as preflight


class PreflightTests(unittest.TestCase):
    def test_dashboard_default_and_explicit_paths_match_probe(self):
        root = preflight.ROOT
        for arguments in ([], ["--", "--run-dir", "runs/encoder-flower-balanced-v2"],
                ["--run-dir=runs/encoder-flower-balanced-v2", "--model-dir=assets/reference-mpnet",
                 "--policy-file="], ["--model-dir", str(root / "assets/reference-mpnet")]):
            with self.subTest(arguments=arguments):
                self.assertTrue(preflight.dashboard_config_matches(arguments, root))

    def test_dashboard_custom_or_malformed_paths_cannot_match_default_probe(self):
        for arguments in (["--run-dir", "runs/unconfirmed"], ["--model-dir=assets/other"],
                ["--run-dir="], ["--model-dir="],
                ["--run=runs/unconfirmed"], ["--model=assets/other"], ["--policy=other.json"],
                ["--policy-file", "other.json"], ["--run-dir"], ["--run-dir", "--model-dir"]):
            with self.subTest(arguments=arguments):
                self.assertFalse(preflight.dashboard_config_matches(arguments, preflight.ROOT))
        self.assertFalse(preflight.dashboard_config_matches([], preflight.ROOT, run_dir="runs/mirror"))

    def test_missing_dependencies_prevent_inference(self):
        with patch.object(preflight, "dependency_check", return_value={"missing": ["torch"], "drift": {}}), \
             patch.object(preflight, "evidence_check", return_value={}), \
             patch.object(preflight, "check_port", return_value={"ok": True, "action": "launch"}), \
             patch.object(preflight, "run_probe") as probe:
            report = preflight.run_checks("run", "assets", "policy")
        self.assertFalse(report["ready"])
        self.assertEqual(report["status"], "blocked")
        probe.assert_not_called()

    def test_changed_evidence_cannot_be_marked_ready(self):
        with patch.object(preflight, "dependency_check", return_value={"missing": [], "drift": {}}), \
             patch.object(preflight, "evidence_check", side_effect=ValueError("Hash mismatch")), \
             patch.object(preflight, "check_port", return_value={"ok": True, "action": "launch"}), \
             patch.object(preflight, "run_probe") as probe:
            report = preflight.run_checks("run", "assets", "policy")
        self.assertFalse(report["ready"])
        probe.assert_not_called()

    def test_real_published_files_verify_without_model_loading(self):
        result = preflight.evidence_check()
        self.assertEqual(result["matrix_cells"], 33)
        self.assertEqual(result["scope_queries"], 2279)

    def test_dependency_drift_is_disclosed_after_successful_probe(self):
        with patch.object(preflight, "dependency_check", return_value={"missing": [], "drift": {"torch": "fixture"}}), \
             patch.object(preflight, "evidence_check", return_value={}), \
             patch.object(preflight, "check_port", return_value={"ok": True, "action": "open_existing"}), \
             patch.object(preflight, "run_probe", return_value={"ok": True}):
            report = preflight.run_checks("run", "assets", "policy")
        self.assertTrue(report["ready"])
        self.assertEqual(report["status"], "ready_with_warnings")
        self.assertEqual(report["next_action"], "open_existing")

    def test_failed_inference_blocks_readiness(self):
        with patch.object(preflight, "dependency_check", return_value={"missing": [], "drift": {}}), \
             patch.object(preflight, "evidence_check", return_value={}), \
             patch.object(preflight, "check_port", return_value={"ok": True, "action": "launch"}), \
             patch.object(preflight, "run_probe", return_value={"ok": False, "error": "Missing assets"}):
            report = preflight.run_checks("run", "assets", "policy")
        self.assertFalse(report["ready"])

    def test_unknown_occupied_port_is_not_accepted_as_this_demo(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with patch.object(preflight, "listener_is_demo", return_value=False):
                self.assertFalse(preflight.check_port(port)["ok"])
        self.assertEqual(preflight.check_port(port)["state"], "available")

    def test_port_range_is_checked(self):
        for port in (True, 0, 80, 65536):
            with self.subTest(port=port), self.assertRaises(ValueError):
                preflight.check_port(port)

    def test_probe_uses_cached_loading_and_bounded_subprocess(self):
        done = subprocess.CompletedProcess([], 0, json.dumps({"ok": True, "measurement": {}}), "")
        with patch.object(preflight.subprocess, "run", return_value=done) as run:
            result = preflight.run_probe("run", "assets", "policy", 17)
        self.assertTrue(result["ok"])
        self.assertEqual(run.call_args.kwargs["timeout"], 17)
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_timeout_or_unstructured_probe_output_cannot_pass(self):
        with patch.object(preflight.subprocess, "run", side_effect=subprocess.TimeoutExpired("fixture", 12)):
            self.assertFalse(preflight.run_probe("run", "assets", "policy", 12)["ok"])
        for stdout, code in (("not JSON", 0), ('{"ok": "yes"}', 0), ('{"ok": true}', 1)):
            with self.subTest(stdout=stdout, code=code), patch.object(preflight.subprocess, "run",
                    return_value=subprocess.CompletedProcess([], code, stdout, "")):
                self.assertFalse(preflight.run_probe("run", "assets", "policy", 12)["ok"])

    def test_reports_stay_in_runs_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(preflight, "ROOT", Path(directory)):
            root = Path(directory)
            with self.assertRaises(ValueError):
                preflight.save_report(root / "README.md", {})
            path = root / "runs/check.json"
            preflight.save_report(path, {"fixture": True})
            with self.assertRaises(FileExistsError):
                preflight.save_report(path, {"fixture": False})
            self.assertTrue(json.loads(path.read_text())["fixture"])


HAS_DEPS = all(importlib.util.find_spec(name) for name in ("torch", "streamlit"))


@unittest.skipUnless(HAS_DEPS, "Optional Torch/Streamlit dependencies are absent")
class PreflightIdentityTests(unittest.TestCase):
    def test_unconfirmed_checkpoint_rejected_before_encoder_loading(self):
        saved = {"b_hash": "different", "config_hash": "different", "version": 500, "run_id": "fixture"}
        with patch("fedlora_edge.encoder_dashboard.load_evidence", return_value=({}, saved)), \
             patch("fedlora_edge.encoder_inference.EncoderPredictor") as predictor:
            with self.assertRaisesRegex(ValueError, "confirmed"):
                preflight.inference_probe("run", "assets", "policy")
        predictor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
