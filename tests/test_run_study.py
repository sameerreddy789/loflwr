"""Study matrix, frozen selectors, and bounded child cleanup without model imports."""

import importlib.util
import json
from argparse import Namespace
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("run_study", Path(__file__).parents[1] / "tools/run_study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class StudyTests(unittest.TestCase):
    def test_pilot_has_declared_comparators_and_refs(self):
        specs = study.study_specs("pilot")
        self.assertEqual([spec["key"] for spec in specs], ["base", "centralized", "sample", "equal", "main", "dropout", "omit4"])
        self.assertEqual(next(item for item in specs if item["key"] == "main")["budgets"], [2, 16, 8, 4, 2])
        self.assertTrue(next(item for item in specs if item["key"] == "dropout")["dropout"])
        self.assertTrue(next(item for item in specs if item["key"] == "omit4")["omission"])

    def test_full_matrix_counts_and_seeds(self):
        specs = study.study_specs("full")
        self.assertEqual(len(specs), 40)
        self.assertEqual(sum(item["kind"] == "federated" for item in specs), 30)
        self.assertEqual(sum(item["kind"] == "centralized" for item in specs), 9)
        self.assertEqual({item["seed"] for item in specs}, {42, 43, 44})
        self.assertEqual(len({item["key"] for item in specs}), 40)

    def test_selection_is_explicit_and_retains_plan_order(self):
        specs = study.study_specs("pilot")
        self.assertEqual(study.select_keys(specs, "main,base"), ["base", "main"])
        with self.assertRaises(ValueError):
            study.select_keys(specs, "missing")
        self.assertEqual(study.select_keys(study.study_specs("full"), "omit4-*") , ["omit4-s42", "omit4-s43", "omit4-s44"])

    def test_child_timeout_preserves_its_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "process.log"
            result = study.run_child(
                [sys.executable, "-u", "-c", "import time; print('started', flush=True); time.sleep(30)"],
                path, time.monotonic() + 1.0,
            )
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["returncode"], 0)
            self.assertIn("command:", path.read_text(encoding="utf-8"))
            self.assertLess(result["wall_seconds"], 10)

    def test_calibration_test_worker_refuses_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps({"run_dir": directory, "kind": "frozen_base",
                                               "study_scope": "CALIBRATION", "evaluate_final_test": False}))
            with self.assertRaisesRegex(ValueError, "disabled for calibration"):
                study.model_worker(config_path, phase="test", resume=False)

    def test_plan_freezes_all_configs_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "initial_b.json").write_text(json.dumps({
                "identity": {"common_a_hash": "test-fixture"}, "b_hash": "test-fixture",
            }))
            data = root / "data"
            (data / "manifests").mkdir(parents=True)
            (data / "manifests/index.json").write_text("{}")
            args = Namespace(study_dir=root / "study", attempts=1, data_dir=data,
                             asset_dir=root / "model", adapter_dir=adapter, mode="pilot",
                             runs="main", study_id="plan-unit-test", cpu_threads=1,
                             eval_batch_size=20, deadline_seconds=60, device="cpu", worker_cpus=1)
            with patch("fedlora_edge.data.verify_manifests", return_value={"manifest_set_hash": "test-fixture"}):
                plan = study.freeze_plan(args)
            self.assertEqual(plan["scope"], "CALIBRATION")
            self.assertEqual(len(plan["runs"]), 7)
            self.assertEqual(plan["initial_selection"], ["main"])
            self.assertTrue(all(Path(run["config_path"]).is_file() for run in plan["runs"]))
            configs = [json.loads(Path(run["config_path"]).read_text()) for run in plan["runs"]]
            self.assertTrue(all(not config["evaluate_final_test"] for config in configs))
            study.validate_plan(plan)
            first_config = Path(plan["runs"][0]["config_path"])
            first_config.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Frozen configuration changed"):
                study.validate_plan(plan)

    def test_reference_checkpoint_rejects_different_budget_contract(self):
        config = {"kind": "centralized", "run_id": "fixture", "budgets": {"0": 2}}
        checkpoint = {"run_id": "fixture", "config_hash": study.canonical_hash(config)}
        study.validate_run_checkpoint(config, checkpoint, Path("unused"))
        with self.assertRaisesRegex(ValueError, "configuration hash"):
            study.validate_run_checkpoint({**config, "budgets": {"0": 3}}, checkpoint, Path("unused"))

    def test_federated_checkpoint_matches_wrapped_runtime_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = {"kind": "federated", "run_id": "fixture", "resume": False, "seed": 42}
            produced = {"runtime": {key: value for key, value in config.items() if key != "resume"}, "clients": {"0": 10}}
            (path / "config.json").write_text(json.dumps(produced))
            checkpoint = {"run_id": "fixture", "config_hash": study.canonical_hash(produced)}
            study.validate_run_checkpoint(config, checkpoint, path)
            with self.assertRaisesRegex(ValueError, "runtime differs"):
                study.validate_run_checkpoint({**config, "seed": 43}, checkpoint, path)


if __name__ == "__main__":
    unittest.main()
