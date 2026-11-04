"""Integration guard fixtures; no model, real query, or training run is loaded."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from fedlora_edge.data import content_hash
from fedlora_edge.state import TensorData
from tools import check_bucketed_integration as integration


@contextmanager
def temporary_root():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.object(integration, "ROOT", root), patch.object(integration.base, "ROOT", root):
            yield root


def fixture_plan(output):
    return {"source": {"participants": {role: {"records_file": f"assets/fixture/{role}.json"}
                                           for role in integration.COUNTS}},
            "model": {"asset_dir": "assets/fake-model"}, "lock_hash": "fixture-lock",
            "runtime": {"participant_environment": {}}, "output": str(output)}


class BucketedIntegrationTests(unittest.TestCase):
    def test_fixed_design_covers_all_participants_without_test_roles(self):
        self.assertEqual(sum(integration.COUNTS.values()), 2960)
        self.assertEqual(set(integration.COUNTS), {"0", "1", "2", "3", "4", "server"})
        self.assertEqual(integration.DESIGN["head_smoke"]["initial_seed"], 42)
        self.assertEqual(integration.DESIGN["head_smoke"]["optimizer_steps_per_client"], 1)
        self.assertEqual(integration.DESIGN["head_smoke"]["strategy"], "equal")
        self.assertEqual(integration.DESIGN["preparation"]["batch_size"], 8)
        self.assertEqual(integration.DESIGN["preparation"]["cpu_threads"], 1)
        self.assertEqual(integration.DESIGN["tokenizer_probes"]["probe_hash"], content_hash(integration.TOKENIZER_PROBES))

    def test_output_paths_are_fresh_private_run_and_optional_public_json(self):
        with temporary_root() as root:
            output, public = integration.output_paths(root / "runs/new", root / "research/results/new.json")
            self.assertEqual(output, root / "runs/new")
            self.assertEqual(public, root / "research/results/new.json")
            for bad in (root / "assets/new", root / "runs", root / "runs/../../outside"):
                with self.assertRaises(ValueError):
                    integration.output_paths(bad)
            for bad in (root / "public.json", root / "research/results/subdir"):
                with self.assertRaises(ValueError):
                    integration.output_paths(output, bad)
            output.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                integration.output_paths(output)

    def test_each_child_receives_exactly_one_owned_bundle(self):
        with temporary_root() as root:
            output = root / "runs/fixture"
            plan = fixture_plan(output)
            for role in integration.COUNTS:
                command = integration.participant_command(plan, output, role)
                index = command.index("--records")
                self.assertEqual(command[index + 1], str(root / f"assets/fixture/{role}.json"))
                self.assertEqual(command[index + 2], "--asset-dir")
                self.assertEqual(command[command.index("--participant") + 1], role)
                self.assertEqual(command[-1], str(output / "participants" / role / "data"))
            with self.assertRaises(ValueError):
                integration.participant_command(plan, output, "confirmation")

    def test_timeout_keeps_started_log_failure_and_never_overwrites_attempt(self):
        with temporary_root() as root:
            output = root / "runs/fixture"
            plan = fixture_plan(output)
            process = Mock()
            process.wait.side_effect = subprocess.TimeoutExpired("fixture", 900)
            with patch.object(integration, "verify_plan", return_value=plan), \
                 patch.object(integration.subprocess, "Popen", return_value=process), \
                 patch.object(integration, "stop_process_tree") as stop:
                with self.assertRaises(subprocess.TimeoutExpired):
                    integration.prepare_participant(plan, output, "0")
                stop.assert_called_once_with(process)
                with self.assertRaises(FileExistsError):
                    integration.prepare_participant(plan, output, "0")
            folder = output / "participants/0"
            self.assertTrue((folder / "started.json").exists())
            self.assertTrue((folder / "process.log").exists())
            self.assertFalse((folder / "completed.json").exists())
            failure = integration.base.read(folder / "failure.json")
            self.assertEqual(failure["error_type"], "TimeoutExpired")
            self.assertEqual(failure["participant"], "0")

    def test_lock_and_input_mutation_are_detected_without_loading_a_model(self):
        with temporary_root() as root:
            output = root / "runs/fixture"
            output.mkdir(parents=True)
            source = root / "source.py"
            source.write_text("# fixture\n", encoding="utf-8")
            source_plan = {"fixture": True}
            model = {"head_hash": "fixture-head", "files": {}}
            source_contract = {"bucketed_plan": source_plan, "files": {"source.py": integration.base.sha(source)}}
            plan = {"kind": integration.KIND, "dataset": integration.DATASET,
                    "design": integration.DESIGN, "output": "runs/fixture", "runtime": {"fixture": True},
                    "model": model, "source": source_contract}
            plan["lock_hash"] = content_hash(plan)
            integration.base.write_new(output / "plan.json", plan)
            integration.base.write_new(output / "bucketed-plan.json", source_plan)
            with patch.object(integration, "runtime", return_value={"fixture": True}), \
                 patch.object(integration, "validate_plan", return_value=source_plan), \
                 patch.object(integration.base, "model_contract", return_value=model), \
                 patch.object(integration, "source_contract", return_value=source_contract):
                self.assertEqual(integration.verify_plan(plan, output), plan)
                source.write_text("# changed\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Frozen integration input"):
                    integration.verify_plan(plan, output)
                plan["runtime"] = {"changed": True}
                with self.assertRaisesRegex(ValueError, "plan or runtime"):
                    integration.verify_plan(plan, output)

    def test_rehashed_plan_cannot_drop_a_required_input(self):
        with temporary_root() as root:
            output = root / "runs/fixture"
            output.mkdir(parents=True)
            source_plan = {"fixture": True}
            model = {"head_hash": "fixture-head", "files": {}}
            complete_source = {"bucketed_plan": source_plan,
                               "files": {"required.py": "a" * 64, "other.py": "b" * 64},
                               "participants": {"0": {"count": 513, "source_ids_hash": "c" * 64}}}
            plan = {"kind": integration.KIND, "dataset": integration.DATASET,
                    "design": integration.DESIGN, "output": "runs/fixture", "runtime": {"fixture": True},
                    "model": model, "source": copy.deepcopy(complete_source)}
            del plan["source"]["files"]["required.py"]
            plan["lock_hash"] = content_hash(plan)
            integration.base.write_new(output / "plan.json", plan)
            integration.base.write_new(output / "bucketed-plan.json", source_plan)
            with patch.object(integration, "runtime", return_value={"fixture": True}), \
                 patch.object(integration, "validate_plan", return_value=source_plan), \
                 patch.object(integration.base, "model_contract", return_value=model), \
                 patch.object(integration, "source_contract", return_value=complete_source):
                with self.assertRaisesRegex(ValueError, "Required model/source contract"):
                    integration.verify_plan(plan, output)

    def test_state_difference_checks_schema_and_finiteness(self):
        before = {"head.weight": TensorData((1, 2), (1., 2.))}
        after = {"head.weight": TensorData((1, 2), (1., 2.25))}
        self.assertEqual(integration.state_difference(before, after), 0.25)
        with self.assertRaises(ValueError):
            integration.state_difference(before, {"other": TensorData((1, 2), (1., 2.))})
        with self.assertRaises(ValueError):
            integration.state_difference(before, {"head.weight": TensorData((1, 2), (float("nan"), 2.))})


if __name__ == "__main__":
    unittest.main()
