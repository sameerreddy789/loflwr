import copy
from pathlib import Path
import tempfile
import unittest

from fedlora_edge.data import content_hash, write_json
from fedlora_edge.encoder_data import sha256
from fedlora_edge.state import TensorData, hash_state, state_to_dict
from tools.run_encoder_matrix import HEAD as MATRIX_HEAD
from tools.run_resource_frontier import (
    CHECKPOINTS, CLIENTS, DATASET, HEAD, SEEDS, SOURCES, cells, dominates, payload_bytes,
    stats, validate_design, validate_result, validate_work, verify,
)


def fixture_plan():
    return {"kind": "resource_frontier_v1", "cells": cells(), "head_config": copy.deepcopy(HEAD),
            "dataset": DATASET, "strategy": "equal", "checkpoints_steps_per_client": list(CHECKPOINTS),
            "steps_per_client": 500, "split": "reused_validation", "workers": 1,
            "client_counts": {c: 4 for c in CLIENTS},
            "initial": {str(seed): {"state_hash": str(seed)} for seed in SEEDS}, "plan_hash": "fixture"}


class ResourceDesignTests(unittest.TestCase):
    def test_all_seeds_have_matched_optimizer_and_example_work(self):
        plan = fixture_plan()
        validate_design(plan)
        self.assertEqual(HEAD, MATRIX_HEAD)
        self.assertEqual(len(cells()), 9)
        self.assertEqual(len({r["id"] for r in cells()}), 9)
        for seed in SEEDS:
            rows = [r for r in cells() if r["seed"] == seed]
            self.assertEqual({r["local_steps"] for r in rows}, {1, 2, 4})
            self.assertEqual({r["rounds"] * r["local_steps"] * 5 for r in rows}, {2500})
            for row in rows:
                self.assertTrue(all(work % row["local_steps"] == 0 for work in CHECKPOINTS))
        self.assertEqual([r["local_steps"] for r in cells()[::3]], [1, 2, 4])

    def test_design_rejects_unmatched_or_unpermitted_configuration(self):
        mutations = (
            lambda p: p.update(split="confirmation"),
            lambda p: p.update(strategy="sample"),
            lambda p: p.update(workers=2),
            lambda p: p["head_config"].update(cpu_threads=2),
            lambda p: p["head_config"].update(batch_size=2),
            lambda p: p["cells"][0].update(rounds=499),
            lambda p: p["cells"][0].update(local_steps=True),
            lambda p: p["client_counts"].update({"0": 1025}),
            lambda p: p["client_counts"].update({"0": True}),
            lambda p: p["initial"].pop("43"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                plan = fixture_plan()
                mutation(plan)
                with self.assertRaises(ValueError):
                    validate_design(plan)

    def test_payload_is_two_directions_and_rejects_invalid_counts(self):
        self.assertEqual(payload_bytes(61520, 500), {
            "estimated_download_bytes": 153800000, "estimated_upload_bytes": 153800000,
            "estimated_total_bytes": 307600000})
        self.assertEqual(payload_bytes(61520, 125)["estimated_total_bytes"], 76900000)
        self.assertEqual(payload_bytes(61520, 0)["estimated_total_bytes"], 0)
        for args in ((True, 1), (0, 1), (1, -1), (1, 1.5), (1, 1, 0), (1, 1, False)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                payload_bytes(*args)

    def test_actual_work_must_match_full_batch_budget(self):
        telemetry = {"optimizer_steps": 4, "completed_optimizer_steps": 4,
                     "examples_processed": 32, "manifest_examples": 8}
        validate_work(telemetry, 4, 8)
        for key in telemetry:
            altered = {**telemetry, key: telemetry[key] - 1}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_work(altered, 4, 8)

    def test_incomplete_and_mixed_evidence_is_rejected(self):
        plan = fixture_plan()
        cell = cells()[0]
        result = {"status": "completed", "plan_hash": "fixture", "cell": copy.deepcopy(cell),
                  "initial_state_hash": "42", "completed_steps": {c: 500 for c in CLIENTS},
                  "examples_processed": {c: 2000 for c in CLIENTS},
                  "participation": {c: 500 for c in CLIENTS},
                  "history": [{"steps_per_client": n} for n in CHECKPOINTS]}
        validate_result(result, plan, cell)
        for mutation in (
            lambda r: r.update(plan_hash="other"),
            lambda r: r.update(initial_state_hash="43"),
            lambda r: r["cell"].update(local_steps=2),
            lambda r: r["completed_steps"].update({"4": 499}),
            lambda r: r["examples_processed"].update({"4": 1999}),
            lambda r: r["history"].pop(1),
        ):
            changed = copy.deepcopy(result)
            mutation(changed)
            with self.assertRaises(ValueError):
                validate_result(changed, plan, cell)

    def test_pareto_keeps_quality_cost_tradeoffs_and_requires_strict_improvement(self):
        quality = {"accuracy": .95, "bytes": 300}
        cheap = {"accuracy": .90, "bytes": 75}
        worse = {"accuracy": .90, "bytes": 400}
        self.assertFalse(dominates(quality, cheap, ("accuracy",), ("bytes",)))
        self.assertFalse(dominates(cheap, quality, ("accuracy",), ("bytes",)))
        self.assertTrue(dominates(quality, worse, ("accuracy",), ("bytes",)))
        self.assertFalse(dominates(quality, quality, ("accuracy",), ("bytes",)))
        self.assertTrue(dominates(cheap, {**cheap, "bytes": 76}, ("accuracy",), ("bytes",)))

    def test_statistics_use_sample_standard_deviation(self):
        self.assertEqual(stats([1, 2, 3]), {"mean": 2, "sample_std": 1, "values": [1, 2, 3]})
        self.assertIsNone(stats([2])["sample_std"])
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                stats(values)

    def test_frozen_plan_rejects_source_public_plan_and_initial_tensor_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, public = root / "output", root / "public"
            output.mkdir()
            public.mkdir()
            plan = fixture_plan()
            plan["files_sha256"] = {}
            for name in SOURCES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture source\n")
                plan["files_sha256"][name] = sha256(path)
            state = {"fixture": TensorData((1,), (1.0,))}
            path = output / "initial.json"
            write_json(path, {"state": state_to_dict(state), "state_hash": hash_state(state)})
            plan["initial"] = {str(seed): {"file": path.name, "sha256": sha256(path), "state_hash": hash_state(state)} for seed in SEEDS}
            plan.pop("plan_hash")
            plan["plan_hash"] = content_hash(plan)
            write_json(public / "plan.json", plan)
            verify(plan, output, public, root=root, check_runtime=False)
            original = path.read_text()
            path.write_text(original + " ")
            with self.assertRaisesRegex(ValueError, "Initial tensor file"):
                verify(plan, output, public, root=root, check_runtime=False)
            path.write_text(original)
            source = root / SOURCES[0]
            source.write_text("altered source\n")
            with self.assertRaisesRegex(ValueError, "Source or validation/training data"):
                verify(plan, output, public, root=root, check_runtime=False)
            source.write_text("fixture source\n")
            changed = copy.deepcopy(plan)
            changed["workers"] = 2
            with self.assertRaisesRegex(ValueError, "plan hash"):
                verify(changed, output, public, root=root, check_runtime=False)
            write_json(public / "plan.json", changed)
            with self.assertRaisesRegex(ValueError, "Public plan differs"):
                verify(plan, output, public, root=root, check_runtime=False)


if __name__ == "__main__":
    unittest.main()
