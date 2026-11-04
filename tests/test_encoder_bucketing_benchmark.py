"""Study guard/summary fixtures; no models or benchmark queries are read."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import benchmark_encoder_bucketing as benchmark


@contextmanager
def prepared():
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        source = root / "source.py"
        source.write_text("# fixture\n", encoding="utf-8")
        model = {"head_hash": "fixture-head", "files": {}}
        with patch.object(benchmark, "ROOT", root), patch.object(benchmark.base, "ROOT", root), \
             patch.object(benchmark.base, "model_contract", return_value=model), \
             patch.object(benchmark, "runtime", return_value={"fixture": True}), \
             patch.object(benchmark, "input_files", side_effect=lambda _model: {"source.py": benchmark.base.sha(source)}):
            output, public = root / "runs/study", root / "research/results/study"
            plan = benchmark.prepare(output, public)
            yield root, output, public, plan


def observations(plan):
    rows = []
    for c in plan["cells"]:
        count = benchmark.DESIGN["cohorts"][c["role"]]
        rows.append({"cell": c, "lock_hash": plan["lock_hash"], "count": count,
             "head_hash": plan["model"]["head_hash"], "effective_torch_threads": 1,
             "source_ids_hash": benchmark.base.digest(c["role"]),
             "observations": {"original": {"seconds": 10.0, "padded_token_slots": count*20, "attention_cells": count*400},
                              "bucketed": {"seconds": 6.0, "padded_token_slots": count*10, "attention_cells": count*100}},
             "work": {"query_count": count, "batch_size": 8, "baseline_padded_token_slots": count*20,
                      "candidate_padded_token_slots": count*10, "baseline_attention_cells": count*400,
                      "candidate_attention_cells": count*100},
             "parity": {"passed": True, "max_feature_absolute_difference": 0., "max_logit_absolute_difference": 0.,
                        "prediction_changes": 0, "strict_margin_answer_changes": 0,
                        "truncations": {"original": 0, "bucketed": 0}, "observed_work_matches_arithmetic": True}})
    return rows


class BucketingBenchmarkTests(unittest.TestCase):
    def test_fixed_full_cohorts_and_order_counterbalance(self):
        cells = benchmark.cells()
        self.assertEqual(len(cells), 6)
        self.assertEqual(sum(c["order"][0] == "original" for c in cells), 3)
        self.assertEqual(benchmark.DESIGN["cohorts"], {"server": 200, "4": 408})
        for role in ("server", "4"):
            self.assertEqual(sorted(c["repeat"] for c in cells if c["role"] == role), [0, 1, 2])

    def test_lock_binds_exact_sources_and_has_no_overwrite(self):
        with prepared() as (root, output, public, plan):
            self.assertEqual(benchmark.verify(output), plan)
            with self.assertRaises(ValueError):
                benchmark.prepare(output, public)
            (root / "source.py").write_text("# changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract changed"):
                benchmark.verify(output)

    def test_rehashed_lock_cannot_drop_required_sources(self):
        with prepared() as (_, output, public, plan):
            plan["files"] = {}
            plan["lock_hash"] = benchmark.base.digest({k:v for k,v in plan.items() if k != "lock_hash"})
            for directory in (output, public):
                (directory / "plan.json").unlink()
                benchmark.base.write_new(directory / "plan.json", plan)
            with self.assertRaisesRegex(ValueError, "contract changed"):
                benchmark.verify(output)

    def test_environment_changes_refused(self):
        with prepared() as (_, output, _, _), patch.object(benchmark, "runtime", return_value={"fixture": False}):
            with self.assertRaisesRegex(ValueError, "environment changed"):
                benchmark.verify(output)

    def test_summary_keeps_all_cohorts_and_repetitions(self):
        with prepared() as (_, _, _, plan):
            rows = observations(plan)
            result = benchmark.summarize(plan, rows)
            for role in benchmark.DESIGN["cohorts"]:
                group = result["cohorts"][role]
                self.assertEqual(group["median_seconds"], {"original": 10.0, "bucketed": 6.0})
                self.assertAlmostEqual(group["median_time_reduction_fraction"], 0.4)
                self.assertEqual(len(group["paired_speedups"]), 3)
            for bad in (rows[:-1], rows[:-1] + [rows[0]]):
                with self.assertRaises(ValueError):
                    benchmark.summarize(plan, bad)

    def test_mixed_identity_or_wrong_thread_count_rejected(self):
        with prepared() as (_, _, _, plan):
            for key, value in (("lock_hash", "changed"), ("head_hash", "wrong-head"),
                               ("effective_torch_threads", 2), ("count", 999)):
                rows = observations(plan)
                rows[0][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    benchmark.summarize(plan, rows)

    def test_failed_parity_and_slower_outcomes_are_preserved(self):
        with prepared() as (_, _, _, plan):
            rows = observations(plan)
            for row in rows:
                row["observations"]["bucketed"]["seconds"] = 12.0
                row["parity"]["passed"] = False
                row["parity"]["prediction_changes"] = 1
            result = benchmark.summarize(plan, rows)
            self.assertFalse(result["cohorts"]["server"]["all_parity_checks_passed"])
            self.assertLess(result["cohorts"]["server"]["median_time_reduction_fraction"], 0)

    def test_invalid_timing_never_reported_as_speedup(self):
        with prepared() as (_, _, _, plan):
            for duration in (float("nan"), float("inf"), 0, -1, True):
                rows = observations(plan)
                rows[0]["observations"]["bucketed"]["seconds"] = duration
                with self.subTest(duration=duration), self.assertRaises(ValueError):
                    benchmark.summarize(plan, rows)

    def test_contradictory_parity_or_missing_work_is_rejected(self):
        with prepared() as (_, _, _, plan):
            for change in ("false_success", "missing_work", "mixed_order", "invalid_difference"):
                rows = observations(plan)
                if change == "false_success":
                    rows[0]["parity"]["prediction_changes"] = 5
                elif change == "missing_work":
                    rows[0]["work"] = {}
                elif change == "mixed_order":
                    rows[0]["source_ids_hash"] = "different"
                else:
                    rows[0]["parity"]["max_feature_absolute_difference"] = float("nan")
                with self.subTest(change=change), self.assertRaises(ValueError):
                    benchmark.summarize(plan, rows)


if __name__ == "__main__":
    unittest.main()
