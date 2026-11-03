"""Evidence reader invariants: no invented quality and no future replay leakage."""

import json
import tempfile
import unittest
from pathlib import Path

from fedlora_edge.results import (
    attempt_rows, build_report, client_rows, discover_runs, load_run,
    report_markdown, work_totals,
)


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test-run"
        self.path.mkdir()

    def save(self, name, value):
        (self.path / name).write_text(json.dumps(value), encoding="utf-8")

    def events(self, values, tail=""):
        (self.path / "events.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in values) + tail, encoding="utf-8"
        )

    def test_empty_run_has_no_quality_or_work(self):
        run = load_run(self.path)
        self.assertEqual(run.evaluations, [])
        self.assertIsNone(work_totals(run)["optimizer_steps"])
        self.assertIn("No model evaluation", report_markdown(build_report(run)))

    def test_partial_jsonl_preserves_complete_records(self):
        self.events([{"event": "attempt_started", "attempt_id": 1}], '{"event":')
        run = load_run(self.path)
        self.assertEqual(len(run.events), 1)
        self.assertEqual(len(run.warnings), 1)
        self.assertIn("writer", run.warnings[0])

    def test_rounds_are_not_counted_twice(self):
        result = {"event": "round_result", "attempt_id": 1, "status": "committed",
                  "version_before": 0, "version_after": 1, "accepted": [0, 1, 2]}
        self.save("round_results.json", [result])
        self.events([result])
        self.assertEqual(len(load_run(self.path).rounds), 1)

    def test_skipped_attempt_keeps_version_and_accepted_work(self):
        self.events([
            {"event": "attempt_started", "attempt_id": 1, "selected": [0, 1, 2], "version": 2},
            {"event": "reply_accepted", "attempt_id": 1, "client_id": 0,
             "telemetry": {"optimizer_steps": 2}},
            {"event": "round_result", "attempt_id": 1, "status": "skipped",
             "version_before": 2, "version_after": 2, "accepted": [0], "absent": [1, 2]},
        ])
        run = load_run(self.path)
        self.assertEqual(attempt_rows(run)[0]["Version after"], 2)
        self.assertEqual(work_totals(run)["optimizer_steps"], 2)
        self.assertEqual(client_rows(run)[1]["Status"], "Absent at deadline")

    def test_replay_excludes_future_summaries_checkpoint_and_round_file(self):
        self.save("summary.json", {"accuracy": 0.5})
        self.save("checkpoint.json", {"version": 4, "state": {}})
        self.save("round_results.json", [{"attempt_id": 2, "status": "committed"}])
        self.events([
            {"event": "attempt_started", "attempt_id": 1, "version": 0},
            {"event": "evaluation", "metrics": {"accuracy": 0.3}},
        ])
        replay = load_run(self.path, event_limit=1)
        self.assertEqual(replay.evaluations, [])
        self.assertEqual(replay.rounds, [])
        self.assertEqual(replay.checkpoint, {})
        self.assertEqual(replay.summary, {})

    def test_work_missing_fields_are_not_zero(self):
        self.events([
            {"event": "reply_accepted", "attempt_id": 1, "client_id": 0,
             "telemetry": {"optimizer_steps": 2}},
            {"event": "reply_accepted", "attempt_id": 1, "client_id": 1, "telemetry": {}},
        ])
        totals = work_totals(load_run(self.path))
        self.assertEqual(totals["optimizer_steps"], 2)
        self.assertIsNone(totals["processed_tokens"])
        self.assertEqual(totals["optimizer_steps_coverage"]["measured_replies"], 1)
        self.assertEqual(totals["optimizer_steps_coverage"]["accepted_replies"], 2)

    def test_report_preserves_configuration_and_source_hashes(self):
        config = {"run_id": "reader-test", "seed": 42, "strategy": "equal_median"}
        self.save("config.json", config)
        self.events([{"event": "evaluation", "split": "validation", "version": 0,
                      "metrics": {"accuracy": 0.25, "macro_f1": 0.2, "count": 4}}])
        report = build_report(load_run(self.path))
        self.assertEqual(report["configuration"], config)
        self.assertEqual(len(report["config_sha256"]), 64)
        self.assertEqual(report["evaluations"][0]["split"], "validation")
        self.assertEqual(len(report["source_files"]), 2)
        self.assertEqual(discover_runs(self.path.parent), [self.path])

    def test_checkpoint_tensor_values_excluded_from_report(self):
        self.save("checkpoint.json", {"version": 1, "b_hash": "test-hash", "state": {"B": [1, 2]}})
        report = build_report(load_run(self.path))
        self.assertNotIn("state", report["checkpoint"])
        self.assertEqual(report["checkpoint"]["version"], 1)

    def test_numeric_client_sizes_and_actual_trainer_telemetry(self):
        self.save("config.json", {"clients": {"0": 10}})
        self.events([
            {"event": "attempt_started", "attempt_id": 1, "selected": ["0"], "budgets": {"0": 2}},
            {"event": "requests_sent", "attempt_id": 1, "serialized_request_bytes": 100},
            {"event": "reply_accepted", "attempt_id": 1, "client_id": "0",
             "telemetry": {"optimizer_steps": 2, "examples_processed": 8,
                           "tokens_processed": 64, "local_train_seconds": 3.5}},
        ])
        run = load_run(self.path)
        row = client_rows(run)[0]
        self.assertEqual(row["Declared examples"], 10)
        self.assertEqual(row["Step budget"], 2)
        self.assertEqual(row["Training seconds"], 3.5)
        self.assertEqual(work_totals(run)["serialized_request_bytes"], 100)

    def test_generic_summary_count_is_not_a_model_evaluation(self):
        self.save("summary.json", {"count": 10})
        self.assertEqual(load_run(self.path).evaluations, [])

    def test_study_summary_does_not_hide_individual_runs_in_discovery(self):
        self.save("summary.json", {"study_id": "fixture", "runs": [{"key": "main"}]})
        child = self.path / "runs" / "main"
        child.mkdir(parents=True)
        (child / "config.json").write_text('{"run_id": "fixture-main"}', encoding="utf-8")
        self.assertEqual(discover_runs(self.path), [child])


if __name__ == "__main__":
    unittest.main()
