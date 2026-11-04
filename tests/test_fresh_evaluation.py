"""Synthetic lifecycle/metric fixtures; no real queries or model inference."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import evaluate_fresh_bank20 as evaluate
from tools import fresh_query_collection as collection
from test_fresh_query_collection import fixture


@contextmanager
def prepared_fixture():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pack = root / "runs/pack"
        collection.create_pack(pack, "fixture-batch")
        data = fixture(20)
        collection.write_json(pack / "collection-plan.json", data[0])
        for name, fields, rows in (
                ("queries", collection.QUERY_FIELDS, data[1]),
                ("participants", collection.PARTICIPANT_FIELDS, data[2]),
                ("labels-a", collection.LABEL_FIELDS, data[3]),
                ("labels-b", collection.LABEL_FIELDS, data[4]),
                ("adjudications", collection.ADJUDICATION_FIELDS, data[5])):
            collection.write_csv(pack / f"{name}.csv", fields, rows)
        sources = []
        for category in collection.OVERLAP_CATEGORIES:
            path = pack / f"old-{category}.json"
            collection.write_json(path, [{"query": f"Previously known unrelated synthetic source {category}"}])
            sources.append({"dataset_id": category, "category": category, "path": path.name,
                            "text_field": "query", "records_key": "records"})
        collection.write_json(pack / "overlap-sources.json", sources)
        source = root / "tools/fixture.py"
        source.parent.mkdir()
        source.write_text("# Synthetic source\n", encoding="utf-8")
        model = {"head_hash": "fixture-head", "files": {"tools/fixture.py": evaluate.sha(source)}}
        with patch.object(evaluate, "ROOT", root), patch.object(evaluate, "model_contract", return_value=model), \
             patch.object(evaluate, "runtime", return_value={"fixture_environment": True}):
            output, public = root / "runs/evaluation", root / "research/results/fresh-fixture"
            plan = evaluate.prepare(pack, output, public)
            yield root, pack, output, public, plan


def fake_git(root, commit="a" * 40, remote_commit=None):
    def command(*args):
        if args[:2] == ("remote", "get-url"):
            return b"https://github.com/sameerreddy789/loflwr.git\n"
        if args[0] == "ls-remote":
            return f"{remote_commit or commit}\trefs/heads/main\n".encode()
        if args[0] == "show":
            requested, name = args[1].split(":", 1)
            if requested != commit:
                raise ValueError("Unknown fixture commit")
            return (root / name).read_bytes()
        if args[0] == "merge-base":
            if args[2:] != (commit, commit):
                raise ValueError("Unpublished fixture commit is not a public ancestor")
            return b""
        raise AssertionError(args)
    return command


def metric_rows():
    result = []
    for index, (label, disposition, provenance) in enumerate((
            (evaluate.INTENTS[0], "supported", "independent_human"),
            (evaluate.INTENTS[1], "supported", "independent_human"),
            (evaluate.INTENTS[0], "secondary", "external"),
            ("unsupported", "secondary", "independent_human"),
            ("ambiguous", "secondary", "independent_human"),
            (evaluate.INTENTS[0], "excluded", "independent_human"))):
        result.append({"query_id": f"q-{index}", "author_id": f"p-{index % 2:04d}",
            "source_id": f"src-{index}",
            "final_label": label, "disposition": disposition, "provenance": provenance,
            "normalized_query_sha256": f"fixture-hash-{index}",
            "adjudication": {"exclusion_reason": "privacy" if disposition == "excluded" else ""}})
    return result


class FreshEvaluationTests(unittest.TestCase):
    def test_empty_collection_cannot_create_lock_or_load_model(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluate, "ROOT", Path(directory)), \
             patch.object(evaluate, "model_contract") as model:
            root = Path(directory)
            pack = root / "runs/empty"
            collection.create_pack(pack, "empty-fixture")
            with self.assertRaisesRegex(ValueError, "not ready"):
                evaluate.prepare(pack, root / "runs/output", root / "research/results/fresh")
            self.assertFalse((root / "runs/output").exists())
            model.assert_not_called()

    def test_complete_fixture_prepares_hash_only_plan_and_all_rows(self):
        with prepared_fixture() as (_, _, output, public, plan):
            verified, records = evaluate.verify(output)
            self.assertEqual(verified, plan)
            self.assertEqual(len(records), 400)
            self.assertEqual(plan["counts"]["supported"], 400)
            self.assertNotIn("Synthetic unit fixture", (public / "plan.json").read_text())
            self.assertFalse((output / "started.json").exists())
            with self.assertRaises(FileExistsError):
                evaluate.prepare(output, output, public)

    def test_lock_data_ontology_and_source_changes_are_rejected(self):
        for relative in ("plan.json", "records.json", "source", "ontology"):
            with self.subTest(path=relative), prepared_fixture() as (root, pack, output, _, _):
                target = (root / "tools/fixture.py" if relative == "source" else
                          pack / "ontology.json" if relative == "ontology" else output / relative)
                target.write_text(target.read_text() + " ", encoding="utf-8")
                if relative == "plan.json":
                    plan = evaluate.read(target)
                    plan["threshold"] = 0
                    target.write_text(json.dumps(plan), encoding="utf-8")
                with self.assertRaises(ValueError):
                    evaluate.verify(output)

    def test_rehashed_plan_cannot_drop_required_source_contract(self):
        with prepared_fixture() as (_, _, output, public, plan):
            plan["committed_files"] = {}
            plan["lock_hash"] = evaluate.digest({k: v for k, v in plan.items() if k != "lock_hash"})
            for folder in (output, public):
                collection.write_json(folder / "plan.json", plan)
            with self.assertRaisesRegex(ValueError, "contract"):
                evaluate.verify(output)

    def test_rehashed_snapshot_cannot_change_the_audited_label_or_counts(self):
        for change in ("label", "counts"):
            with self.subTest(change=change), prepared_fixture() as (_, _, output, public, plan):
                if change == "label":
                    records = evaluate.read(output / "records.json")
                    records[0]["final_label"] = evaluate.INTENTS[1]
                    collection.write_json(output / "records.json", records)
                    plan["records_sha256"] = evaluate.digest(records)
                    plan["record_commitments"] = [evaluate.digest(r) for r in records]
                    plan["files"][evaluate.relative(output / "records.json")] = evaluate.sha(output / "records.json")
                else:
                    plan["counts"]["supported"] = 999
                plan["lock_hash"] = evaluate.digest({k: v for k, v in plan.items() if k != "lock_hash"})
                for folder in (output, public):
                    collection.write_json(folder / "plan.json", plan)
                with self.assertRaisesRegex(ValueError, "audited collection"):
                    evaluate.verify(output)

    def test_publication_checks_remote_and_exact_committed_bytes(self):
        with prepared_fixture() as (root, _, output, _, plan):
            with patch.object(evaluate, "git", side_effect=fake_git(root, remote_commit="b" * 40)):
                with self.assertRaisesRegex(ValueError, "remote main"):
                    evaluate.attest_publication(output, "a" * 40)
            self.assertFalse((output / "publication.json").exists())
            with patch.object(evaluate, "git", side_effect=fake_git(root)):
                receipt = evaluate.attest_publication(output, "a" * 40)
                self.assertEqual(evaluate.verify_receipt(output, plan), receipt)
            def wrong(*args):
                return b"changed source" if args[0] == "show" else fake_git(root)(*args)
            with patch.object(evaluate, "git", side_effect=wrong):
                with self.assertRaisesRegex(ValueError, "Committed"):
                    evaluate.verify_receipt(output, plan)

    def test_caller_written_publication_boolean_cannot_authorize_inference(self):
        with prepared_fixture() as (_, _, output, _, plan):
            collection.write_json(output / "publication.json", {"published": True})
            with self.assertRaises(ValueError):
                evaluate.verify_receipt(output, plan)

    def test_offline_or_unpublished_local_commit_cannot_claim_inference(self):
        with prepared_fixture() as (root, _, output, _, _), patch.object(evaluate, "predict") as predict:
            with patch.object(evaluate, "git", side_effect=fake_git(root)):
                evaluate.attest_publication(output, "a" * 40)
            # The local commit and receipt still validate, but it is not in the
            # freshly observed remote history. No model or shared claim starts.
            with patch.object(evaluate, "git", side_effect=fake_git(root, remote_commit="b" * 40)):
                with self.assertRaisesRegex(ValueError, "public ancestor"):
                    evaluate.score(output)
            self.assertFalse((output / "started.json").exists())
            predict.assert_not_called()

    def test_exclusive_claim_prevents_new_output_and_changed_collection_rescoring(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluate, "ROOT", Path(directory)):
            root = Path(directory)
            first, second = root / "runs/first", root / "runs/second"
            first.mkdir(parents=True); second.mkdir()
            plan = {"lock_hash": "fixture-lock", "collection_sha256": "fixture-collection"}
            evaluate.claim_inference(first, plan, metric_rows())
            with self.assertRaises(FileExistsError):
                evaluate.claim_inference(first, plan, metric_rows())
            with self.assertRaises(FileExistsError):
                evaluate.claim_inference(second, {**plan, "collection_sha256": "changed-metadata"}, metric_rows())

    def test_missing_receipt_and_post_claim_failure_never_run_or_retry_model(self):
        with prepared_fixture() as (root, _, output, _, _), patch.object(evaluate, "predict") as predict:
            with self.assertRaises(FileNotFoundError):
                evaluate.score(output)
            predict.assert_not_called()
            with patch.object(evaluate, "git", side_effect=fake_git(root)):
                evaluate.attest_publication(output, "a" * 40)
                predict.side_effect = RuntimeError("Synthetic model failure")
                with self.assertRaises(RuntimeError):
                    evaluate.score(output)
                self.assertTrue((output / "started.json").exists())
                self.assertTrue((output / "failure.json").exists())
                with self.assertRaises(FileExistsError):
                    evaluate.score(output)
                self.assertEqual(predict.call_count, 1)

    def test_report_only_reuses_bound_predictions_without_inference(self):
        with prepared_fixture() as (root, _, output, public, _), \
             patch.object(evaluate, "git", side_effect=fake_git(root)):
            evaluate.attest_publication(output, "a" * 40)
            def prediction(plan, rows):
                return [[4.0 if label == r["final_label"] else 0.0 for label in evaluate.INTENTS] for r in rows], 0
            with patch.object(evaluate, "predict", side_effect=prediction) as predict:
                result = evaluate.score(output)
                def offline_git(*args):
                    if args[0] in {"ls-remote", "remote"}:
                        raise AssertionError("Report-only must not contact or inspect a live remote")
                    return fake_git(root)(*args)
                with patch.object(evaluate, "git", side_effect=offline_git):
                    self.assertEqual(evaluate.report(output), result)
                self.assertEqual(predict.call_count, 1)
            self.assertEqual(result["primary_human_supported"]["accuracy"], 1.0)
            self.assertNotIn("Synthetic unit fixture", (public / "result.json").read_text())
            saved = evaluate.read(output / "predictions.json")
            saved["predictions"][0]["answered"] = False
            collection.write_json(output / "predictions.json", saved)
            with self.assertRaises(ValueError):
                evaluate.report(output)

    def test_metric_denominators_keep_declines_external_rows_and_exclusions_separate(self):
        records = metric_rows()
        predictions = [{"query_id": r["query_id"], "prediction": evaluate.INTENTS[0],
                        "margin": 0.0 if index in (0, 4) else 3.0, "answered": index not in (0, 4)}
                       for index, r in enumerate(records[:-1])]
        result = evaluate.summarize(records, predictions)
        primary = result["primary_human_supported"]
        self.assertEqual((primary["count"], primary["correct"], primary["accuracy"]), (2, 1, 0.5))
        self.assertEqual(primary["selective"]["wrong_answered"], 1)
        self.assertEqual(primary["selective"]["declined"], 1)
        self.assertEqual(primary["selective"]["correct_answers_per_supported_query"], 0)
        self.assertEqual(result["secondary_by_provenance"]["external"]["supported_labels"]["count"], 1)
        self.assertEqual(result["secondary_by_provenance"]["independent_human"]["unsupported"]["answered"], 1)
        self.assertEqual(result["accounting"]["scored"], 5)
        self.assertEqual(result["accounting"]["excluded_reasons"], {"privacy": 1})

    def test_invalid_scores_ids_nonfinite_and_threshold_boundary(self):
        records = metric_rows()[:1]
        row = [0.0] * 20
        row[0] = evaluate.THRESHOLD
        self.assertTrue(evaluate.predictions_from_scores(records, [row])[0]["answered"])
        self.assertEqual(evaluate.predictions_from_scores(records, [[0.0] * 20])[0]["prediction"], evaluate.INTENTS[0])
        for scores in ([], [[0.0] * 19], [[float("nan")] * 20], [[True] * 20], [[1e308] + [-1e308] * 19]):
            with self.subTest(scores=str(scores)[:20]), self.assertRaises(ValueError):
                evaluate.predictions_from_scores(records, scores)
        with self.assertRaises(ValueError):
            evaluate.summarize(records, [{"query_id": "wrong", "prediction": evaluate.INTENTS[0], "margin": 0.0, "answered": False}])

    def test_frozen_fp32_margin_rounding_at_acceptance_boundary(self):
        row = [0.0] * 20
        row[0] = evaluate.THRESHOLD
        row[1] = evaluate.float32(1e-8)
        self.assertLess(row[0] - row[1], evaluate.THRESHOLD)
        prediction = evaluate.predictions_from_scores(metric_rows()[:1], [row])[0]
        self.assertEqual(prediction["margin"], evaluate.THRESHOLD)
        self.assertTrue(prediction["answered"])
        with prepared_fixture() as (_, _, output, public, plan):
            plan["numerics"] = "float64 subtraction"
            plan["lock_hash"] = evaluate.digest({k: v for k, v in plan.items() if k != "lock_hash"})
            for folder in (output, public):
                collection.write_json(folder / "plan.json", plan)
            with self.assertRaisesRegex(ValueError, "fixed design"):
                evaluate.verify(output)


if __name__ == "__main__":
    unittest.main()
