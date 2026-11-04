"""Synthetic paired-evaluation fixtures, never independent accuracy evidence."""
from contextlib import contextmanager
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import evaluate_fresh_scope as scope
from tools import evaluate_fresh_bank20 as base
from tools import fresh_query_collection as collection
from test_fresh_evaluation import prepared_fixture, fake_git, metric_rows


def add_negatives(pack, count=200, provenance="independent_human"):
    names = (("queries", collection.QUERY_FIELDS), ("labels-a", collection.LABEL_FIELDS),
             ("labels-b", collection.LABEL_FIELDS), ("adjudications", collection.ADJUDICATION_FIELDS))
    data = {name: collection.read_rows(pack / f"{name}.csv", fields) for name, fields in names}
    for index in range(count):
        qid = f"q-{1000 + index:04d}"
        query = f"Synthetic unscored scope fixture other topic {index}"
        data["queries"].append({**data["queries"][0], "query_id": qid, "source_id": f"src-{1000 + index:04d}",
                                "query": query, "provenance": provenance,
                                "author_id": "p-0001" if provenance == "independent_human" else "p-0005",
                                "provenance_detail": "Synthetic test only"})
        for name in ("labels-a", "labels-b"):
            data[name].append({**data[name][0], "query_id": qid, "query": query, "label": "unsupported"})
        data["adjudications"].append({**data["adjudications"][0], "query_id": qid, "final_label": "unsupported"})
    for name, fields in names:
        collection.write_csv(pack / f"{name}.csv", fields, data[name])


@contextmanager
def prepared_scope():
    with prepared_fixture() as (root, pack, _, _, base_plan):
        add_negatives(pack)
        model = {"head": {"head_hash": "fixture-head"}, "scope_sha256": "fixture-scope",
                 "files": base_plan["model"]["files"]}
        with patch.object(scope, "ROOT", root), patch.object(scope, "model_contract", return_value=model), \
             patch.object(scope, "runtime", return_value={"fixture_environment": True}):
            output, public = root / "runs/paired", root / "research/results/paired"
            plan = scope.prepare(pack, output, public)
            yield root, pack, output, public, plan


def fake_scope_model():
    return {"coef": [0.0] * 768, "intercept": scope.SCOPE_THRESHOLD,
            "threshold": scope.SCOPE_THRESHOLD, "encoder_identity_hash": scope.ENCODER_ID}


def fake_scores(records):
    rows = [r for r in records if r["disposition"] != "excluded"]
    logits = [[3.0] + [0.0] * 19 for _ in rows]
    return logits, [scope.SCOPE_THRESHOLD] * len(rows), 0


class FreshScopeEvaluationTests(unittest.TestCase):
    def test_empty_pack_cannot_lock_or_load_model(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            pack = root / "runs/empty"
            collection.create_pack(pack, "empty-fixture")
            with patch.object(base, "ROOT", root), patch.object(scope, "ROOT", root), \
                 patch.object(scope, "model_contract") as model:
                with self.assertRaisesRegex(ValueError, "not ready"):
                    scope.prepare(pack, root / "runs/result", root / "research/results/fresh")
                model.assert_not_called()
                self.assertFalse((root / "runs/result").exists())

    def test_positive_only_and_external_negatives_fail_scope_minimum(self):
        for provenance in (None, "external", "model_generated"):
            with self.subTest(provenance=provenance), prepared_fixture() as (root, pack, _, _, _), \
                 patch.object(scope, "ROOT", root), patch.object(scope, "model_contract") as model:
                if provenance:
                    add_negatives(pack, provenance=provenance)
                with self.assertRaisesRegex(ValueError, "200 reviewed independent-human"):
                    scope.prepare(pack, root / "runs/new", root / "research/results/new")
                model.assert_not_called()
                self.assertFalse((root / "runs/new").exists())

    def test_negative_author_attestations_and_blinded_review_are_required(self):
        with prepared_fixture() as (root, pack, _, _, _), patch.object(scope, "ROOT", root):
            add_negatives(pack)
            people = collection.read_rows(pack / "participants.csv", collection.PARTICIPANT_FIELDS)
            people.append({**people[0], "participant_id": "p-0006", "no_model_outputs": "false"})
            collection.write_csv(pack / "participants.csv", collection.PARTICIPANT_FIELDS, people)
            queries = collection.read_rows(pack / "queries.csv", collection.QUERY_FIELDS)
            queries[-1]["author_id"] = "p-0006"
            collection.write_csv(pack / "queries.csv", collection.QUERY_FIELDS, queries)
            with self.assertRaisesRegex(ValueError, "not ready"):
                scope.prepare(pack, root / "runs/new", root / "research/results/new")

    def test_complete_fixture_lock_is_hash_only_and_both_policies_fixed(self):
        with prepared_scope() as (_, _, output, public, plan):
            actual, records = scope.verify(output)
            self.assertEqual(actual, plan)
            self.assertEqual(len(records), 600)
            self.assertEqual(plan["cohorts"]["human_unsupported"], 200)
            self.assertEqual(plan["cohorts"]["human_supported"], 400)
            self.assertNotIn("Synthetic unscored", (public / "plan.json").read_text())
            self.assertEqual(plan["design"], scope.DESIGN)
            self.assertFalse((output / "started.json").exists())

    def test_rehashed_label_model_source_and_cohort_changes_are_rejected(self):
        for change in ("label", "cohort", "contract", "threshold"):
            with self.subTest(change=change), prepared_scope() as (_, _, output, public, plan):
                if change == "label":
                    records = base.read(output / "records.json")
                    records[0]["final_label"] = base.INTENTS[-1]
                    collection.write_json(output / "records.json", records)
                    plan["files"][base.relative(output / "records.json")] = base.sha(output / "records.json")
                    plan["records_sha256"] = base.digest(records)
                    plan["record_commitments"] = [base.digest(r) for r in records]
                elif change == "cohort":
                    plan["cohorts"]["human_unsupported"] = 999
                elif change == "contract":
                    plan["committed_files"] = {}
                else:
                    # JSON round-trip detaches the runtime constant from the plan.
                    plan = base.read(output / "plan.json")
                    plan["design"]["combined"] = "answer everything"
                plan["lock_hash"] = base.digest({k: v for k, v in plan.items() if k != "lock_hash"})
                for folder in (output, public):
                    collection.write_json(folder / "plan.json", plan)
                with self.assertRaises(ValueError):
                    scope.verify(output)

    def test_live_publication_failure_prevents_query_claim_or_model_load(self):
        with prepared_scope() as (root, _, output, _, _), patch.object(scope, "predict") as predict:
            with patch.object(base, "git", side_effect=fake_git(root)):
                scope.attest_publication(output, "a" * 40)
            with patch.object(base, "git", side_effect=fake_git(root, remote_commit="b" * 40)):
                with self.assertRaisesRegex(ValueError, "public ancestor"):
                    scope.score(output)
            self.assertFalse((output / "started.json").exists())
            self.assertFalse((root / "runs/fresh-evaluation-exposures").exists())
            predict.assert_not_called()

    def test_cross_runner_claims_cannot_be_bypassed_with_new_collection_or_path(self):
        for first in ("base", "scope"):
            with self.subTest(first=first), prepared_scope() as (root, _, output, _, plan):
                _, records = scope.verify(output)
                first_output = root / "runs/first"
                first_output.mkdir()
                first_plan = {**plan, "collection_sha256": "another-metadata-hash"}
                base.claim_inference(first_output, first_plan, records)
                with patch.object(base, "git", side_effect=fake_git(root)), patch.object(scope, "predict") as predict:
                    scope.attest_publication(output, "a" * 40)
                    with self.assertRaises(FileExistsError):
                        if first == "base":
                            scope.score(output)
                        else:
                            base.claim_inference(output, plan, records)
                    predict.assert_not_called()
                self.assertFalse((output / "started.json").exists())

    def test_score_once_reports_both_arms_without_repeated_encoding(self):
        with prepared_scope() as (root, _, output, _, plan), patch.object(base, "git", side_effect=fake_git(root)):
            scope.attest_publication(output, "a" * 40)
            with patch.object(scope, "predict", side_effect=lambda _p, rows: fake_scores(rows)) as predict:
                result = scope.score(output)
                predict.assert_called_once()
                self.assertEqual(result, scope.report(output))
                with self.assertRaises(FileExistsError):
                    scope.score(output)
                predict.assert_called_once()
            self.assertEqual(result["primary_human_supported"]["baseline_margin"]["count"], 400)
            self.assertEqual(result["primary_human_unsupported"]["baseline_margin"]["count"], 200)
            self.assertEqual(result["lock_hash"], plan["lock_hash"])

    def test_failed_inference_is_permanently_claimed(self):
        with prepared_scope() as (root, _, output, _, _), patch.object(base, "git", side_effect=fake_git(root)):
            scope.attest_publication(output, "a" * 40)
            with patch.object(scope, "predict", side_effect=RuntimeError("Synthetic failure")):
                with self.assertRaises(RuntimeError):
                    scope.score(output)
                with self.assertRaises(FileExistsError):
                    scope.score(output)
            self.assertTrue((output / "failure.json").exists())
            self.assertFalse((output / "predictions.json").exists())

    def test_fp32_head_and_fp64_scope_boundaries(self):
        rows = metric_rows()[:2]
        logits = [[base.THRESHOLD, base.float32(1e-8)] + [0.0] * 18 for _ in rows]
        scores = [scope.SCOPE_THRESHOLD, math.nextafter(scope.SCOPE_THRESHOLD, -math.inf)]
        predictions = scope.predictions_from_scores(rows, logits, scores)
        self.assertEqual([r["answered"] for r in predictions], [True, True])
        self.assertEqual([r["combined_answered"] for r in predictions], [True, False])

    def test_scope_vector_shape_precision_and_finite_checks(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy optional dependency is not installed")
        features = np.zeros((2, 768), dtype=np.float32)
        features[:, 0] = 1.0
        model = fake_scope_model()
        model["coef"][0] = 1e-10
        actual = scope.scores_from_features(features, model)
        expected = features @ np.array(model["coef"]) + model["intercept"]
        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_array_equal(actual, expected)
        for bad in (features.astype(np.float64), features[:, :-1], np.full_like(features, np.nan), features * 2):
            with self.assertRaises(ValueError):
                scope.scores_from_features(bad, model)
        for field, value in (("coef", [0.0]), ("intercept", float("nan")), ("threshold", 0),
                             ("encoder_identity_hash", "wrong")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                scope.validate_scope_model({**model, field: value})

    def test_paired_denominators_and_provenance_are_separate(self):
        records = metric_rows()
        logits = [[3.0] + [0.0] * 19 for _ in range(5)]
        predictions = scope.predictions_from_scores(records, logits, [scope.SCOPE_THRESHOLD - 1] * 5)
        result = scope.summarize(records, predictions)
        primary = result["primary_human_supported"]
        self.assertEqual(primary["baseline_margin"]["count"], 2)
        self.assertEqual(primary["paired"]["correct_answers_lost"], 1)
        self.assertEqual(primary["paired"]["wrong_answers_removed"], 1)
        self.assertEqual(primary["combined_margin_scope"]["selective"]["coverage"], 0)
        self.assertEqual(primary["baseline_margin"]["accuracy"], primary["combined_margin_scope"]["accuracy"])
        self.assertEqual(result["primary_human_unsupported"]["paired"]["false_accepts_removed"], 1)
        self.assertEqual(result["secondary_by_provenance"]["external"]["supported"]["baseline_margin"]["count"], 1)
        self.assertEqual(result["secondary_by_provenance"]["independent_human"]["ambiguous"]["paired"]["answers_removed"], 1)
        self.assertEqual(result["accounting"]["submitted"], 6)
        domains = result["primary_human_supported_by_domain"]
        self.assertEqual(sum(d["count"] for d in domains.values()), 2)
        self.assertEqual(sum(d["paired"]["correct_answers_lost"] for d in domains.values()), 1)
        self.assertEqual(sum(d["paired"]["wrong_answers_removed"] for d in domains.values()), 1)
        self.assertTrue(all(d["combined_margin_scope"]["coverage"] == 0 for d in domains.values() if d["count"]))

    def test_prediction_tampering_or_nonfinite_values_are_rejected(self):
        records = metric_rows()
        logits = [[3.0] + [0.0] * 19 for _ in range(5)]
        for field, value in (("scope_score", float("nan")), ("combined_answered", False),
                             ("scope_passed", False), ("final_label", base.INTENTS[0]),
                             ("query_id", "wrong-id")):
            predictions = scope.predictions_from_scores(records, logits, [scope.SCOPE_THRESHOLD] * 5)
            predictions[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                scope.summarize(records, predictions)
        with self.assertRaises(ValueError):
            scope.predictions_from_scores(records, logits, [0.0])


if __name__ == "__main__":
    unittest.main()
