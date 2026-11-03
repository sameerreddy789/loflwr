"""External test fixtures contain no downloaded evaluation queries."""
import json
from pathlib import Path
import tempfile
import unittest

from fedlora_edge.data import INTENTS, content_hash
from tools.evaluate_external_clinc import (CLASSES, MAPPING, build_records, near_duplicate_audit,
                                         sha, summarize, verify_plan)


def fixture():
    return {"train": [], "val": [], "test": [[f"fixture {label} row {n}", label]
            for label in CLASSES for n in range(2)]}


class ExternalSelectionTests(unittest.TestCase):
    def test_predeclared_mapping_uses_test_only(self):
        source = fixture()
        source["train"] = [["training-only", "card_declined"]]
        source["test"].append(["unselected test topic", "transfer"])
        rows, audit = build_records(source, [], expected_per_class=2)
        self.assertEqual(len(rows), 18)
        self.assertEqual(audit["candidates"], 18)
        self.assertEqual({r["label"] for r in rows if r["label"]}, set(MAPPING.values()))
        self.assertTrue(all(r["source_split"] == "test" for r in rows))
        self.assertNotIn("training-only", [r["query"] for r in rows])

    def test_normalized_overlap_and_duplicate_exclusions(self):
        source = fixture()
        source["test"][3][0] = source["test"][2][0].upper()
        bank = [{"query": "  FIXTURE CARD_DECLINED ROW 0\n"}]
        rows, audit = build_records(source, bank, expected_per_class=2)
        self.assertEqual(len(rows), 16)
        self.assertEqual([e["reason"] for e in audit["exclusions"]],
                         ["matches_original_banking77", "duplicate_selected_test"])

    def test_conflicting_duplicates_fail_before_exclusion(self):
        source = fixture()
        source["test"][2][0] = source["test"][0][0]
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            build_records(source, [{"query": source["test"][0][0]}], expected_per_class=2)

    def test_malformed_missing_and_empty_classes_fail(self):
        for source in ({"test": []}, {**fixture(), "test": [["", "card_declined"]]},
                       {**fixture(), "test": fixture()["test"][:-1]}):
            with self.assertRaises(ValueError):
                build_records(source, [], expected_per_class=2)
        source = fixture()
        with self.assertRaisesRegex(ValueError, "empty"):
            build_records(source, [{"query": q} for q, label in source["test"] if label == "card_declined"],
                          expected_per_class=2)

    def test_near_duplicates_are_diagnostic_only(self):
        rows, _ = build_records(fixture(), [], expected_per_class=2)
        before = content_hash(rows)
        audit = near_duplicate_audit(rows, [{"query": rows[0]["query"], "source_id": "train:1"}])
        self.assertGreaterEqual(audit["flagged_count"], 1)
        self.assertEqual(before, content_hash(rows))

    def test_lock_rejects_modified_file_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "file.json").write_text("{}", encoding="utf-8")
            plan = {"locked_files": {"file.json": sha(root / "file.json")}}
            plan["lock_hash"] = content_hash(plan)
            verify_plan(plan, root)
            bad = json.loads(json.dumps(plan))
            bad["new_field"] = True
            with self.assertRaisesRegex(ValueError, "lock"):
                verify_plan(bad, root)
            (root / "file.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file changed"):
                verify_plan(plan, root)


class ExternalMetricTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy required for score fixtures")
        self.np = np
        self.rows, _ = build_records(fixture(), [], expected_per_class=2)
        self.scores = np.zeros((len(self.rows), len(INTENTS)))
        self.scores[0, INTENTS.index("declined_card_payment")] = 3
        self.scores[1, INTENTS.index("card_arrival")] = 3
        self.scores[2:, 0] = 2

    def test_no_candidate_restriction_and_oos_denominators(self):
        result = summarize(self.rows, self.scores, [{"threshold": 2, "target_validation_accuracy": 1}])
        self.assertEqual(result["mapped_intent"]["accuracy"], 0.5)
        policy = result["policies"][0]
        self.assertEqual(policy["mapped_intent"]["wrong_answered"], 1)
        self.assertEqual(policy["unsupported_groups"]["unsupported_banking"]["false_accepts"], 8)
        self.assertEqual(policy["unsupported_groups"]["unrelated_topics"]["false_accept_rate"], 1)
        self.assertNotIn("overall_accuracy", result)

    def test_zero_coverage_is_null_accuracy(self):
        result = summarize(self.rows, self.scores, [{"threshold": 4, "target_validation_accuracy": 1}])
        self.assertIsNone(result["policies"][0]["mapped_intent"]["accuracy_on_answered"])
        self.assertEqual(result["policies"][0]["mapped_intent"]["coverage"], 0)

    def test_invalid_scores_labels_and_thresholds_fail(self):
        for scores in (self.scores[:, :1], self.scores * float("nan")):
            with self.assertRaises(ValueError):
                summarize(self.rows, scores, [])
        for threshold in (True, -1, float("inf")):
            with self.assertRaises(ValueError):
                summarize(self.rows, self.scores, [{"threshold": threshold}])
        self.rows[0]["label"] = "cancel_transfer"
        with self.assertRaises(ValueError):
            summarize(self.rows, self.scores, [])


if __name__ == "__main__":
    unittest.main()
