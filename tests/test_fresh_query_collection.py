"""Synthetic fixtures only: never open spent queries or load a model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import fresh_query_collection as collection


def fixture(count_per_intent=1):
    plan = {"version": collection.VERSION, "batch_id": "fixture-batch",
            "target_per_intent": 20, "normalization": collection.NORMALIZATION,
            "protocol_reviewed": True, "overlap_inventory_reviewed": True,
            "curator_id": "p-0005"}
    people = [{"participant_id": f"p-{index:04d}", "roles": role,
               "no_model_outputs": "true", "no_old_queries": "true", "independent_reviews": "true"}
              for index, role in enumerate(("writer", "labeler", "labeler", "adjudicator", "collector"), 1)]
    queries, labels_a, labels_b, adjudications = [], [], [], []
    for label in collection.INTENTS:
        for number in range(count_per_intent):
            key = len(queries) + 1
            row = {"query_id": f"q-{key:04d}", "query": f"Synthetic unit fixture {label} {number}",
                   "author_id": "p-0001", "source_id": f"src-{key:04d}", "collected_at": "2026-01-01",
                   "collection_batch": "fixture-batch", "provenance": "independent_human", "provenance_detail": ""}
            queries.append(row)
            for output, reviewer in ((labels_a, "p-0002"), (labels_b, "p-0003")):
                output.append({"query_id": row["query_id"], "query": row["query"], "reviewer_id": reviewer,
                               "label": label, "notes": ""})
            adjudications.append({"query_id": row["query_id"], "adjudicator_id": "p-0004",
                                  "final_label": label, "decision": "retain", "exclusion_reason": "",
                                  "near_duplicate_of": "", "privacy_checked": "true",
                                  "near_duplicate_checked": "true", "reason": "Synthetic agreement fixture"})
    return [plan, queries, people, labels_a, labels_b, adjudications, {}, [], []]


class FreshQueryCollectionTests(unittest.TestCase):
    def test_init_creates_no_queries_and_never_overwrites_a_pack(self):
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory) / "pack"
            collection.create_pack(pack, "fixture-batch")
            self.assertEqual(collection.read_rows(pack / "queries.csv", collection.QUERY_FIELDS), [])
            report = collection.ready_check(pack)
            self.assertFalse(report["collection_ready"])
            self.assertEqual(report["counts"]["submitted"], 0)
            self.assertIn("human_collection_pending", report["errors"])
            self.assertEqual(sum(report["target_deficits"].values()), 400)
            self.assertFalse(json.loads((pack / "evaluation-handoff.json").read_text())["evaluation_ready"])
            with self.assertRaises(FileExistsError):
                collection.create_pack(pack, "fixture-batch")

    def test_csv_json_and_jsonl_imports_are_equivalent_and_reject_model_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            rows = fixture()[1][:1]
            collection.write_csv(base / "queries.csv", collection.QUERY_FIELDS, rows)
            collection.write_json(base / "queries.json", {"records": rows})
            (base / "queries.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            for name in ("queries.csv", "queries.json", "queries.jsonl"):
                self.assertEqual(collection.read_rows(base / name, collection.QUERY_FIELDS), rows)
            collection.write_json(base / "bad.json", [{**rows[0], "prediction": "card_arrival"}])
            with self.assertRaisesRegex(ValueError, "unexpected fields"):
                collection.read_rows(base / "bad.json", collection.QUERY_FIELDS)
            (base / "bad.csv").write_text("query_id,query_id\nq-1,q-2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                collection.read_rows(base / "bad.csv", collection.QUERY_FIELDS)

    def test_label_packets_hide_metadata_and_empty_packets_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "queries.csv"
            collection.write_csv(source, collection.QUERY_FIELDS, fixture()[1][:2])
            collection.prepare_labels(source, base / "packets")
            for name in ("reviewer-a", "reviewer-b"):
                rows = collection.read_rows(base / "packets" / name / "labels.csv", collection.LABEL_FIELDS)
                self.assertEqual(len(rows), 2)
                self.assertNotIn("author_id", rows[0])
                self.assertNotIn("provenance", rows[0])
                self.assertEqual(rows[0]["label"], "")
                self.assertEqual(rows[0]["reviewer_id"], "")
            with self.assertRaises(FileExistsError):
                collection.prepare_labels(source, base / "packets")
            collection.write_csv(source, collection.QUERY_FIELDS, [])
            with self.assertRaisesRegex(ValueError, "No human queries"):
                collection.prepare_labels(source, base / "empty-packets")
            self.assertFalse((base / "empty-packets").exists())

    def test_complete_400_fixture_is_only_ready_for_future_lock(self):
        report = collection.audit_collection(*fixture(20))
        self.assertEqual(report["status"], "ready_for_lock")
        self.assertTrue(report["collection_ready"])
        self.assertFalse(report["evaluation_ready"])
        self.assertFalse(report["model_inference_performed"])
        self.assertEqual(report["counts"]["supported"], 400)
        self.assertNotIn("accuracy", report)
        report = collection.audit_collection(*fixture(19))
        self.assertFalse(report["collection_ready"])
        self.assertEqual(sum(report["target_deficits"].values()), 20)

    def test_target_cannot_be_silently_reduced(self):
        data = fixture()
        data[0]["target_per_intent"] = 1
        report = collection.audit_collection(*data)
        self.assertIn("collection_protocol_changed", report["errors"])
        self.assertFalse(report["collection_ready"])

    def test_both_labels_and_adjudication_are_required_even_on_agreement(self):
        data = fixture()
        data[3].pop(0)
        data[5].pop(0)
        report = collection.audit_collection(*data)
        self.assertEqual(report["counts"]["missing_or_invalid_labels"], 1)
        self.assertEqual(report["counts"]["missing_adjudications"], 1)
        self.assertEqual(report["records"][0]["disposition"], "pending")
        self.assertIn("unfinished_records", report["errors"])

    def test_labels_and_adjudicator_must_be_independent_from_each_other_and_author(self):
        for index, field, replacement, expected in (
            (4, "reviewer_id", "p-0002", "labelers_not_independent"),
            (3, "reviewer_id", "p-0001", "author_is_labeler_1"),
            (5, "adjudicator_id", "p-0002", "adjudicator_not_independent"),
            (3, "query", "Changed after review", "reviewed_query_changed_1"),
        ):
            with self.subTest(expected=expected):
                data = fixture()
                data[index][0][field] = replacement
                report = collection.audit_collection(*data)
                self.assertIn(expected, report["records"][0]["issues"])
                self.assertEqual(report["records"][0]["disposition"], "pending")

    def test_disagreement_needs_recorded_adjudication(self):
        data = fixture()
        data[4][0]["label"] = "ambiguous"
        data[5][0]["reason"] = ""
        report = collection.audit_collection(*data)
        self.assertEqual(report["counts"]["label_disagreements"], 1)
        self.assertIn("adjudication_reason_missing", report["records"][0]["issues"])
        data[5][0]["reason"] = "Fixture adjudication based on the ontology definition"
        self.assertEqual(collection.audit_collection(*data)["records"][0]["disposition"], "supported")

    def test_ambiguous_unsupported_and_external_rows_remain_visible(self):
        data = fixture()
        for index, label in enumerate(("ambiguous", "unsupported")):
            data[3][index]["label"] = data[4][index]["label"] = data[5][index]["final_label"] = label
        data[1][2].update(provenance="external", provenance_detail="Synthetic external-source fixture", author_id="p-0005")
        report = collection.audit_collection(*data)
        self.assertEqual(report["counts"]["submitted"], 20)
        self.assertEqual(report["counts"]["secondary"], 3)
        self.assertEqual(report["counts"]["supported"], 17)
        self.assertEqual(report["flag_counts"]["unsupported"], 1)
        self.assertEqual(report["flag_counts"]["ambiguous"], 1)
        self.assertEqual(report["flag_counts"]["external"], 1)

    def test_normalized_duplicates_require_exclusion_and_conflicting_labels_block(self):
        data = fixture()
        duplicate_text = "  " + data[1][0]["query"].upper() + "\t"
        data[1][1]["query"] = duplicate_text
        data[3][1]["query"] = data[4][1]["query"] = duplicate_text
        report = collection.audit_collection(*data)
        self.assertIn("duplicate_must_be_adjudicated_for_exclusion", report["records"][1]["issues"])
        self.assertTrue(any(error.startswith("conflicting_duplicate_labels:") for error in report["errors"]))
        data[5][1].update(final_label=data[5][0]["final_label"], decision="exclude", exclusion_reason="duplicate")
        report = collection.audit_collection(*data)
        self.assertEqual(report["records"][1]["disposition"], "excluded")
        self.assertEqual(report["records"][1]["duplicate_of"], "q-0001")
        self.assertFalse(any(error.startswith("conflicting_duplicate_labels:") for error in report["errors"]))
        self.assertEqual(collection.normalize_query("ＦＯＯ\tBar"), "foo bar")

    def test_old_sources_are_only_hashed_and_missing_categories_block(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            text = fixture()[1][0]["query"]
            collection.write_json(base / "old.json", {"records": [{"query": text, "label": "unused_label"}]})
            sources = [{"dataset_id": category, "category": category, "path": "old.json",
                        "text_field": "query", "records_key": "records"} for category in collection.OVERLAP_CATEGORIES]
            hashes, evidence, errors = collection.overlap_index(sources, base)
            self.assertEqual(errors, [])
            self.assertEqual(len(evidence), 6)
            self.assertNotIn(text, json.dumps(evidence))
            self.assertEqual(len(evidence[0]["file_sha256"]), 64)
            data = fixture()
            data[6:9] = [hashes, evidence, errors]
            report = collection.audit_collection(*data)
            self.assertIn("known_overlap", report["records"][0]["flags"])
            self.assertEqual(report["records"][0]["disposition"], "pending")
            self.assertNotIn(text, json.dumps(report))
            _, _, errors = collection.overlap_index(sources[:1], base)
            self.assertIn("unverified_overlap_category:scope-negative-2279", errors)
            _, _, errors = collection.overlap_index([{**sources[0], "path": "missing.json"}], base)
            self.assertIn("unreadable_or_invalid_overlap_source:training", errors)

    def test_manual_near_duplicate_privacy_and_author_attestations_fail_closed(self):
        data = fixture()
        data[5][0].update(decision="exclude", exclusion_reason="near_duplicate")
        self.assertIn("missing_near_duplicate_reference", collection.audit_collection(*data)["records"][0]["issues"])
        data[5][0]["near_duplicate_of"] = "fresh:q-0002"
        self.assertEqual(collection.audit_collection(*data)["records"][0]["disposition"], "excluded")
        data[5][1].update(decision="exclude", exclusion_reason="privacy")
        report = collection.audit_collection(*data)
        self.assertEqual(report["counts"]["excluded"], 2)
        self.assertEqual(len(report["records"]), 20)
        data[2][0]["no_old_queries"] = "false"
        report = collection.audit_collection(*data)
        self.assertIn("independent_author_attestation_missing", report["records"][2]["issues"])

    def test_bad_provenance_dates_and_source_ids_are_reported(self):
        data = fixture()
        data[1][0].update(collected_at="tomorrow", provenance="invented", source_id="someone@example.test")
        report = collection.audit_collection(*data)
        self.assertIn("invalid_collection_date", report["records"][0]["issues"])
        self.assertIn("invalid_provenance", report["records"][0]["issues"])
        self.assertIn("invalid_source_pseudonym", report["records"][0]["issues"])

    def test_cli_empty_pack_and_invalid_schema_exit_two_without_scores(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(collection, "ROOT", Path(directory)):
            pack = Path(directory) / "runs" / "fixture"
            with patch("sys.stdout"):
                self.assertEqual(collection.main(["init", "--output", str(pack), "--batch-id", "fixture-batch"]), 0)
                self.assertEqual(collection.main(["ready-check", "--pack", str(pack)]), 2)
                (pack / "queries.csv").write_text("wrong,columns\n", encoding="utf-8")
                self.assertEqual(collection.main(["ready-check", "--pack", str(pack)]), 2)
            report = json.loads((pack / "ready-report.json").read_text())
            self.assertEqual(report["status"], "blocked")
            self.assertFalse(report["evaluation_ready"])
            self.assertFalse(report["model_inference_performed"])
            with self.assertRaises(ValueError):
                collection.private_output(Path(directory) / "public-queries.csv")

    def test_changed_ontology_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory) / "pack"
            collection.create_pack(pack, "fixture-batch")
            changed = copy.deepcopy(collection.ontology_document())
            changed["intents"][0]["label"] = "substituted"
            collection.write_json(pack / "ontology.json", changed)
            with self.assertRaisesRegex(ValueError, "Ontology differs"):
                collection.ready_check(pack)

    def test_report_cannot_overwrite_overlap_source_or_override_input(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(collection, "ROOT", Path(directory)):
            pack = Path(directory) / "runs" / "fixture"
            collection.create_pack(pack, "fixture-batch")
            old_source = pack.parent / "old.json"
            collection.write_json(old_source, [{"query": "Synthetic fixture only"}])
            collection.write_json(pack / "overlap-sources.json", [{
                "dataset_id": "fixture-old", "category": "training", "path": "../old.json",
                "text_field": "query", "records_key": "records"}])
            before = old_source.read_bytes()
            with patch("sys.stdout"), patch("sys.stderr"):
                self.assertEqual(collection.main(["ready-check", "--pack", str(pack),
                                                  "--report", str(old_source)]), 2)
                report_path = pack / "ready-report.json"
                collection.write_csv(report_path, collection.QUERY_FIELDS, [])
                original = report_path.read_bytes()
                self.assertEqual(collection.main(["ready-check", "--pack", str(pack),
                                                  "--queries", str(report_path)]), 2)
            self.assertEqual(old_source.read_bytes(), before)
            self.assertEqual(report_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
