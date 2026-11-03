"""Dataset protocol tests use synthetic rows, never synthetic task metrics."""

import copy
import tempfile
import unittest
from pathlib import Path

from fedlora_edge.data import (
    DATASET_REVISION,
    INTENTS,
    NORMALIZATION_VERSION,
    build_selections,
    content_hash,
    load_manifest,
    normalize_query,
    verify_selections,
    write_json,
)


def synthetic_source():
    def rows(split, count):
        return [
            {"query": f"{split} {label} query {index}", "label": label,
             "source_split": split, "source_row": offset * count + index + 1,
             "source_id": f"{split}:{offset * count + index + 1}"}
            for offset, label in enumerate(INTENTS) for index in range(count)
        ]
    return rows("train", 120), rows("test", 30)


class ManifestTests(unittest.TestCase):
    def test_unicode_case_and_whitespace_normalization(self):
        self.assertEqual(normalize_query("  ＣＡＲＤ\tStraße\n"), "card strasse")

    def test_exact_allocations_and_deterministic_selection(self):
        train, test = synthetic_source()
        selections, _ = build_selections(train, test)
        repeated, _ = build_selections(train, test)
        self.assertEqual(selections, repeated)
        self.assertEqual([len(selections[f"imbalanced/client_{client}"]) for client in range(5)], [10, 360, 10, 10, 10])
        verify_selections(selections)
        changed, _ = build_selections(train, test, selection_seed=17)
        self.assertNotEqual(selections["validation"], changed["validation"])

    def test_conflicting_labels_stop_even_outside_selected_labels(self):
        train, test = synthetic_source()
        train.extend([{"query": "same query", "label": "outside", "source_id": "extra1"}, {"query": "ＳＡＭＥ  QUERY", "label": INTENTS[0], "source_id": "extra2"}])
        with self.assertRaisesRegex(ValueError, "Conflicting labels"):
            build_selections(train, test)

    def test_official_test_matches_are_removed_from_training(self):
        train, test = synthetic_source()
        duplicate = {**test[0], "source_split": "train", "source_row": 10000, "source_id": "train:10000", "query": test[0]["query"].upper()}
        train.append(duplicate)
        selections, diagnostics = build_selections(train, test)
        self.assertEqual(diagnostics["excluded_duplicates"]["train_matching_official_test"], 1)
        self.assertNotIn("train:10000", {row["source_id"] for name, rows in selections.items() if name != "test" for row in rows})

    def test_no_silent_oversampling(self):
        train, test = synthetic_source()
        train = [row for row in train if row["label"] != "pending_card_payment"]
        with self.assertRaisesRegex(ValueError, "Insufficient unique rows"):
            build_selections(train, test)

    def test_duplicate_inside_client_fails_verification(self):
        selections, _ = build_selections(*synthetic_source())
        selections["balanced/client_0"][1] = copy.deepcopy(selections["balanced/client_0"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate or held-out leakage"):
            verify_selections(selections)

    def test_heldout_leakage_fails_verification(self):
        selections, _ = build_selections(*synthetic_source())
        selections["balanced/client_0"][0] = copy.deepcopy(selections["validation"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate or held-out leakage"):
            verify_selections(selections)

    def test_tampered_manifest_fails_loading(self):
        manifest = {"dataset_revision": DATASET_REVISION, "normalization_version": NORMALIZATION_VERSION, "records": [{"query": "original"}]}
        manifest["manifest_hash"] = content_hash(manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_json(path, manifest)
            self.assertEqual(load_manifest(path), [{"query": "original"}])
            manifest["records"][0]["query"] = "modified"
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
