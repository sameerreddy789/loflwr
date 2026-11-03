import json
from pathlib import Path
import tempfile
import unittest

from fedlora_edge.data import INTENTS, content_hash
from fedlora_edge.encoder_data import load_encoder_manifest, load_encoder_records, load_feature_split, sha256

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "Optional NumPy not installed")
class EncoderDataTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        records = [{"source_id": "train:1", "source_split": "train", "query": "example one", "label": INTENTS[0]},
                   {"source_id": "train:2", "source_split": "train", "query": "example two", "label": INTENTS[1]}]
        X = np.zeros((2, 768), dtype=np.float32)
        X[:, 0] = 1
        np.savez(root/"data.npz", X=X, y=np.array([0, 1], dtype=np.int64), source_ids=np.array(["train:1", "train:2"]))
        (root/"records.json").write_text(json.dumps({"records": records, "records_hash": content_hash(records)}))
        entry = {"file": "data.npz", "sha256": sha256(root/"data.npz"), "count": 2,
                 "records_file": "records.json", "records_hash": content_hash(records)}
        manifest = {"format_version": 1, "kind": "mpnet_local_feature_dataset", "labels": list(INTENTS),
                    "feature_dim": 768, "clients": {"0": entry}, "validation": entry}
        self.save_manifest(root, manifest)
        return root, manifest, records

    def save_manifest(self, root, manifest):
        manifest["manifest_hash"] = content_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})
        (root/"manifest.json").write_text(json.dumps(manifest))

    def test_local_archive_and_text_share_frozen_record_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _, records = self.fixture(folder)
            X, y, ids = load_feature_split(root, "0")
            self.assertEqual(X.shape, (2, 768))
            self.assertEqual(y.tolist(), [0, 1])
            self.assertEqual(ids.tolist(), [r["source_id"] for r in records])
            self.assertEqual(load_encoder_records(root, "0"), records)

    def test_corrupt_archive_and_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root, manifest, _ = self.fixture(folder)
            with (root/"data.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_feature_split(root, "0")
            manifest["feature_dim"] = 100
            (root/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "hash"):
                load_encoder_manifest(root)

    def test_normalization_and_label_bounds_are_checked_after_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            root, manifest, _ = self.fixture(folder)
            np.savez(root/"data.npz", X=np.zeros((2,768),dtype=np.float32),
                     y=np.array([0,20]), source_ids=np.array(["a","b"]))
            manifest["clients"]["0"]["sha256"] = sha256(root/"data.npz")
            self.save_manifest(root, manifest)
            with self.assertRaisesRegex(ValueError, "Invalid local feature"):
                load_feature_split(root, "0")

    def test_paths_cannot_escape_dataset_and_confirmation_is_not_feature_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root, manifest, _ = self.fixture(folder)
            with self.assertRaises(KeyError):
                load_feature_split(root, "confirmation")
            manifest["clients"]["0"]["file"] = "../outside.npz"
            self.save_manifest(root, manifest)
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_feature_split(root, "0")

    def test_changed_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _, records = self.fixture(folder)
            records[0]["label"] = INTENTS[2]
            (root/"records.json").write_text(json.dumps({"records":records,"records_hash":content_hash(records)}))
            with self.assertRaisesRegex(ValueError, "frozen manifest"):
                load_encoder_records(root, "0")


if __name__ == "__main__":
    unittest.main()
