"""Verify the derived training split cannot introduce unseen source IDs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.prepare_encoder_imbalance import prepare


@unittest.skipUnless(importlib.util.find_spec("numpy"), "Optional numpy is absent")
class EncoderImbalanceTests(unittest.TestCase):
    def test_missing_or_changed_training_record_fails_before_writing(self):
        import numpy as np
        rows = [{"source_id": f"train:{i}", "label": "card_arrival", "query": f"fixture {i}"} for i in range(10)]
        features = (np.zeros((10,768),dtype=np.float32), np.zeros(10,dtype=np.int64), np.array([r["source_id"] for r in rows]))
        for field, value in (("source_id", "validation:0"), ("query", "changed query")):
            requested = [dict(row) for row in rows]
            requested[0][field] = value
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "output"
                with patch("tools.prepare_encoder_imbalance.load_encoder_manifest", return_value={}), \
                     patch("tools.prepare_encoder_imbalance.load_feature_split", return_value=features), \
                     patch("tools.prepare_encoder_imbalance.load_encoder_records", return_value=rows), \
                     patch("tools.prepare_encoder_imbalance.load_manifest", return_value=requested):
                    with self.assertRaises(ValueError):
                        prepare("unused-source", "unused-original", output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
