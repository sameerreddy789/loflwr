import unittest

from fedlora_edge.data import INTENTS
from tools.accuracy_feasibility import expanded_training, wilson_interval


class FeasibilityTests(unittest.TestCase):
    def test_expanded_data_excludes_normalized_validation_and_official_test(self):
        rows = [{"source_id": f"train:{i}", "source_split": "train", "label": label,
                 "query": f"unique query {i}"} for i, label in enumerate(INTENTS)]
        rows += [{"source_id": "train:100", "source_split": "train", "label": INTENTS[0], "query": " HELD  OUT "},
                 {"source_id": "train:101", "source_split": "train", "label": INTENTS[1], "query": "official row"}]
        validation = [{"source_id": "train:200", "label": INTENTS[0], "query": "held out"}]
        test = [{"source_id": "test:1", "label": INTENTS[1], "query": "OFFICIAL ROW"}]
        chosen, _ = expanded_training(rows, test, validation)
        self.assertEqual(len(chosen), 20)
        self.assertFalse({"train:100", "train:101"} & {row["source_id"] for row in chosen})

    def test_expanded_data_rejects_original_test_rows(self):
        with self.assertRaises(ValueError):
            expanded_training([{"source_split": "test"}], [], [])

    def test_wilson_99_percent_on_500_does_not_establish_population_99(self):
        lo, hi = wilson_interval(495, 500)
        self.assertAlmostEqual(lo, 0.9768, places=3)
        self.assertTrue(lo < 0.99 < hi)

    def test_wilson_rejects_invalid_counts(self):
        for k, n in [(1, 0), (-1, 2), (3, 2)]:
            with self.assertRaises(ValueError):
                wilson_interval(k, n)


if __name__ == "__main__":
    unittest.main()
