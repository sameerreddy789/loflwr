"""Diagnostic score transforms; no model-performance claims."""

import unittest

from fedlora_edge.data import INTENTS
from tools.diagnose_scores import RULES, scoring_variants, summarize


class ScoringDiagnosticTests(unittest.TestCase):
    def test_normalization_and_centering_are_distinct(self):
        result = scoring_variants([-3.0, -4.0], [1, 4], [-1.0, -5.0])
        self.assertEqual(result[RULES[0]], [-3.0, -4.0])
        self.assertEqual(result[RULES[1]], [-3.0, -1.0])
        self.assertEqual(result[RULES[2]], [-2.0, 1.0])

    def test_rejects_nonfinite_or_misaligned_scores(self):
        for scores, lengths, null in [([], [], []), ([0], [1, 2], [0]),
                                     ([float("nan")], [1], [0]),
                                     ([0], [1], [float("inf")]), ([0], [0], [0]),
                                     ([0], [True], [0])]:
            with self.subTest(scores=scores, lengths=lengths, null=null):
                with self.assertRaises(ValueError):
                    scoring_variants(scores, lengths, null)

    def test_paired_counts_report_harms_as_well_as_corrections(self):
        scores = [-3.0, -4.0] + [-100.0] * 18
        rows = [{"label": INTENTS[0], "scores": scores},
                {"label": INTENTS[1], "scores": scores}]
        result = summarize(rows, [1, 4] + [1] * 18, [0] * 20)
        self.assertEqual(result[RULES[1]]["paired_against_sum"],
                         {"changed": 2, "corrected": 1, "harmed": 1})
        self.assertEqual(result[RULES[1]]["metrics"]["accuracy"], 0.5)

    def test_ties_keep_contract_label_order(self):
        result = summarize([{"label": INTENTS[0], "scores": [0] * 20}], [1] * 20, [0] * 20)
        for rule in RULES:
            self.assertEqual(result[rule]["prediction_counts"], {INTENTS[0]: 1})

    def test_rejects_missing_candidates(self):
        with self.assertRaises(ValueError):
            summarize([{"label": INTENTS[0], "scores": [0]}], [1] * 20, [0] * 20)


if __name__ == "__main__":
    unittest.main()
