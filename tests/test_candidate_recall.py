import unittest

from tools.candidate_recall import candidate_recall


class CandidateRecallTests(unittest.TestCase):
    def test_two_nearest_examples_with_same_label_use_one_candidate_slot(self):
        result = candidate_recall([[0.9, 0.8, 0.7, 0.6]], ["a", "a", "b", "c"], ["b"], ["a", "b", "c"])
        self.assertEqual(result["correct_within_top_k"], {"1": 0, "2": 1, "3": 1})

    def test_candidate_recall_can_be_perfect_when_accuracy_is_not(self):
        result = candidate_recall([[0.9, 0.8], [0.1, 0.2]], ["a", "b"], ["b", "b"], ["a", "b"], (1, 2))
        self.assertEqual(result["recall"], {"1": 0.5, "2": 1.0})

    def test_ties_follow_label_order_not_training_row_order(self):
        result = candidate_recall([[1, 1]], ["b", "a"], ["a"], ["a", "b"], (1,))
        self.assertEqual(result["correct_within_top_k"], {"1": 1})

    def test_rejects_nonfinite_or_misaligned_scores(self):
        for values in ([[float("nan"), 1]], [[1]], []):
            with self.assertRaises(ValueError):
                candidate_recall(values, ["a", "b"], ["a"], ["a", "b"], (1,))

    def test_rejects_missing_labels_and_invalid_candidate_counts(self):
        for labels, ks in [(["a", "a"], (1,)), (["a", "b"], (3,)), (["a", "b"], (True,))]:
            with self.assertRaises(ValueError):
                candidate_recall([[0.8, 0.6]], labels, ["a"], ["a", "b"], ks)


if __name__ == "__main__":
    unittest.main()
