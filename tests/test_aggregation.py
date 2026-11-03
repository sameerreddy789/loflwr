"""Dependency-free mathematical checks; these are not LM experiment results."""

import math
import unittest

from fedlora_edge.aggregation import aggregate
from fedlora_edge.state import TensorData


def state(a, b=0):
    return {"q": TensorData((1,), (a,)), "v": TensorData((1,), (b,))}


class AggregationTests(unittest.TestCase):
    def test_equal_is_survivor_mean(self):
        result = aggregate(state(10), {"0": state(2), "1": state(4)}, strategy="equal")
        self.assertEqual(result.state["q"].values, (13,))
        self.assertEqual(result.telemetry["coefficients"], {"0": 0.5, "1": 0.5})

    def test_sample_weights_use_manifest_sizes(self):
        result = aggregate(state(0), {"0": state(2), "1": state(4)}, strategy="sample",
                           sample_sizes={"0": 9, "1": 1})
        self.assertAlmostEqual(result.state["q"].values[0], 2.2, places=6)
        self.assertEqual(result.telemetry["coefficients"], {"0": 0.9, "1": 0.1})

    def test_cross_tensor_norm_and_even_median(self):
        result = aggregate(state(0), {"0": state(3, 4), "1": state(0, 10)})
        log = result.telemetry
        self.assertEqual(log["norms"], {"0": 5, "1": 10})
        self.assertEqual(log["clip_threshold"], 7.5)
        self.assertAlmostEqual(log["clip_factors"]["1"], 0.75)
        self.assertEqual(log["clip_factors"]["0"], 1)
        self.assertAlmostEqual(result.state["q"].values[0], 1.5)
        self.assertAlmostEqual(result.state["v"].values[0], 5.75)

    def test_zero_threshold_is_not_replaced_by_positive_floor(self):
        result = aggregate(state(7), {"0": state(0), "1": state(0), "2": state(100)})
        self.assertEqual(result.state, state(7))
        self.assertEqual(result.telemetry["clip_threshold"], 0)
        self.assertEqual(result.telemetry["aggregate_norm"], 0)
        self.assertEqual(result.telemetry["zero_delta_clients"], ["0", "1"])

    def test_single_zero_delta_is_valid(self):
        result = aggregate(state(3), {"0": state(0)})
        self.assertEqual(result.state, state(3))
        self.assertTrue(result.telemetry["zero_threshold"])

    def test_common_a_effective_update_equivalence(self):
        # Two output rows, rank two, one shared A with three input columns.
        a = ((2, -1, 3), (4, 0, 1))
        first = ((1, 2), (3, 4))
        second = ((5, -2), (1, 0))
        base = {"b": TensorData((2, 2), (0, 0, 0, 0))}
        deltas = {"0": {"b": TensorData((2, 2), sum(first, ()))},
                  "1": {"b": TensorData((2, 2), sum(second, ()))}}
        merged = aggregate(base, deltas, strategy="equal").state["b"].values
        for row in range(2):
            for col in range(3):
                effective = sum(merged[2 * row + k] * a[k][col] for k in range(2))
                mean_effective = sum(sum(b[row][k] * a[k][col] for k in range(2))
                                     for b in (first, second)) / 2
                self.assertEqual(effective, mean_effective)

    def test_rejects_invalid_aggregation_inputs(self):
        cases = [({}, {"0": state(1)}, {}),
                 (state(0), {}, {}),
                 (state(0), {"0": state(math.inf)}, {}),
                 (state(0), {"0": {"x": TensorData((1,), (1,))}}, {}),
                 (state(0), {"0": state(1)}, {"epsilon": 0}),
                 (state(0), {"0": state(1)}, {"strategy": "sample", "sample_sizes": {"0": 0}})]
        for base, updates, kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                aggregate(base, updates, **kwargs)


if __name__ == "__main__":
    unittest.main()
