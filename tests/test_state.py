"""Tensor-state contract checks, not language-model training experiments."""

import json
import math
import unittest

from fedlora_edge.state import (
    TensorData,
    finite_state,
    hash_state,
    state_from_dict,
    state_nbytes,
    state_norm,
    state_schema,
    state_to_dict,
    subtract_states,
)


class TensorContractTests(unittest.TestCase):
    def test_all_construction_paths_have_one_float32_value(self):
        direct = TensorData((1,), (0.1,))
        flat = TensorData.from_flat([0.1], [1])
        decoded = TensorData.from_dict({"shape": [1], "dtype": "float32", "values": [0.1]})
        self.assertEqual(direct, flat)
        self.assertEqual(flat, decoded)
        self.assertNotEqual(direct.values[0], 0.1)
        self.assertEqual(hash_state({"b": direct}), hash_state({"b": flat}))

    def test_serialization_round_trip_preserves_state_and_hash(self):
        original = {"b": TensorData.from_flat([0.1, -2.5, 3.0, 4.0], [2, 2])}
        decoded = state_from_dict(json.loads(json.dumps(state_to_dict(original))))
        self.assertEqual(decoded, original)
        self.assertEqual(hash_state(decoded), hash_state(original))

    def test_shape_and_value_count_must_match(self):
        with self.assertRaises(ValueError):
            TensorData((2, 2), (1, 2, 3))

    def test_dimensions_are_positive_integers_at_each_boundary(self):
        for shape in ((0,), (-1,), (1.5,), (True,)):
            for constructor in (
                lambda: TensorData(shape, (1,)),
                lambda: TensorData.from_flat([1], shape),
                lambda: TensorData.from_dict({"shape": shape, "values": [1], "dtype": "float32"}),
            ):
                with self.subTest(shape=shape), self.assertRaises(ValueError):
                    constructor()

    def test_scalar_shape_is_supported(self):
        self.assertEqual(TensorData((), (2,)).nbytes, 4)

    def test_only_float32_is_supported(self):
        with self.assertRaises(ValueError):
            TensorData((1,), (1,), dtype="float16")

    def test_hash_is_independent_of_dictionary_order(self):
        a = TensorData((1,), (1,))
        b = TensorData((1,), (2,))
        self.assertEqual(hash_state({"a": a, "b": b}), hash_state({"b": b, "a": a}))

    def test_hash_binds_tensor_name_shape_and_values(self):
        baseline = {"b": TensorData((2,), (1, 2))}
        for changed in (
            {"other": TensorData((2,), (1, 2))},
            {"b": TensorData((1, 2), (1, 2))},
            {"b": TensorData((2,), (1, 3))},
        ):
            self.assertNotEqual(hash_state(baseline), hash_state(changed))

    def test_norm_includes_every_tensor(self):
        state = {"one": TensorData((1,), (3,)), "two": TensorData((1,), (4,))}
        self.assertEqual(state_norm(state), 5)

    def test_payload_counts_four_bytes_per_element(self):
        state = {"one": TensorData((1,), (3,)), "two": TensorData((2,), (4, 5))}
        self.assertEqual(state_nbytes(state), 12)

    def test_nonfinite_values_remain_available_for_protocol_rejection(self):
        for value in (math.nan, math.inf, -math.inf):
            state = {"b": TensorData((1,), (value,))}
            self.assertFalse(finite_state(state))
            self.assertFalse(math.isfinite(state["b"].values[0]))

    def test_empty_state_is_not_a_finite_update(self):
        self.assertFalse(finite_state({}))

    def test_zero_delta_is_finite_and_has_zero_norm(self):
        state = {"b": TensorData((2,), (0, 0))}
        self.assertTrue(finite_state(state))
        self.assertEqual(state_norm(state), 0)

    def test_subtraction_yields_adapter_delta(self):
        base = {"b": TensorData((2,), (1, 2))}
        local = {"b": TensorData((2,), (4, 6))}
        self.assertEqual(subtract_states(local, base), {"b": TensorData((2,), (3, 4))})
        self.assertEqual(state_norm(subtract_states(local, base)), 5)

    def test_subtraction_rejects_names_or_shapes_that_differ(self):
        base = {"b": TensorData((2,), (1, 2))}
        for local in ({"other": TensorData((2,), (1, 2))}, {"b": TensorData((1, 2), (1, 2))}):
            with self.assertRaises(ValueError):
                subtract_states(local, base)

    def test_schema_order_is_stable(self):
        schema = state_schema({"z": TensorData((1,), (0,)), "a": TensorData((2,), (0, 0))})
        self.assertEqual(list(schema), ["a", "z"])
        self.assertEqual(schema["a"], {"shape": [2], "dtype": "float32"})


if __name__ == "__main__":
    unittest.main()
