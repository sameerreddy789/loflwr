"""B-coordinate aggregation shared by the protocol and every Flower runtime."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Mapping

from .state import TensorData, TensorState, finite_state, state_norm, state_schema


@dataclass(frozen=True)
class AggregationResult:
    state: TensorState
    telemetry: dict


def aggregate(
    base: Mapping[str, TensorData],
    deltas: Mapping[str, Mapping[str, TensorData]],
    *,
    strategy: str = "equal_clipped",
    sample_sizes: Mapping[str, int] | None = None,
    epsilon: float = 1e-12,
) -> AggregationResult:
    """Add one weighted delta to B; clip each *whole* delta to the median norm.

    ``sample`` uses declared manifest ownership, never completed-work telemetry.
    Zero medians deliberately produce a zero update. Even medians are the mean
    of the two middle norms. All persisted tensors are rounded back to FP32.
    """
    if strategy not in {"sample", "equal", "equal_clipped"}:
        raise ValueError(f"Unknown strategy: {strategy}")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive")
    if not finite_state(base) or not deltas:
        raise ValueError("A finite nonempty base and contributor set are required")
    clients = sorted(deltas)
    schema = state_schema(base)
    for client_id in clients:
        if state_schema(deltas[client_id]) != schema or not finite_state(deltas[client_id]):
            raise ValueError(f"Invalid delta for {client_id}")
    norms = {client_id: state_norm(deltas[client_id]) for client_id in clients}
    if not all(math.isfinite(norm) for norm in norms.values()):
        raise ValueError("Nonfinite delta norm")
    threshold = statistics.median(norms.values()) if strategy == "equal_clipped" else None
    factors = {
        client_id: min(1.0, threshold / (norms[client_id] + epsilon))
        if threshold is not None else 1.0
        for client_id in clients
    }
    if strategy == "sample":
        if sample_sizes is None or any(
            not isinstance(sample_sizes.get(c), int)
            or isinstance(sample_sizes[c], bool) or sample_sizes[c] <= 0 for c in clients
        ):
            raise ValueError("Sample weighting requires positive integer manifest sizes")
        total = sum(sample_sizes[c] for c in clients)
        coefficients = {c: sample_sizes[c] / total for c in clients}
    else:
        coefficients = {c: 1.0 / len(clients) for c in clients}
    delta = {
        key: TensorData.from_flat(
            (math.fsum(coefficients[c] * factors[c] * deltas[c][key].values[i]
                       for c in clients) for i in range(len(base[key].values))),
            base[key].shape,
        )
        for key in sorted(base)
    }
    state = {
        key: TensorData.from_flat(
            (x + y for x, y in zip(base[key].values, delta[key].values)), base[key].shape
        )
        for key in sorted(base)
    }
    if not finite_state(delta) or not finite_state(state):
        raise ValueError("Nonfinite computed aggregate")
    return AggregationResult(state, {
        "strategy": strategy,
        "coefficients": coefficients,
        "declared_sample_sizes": {c: (sample_sizes or {}).get(c) for c in clients},
        "norms": norms,
        "clip_threshold": threshold,
        "clip_factors": factors,
        "clipping_fraction": sum(factor < 1 for factor in factors.values()) / len(clients),
        "clipped_norms": {c: norms[c] * factors[c] for c in clients},
        "zero_delta_clients": [c for c in clients if norms[c] == 0],
        "zero_threshold": threshold == 0 if threshold is not None else False,
        "aggregate_norm": state_norm(delta),
        "epsilon": epsilon,
    })
