"""Small, dependency-free tensor contract; optional framework imports stay at the edge.

All network/checkpoint tensor values are float32. Calculations may accumulate in
float64. Hashes use canonical little-endian bytes, independent of JSON formatting.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from array import array
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class TensorData:
    shape: tuple[int, ...]
    values: tuple[float, ...]
    dtype: str = "float32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "values", tuple(array("f", (float(x) for x in self.values))))
        if self.dtype != "float32":
            raise ValueError("Only float32 adapter payloads are supported.")
        if any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in self.shape):
            raise ValueError("Tensor dimensions must be positive integers.")
        if math.prod(self.shape) != len(self.values):
            raise ValueError("Tensor shape does not match its value count.")

    @classmethod
    def from_flat(cls, values: Iterable[float], shape: Iterable[int]) -> "TensorData":
        # Round once at the adapter boundary. NaN/Inf are retained for explicit
        # protocol validation rather than silently replaced.
        packed = array("f", (float(x) for x in values))
        return cls(tuple(shape), tuple(packed))

    @classmethod
    def from_numpy(cls, value: Any) -> "TensorData":
        return cls.from_flat(value.astype("float32").ravel().tolist(), value.shape)

    @classmethod
    def from_torch(cls, value: Any) -> "TensorData":
        tensor = value.detach().cpu().float().contiguous()
        return cls.from_flat(tensor.reshape(-1).tolist(), tuple(tensor.shape))

    @property
    def nbytes(self) -> int:
        return 4 * len(self.values)

    def to_bytes(self) -> bytes:
        return struct.pack("<" + str(len(self.values)) + "f", *self.values)

    def to_numpy(self) -> Any:
        import numpy as np

        return np.asarray(self.values, dtype=np.float32).reshape(self.shape)

    def to_torch(self) -> Any:
        import torch

        return torch.tensor(self.values, dtype=torch.float32).reshape(self.shape)

    def to_dict(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "dtype": self.dtype, "values": list(self.values)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorData":
        return cls(tuple(payload["shape"]), tuple(payload["values"]), payload["dtype"])


TensorState = dict[str, TensorData]


def state_schema(state: Mapping[str, TensorData]) -> dict[str, dict[str, Any]]:
    return {key: {"shape": list(state[key].shape), "dtype": state[key].dtype} for key in sorted(state)}


def state_nbytes(state: Mapping[str, TensorData]) -> int:
    return sum(tensor.nbytes for tensor in state.values())


def state_norm(state: Mapping[str, TensorData]) -> float:
    return math.sqrt(math.fsum(value * value for tensor in state.values() for value in tensor.values))


def finite_state(state: Mapping[str, TensorData]) -> bool:
    return bool(state) and all(math.isfinite(x) for tensor in state.values() for x in tensor.values)


def hash_state(state: Mapping[str, TensorData]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        descriptor = json.dumps(
            {"name": key, "shape": state[key].shape, "dtype": state[key].dtype},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        digest.update(struct.pack("<Q", len(descriptor)))
        digest.update(descriptor)
        digest.update(state[key].to_bytes())
    return digest.hexdigest()


def state_to_dict(state: Mapping[str, TensorData]) -> dict[str, Any]:
    return {key: value.to_dict() for key, value in sorted(state.items())}


def state_from_dict(payload: Mapping[str, Any]) -> TensorState:
    return {key: TensorData.from_dict(value) for key, value in payload.items()}


def subtract_states(local: Mapping[str, TensorData], base: Mapping[str, TensorData]) -> TensorState:
    if state_schema(local) != state_schema(base):
        raise ValueError("Adapter schemas differ.")
    return {
        key: TensorData.from_flat(
            (value - original for value, original in zip(local[key].values, base[key].values)),
            base[key].shape,
        )
        for key in base
    }
