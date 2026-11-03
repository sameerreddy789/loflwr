"""Validated, atomic JSON checkpoints and append-only machine-readable events."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from .state import TensorData, finite_state, hash_state, state_from_dict, state_to_dict


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    """Flush the full new file before replacement; old files survive failed writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as output:
            name = output.name
            json.dump(value, output, sort_keys=True, separators=(",", ":"), allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
        name = None
        # POSIX durability across power loss. Windows os.replace remains atomic.
        if os.name != "nt":
            fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


class CheckpointStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "checkpoint.json"

    def save(self, *, run_id: str, attempt_id: int, version: int,
             state: Mapping[str, TensorData], identity: dict, config_hash: str,
             metadata: dict | None = None) -> dict:
        if not finite_state(state):
            raise ValueError("Cannot checkpoint an empty or nonfinite state")
        if (any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in (attempt_id, version)) or version > attempt_id):
            raise ValueError("Invalid attempted/committed checkpoint counters")
        payload = {
            "format_version": 1, "run_id": run_id, "attempt_id": attempt_id,
            "version": version, "b_hash": hash_state(state), "identity": identity,
            "config_hash": config_hash, "state": state_to_dict(state),
            "metadata": metadata or {},
        }
        atomic_json(self.path, payload)
        return payload

    def load(self) -> dict:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        state = state_from_dict(payload["state"])
        if not finite_state(state) or hash_state(state) != payload["b_hash"]:
            raise ValueError("Checkpoint state/hash mismatch")
        if any(not isinstance(payload.get(key), int) or isinstance(payload[key], bool)
               or payload[key] < 0 for key in ("attempt_id", "version")):
            raise ValueError("Invalid checkpoint counters")
        if payload["version"] > payload["attempt_id"]:
            raise ValueError("Committed version cannot exceed attempted rounds")
        return {**payload, "state": state}


class EventLog:
    def __init__(self, run_dir: str | Path, run_id: str):
        self.path = Path(run_dir) / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def emit(self, event: str, **fields: Any) -> dict:
        record = {"event": event, "run_id": self.run_id,
                  "timestamp": time.time(), **fields}
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            output.flush()
        return record
