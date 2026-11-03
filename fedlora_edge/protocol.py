"""Transport-independent round contracts, validation, quorum and commit semantics.

Authenticated identity is supplied by the transport, never trusted from a reply.
Time/work telemetry is a report from a trusted node, not proof of training.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .aggregation import aggregate
from .checkpoint import CheckpointStore, EventLog, atomic_json, canonical_hash
from .state import TensorData, TensorState, finite_state, hash_state, state_nbytes, state_schema


@dataclass(frozen=True)
class RoundRequest:
    run_id: str
    attempt_id: int
    version: int
    base_b_hash: str
    contract_hash: str
    common_a_hash: str
    model_revision: str
    client_id: str
    optimizer_steps: int
    deadline_unix: float
    seed: int
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("run_id", "base_b_hash", "contract_hash", "common_a_hash",
                     "model_revision", "client_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"Request {name} must be a nonempty string")
        for name, minimum in (("attempt_id", 1), ("version", 0), ("optimizer_steps", 1)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"Invalid request {name}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("Request seed must be an integer")
        if (not isinstance(self.deadline_unix, (float, int)) or isinstance(self.deadline_unix, bool)
                or not math.isfinite(self.deadline_unix)):
            raise ValueError("Request deadline must be finite")
        if not isinstance(self.manifest_hash, str):
            raise ValueError("Request manifest_hash must be a string")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "RoundRequest":
        return cls(**value)


@dataclass
class ClientReply:
    request: RoundRequest
    delta: TensorState
    status: str
    telemetry: dict[str, Any] = field(default_factory=dict)
    serialized_reply_bytes: int = 0


def completed_steps(telemetry: Mapping[str, Any]) -> int:
    value = telemetry.get("optimizer_steps", telemetry.get("completed_optimizer_steps", 0))
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def validate_request(request: RoundRequest, state: Mapping[str, TensorData],
                     identity: Mapping[str, Any], *, now: float | None = None) -> None:
    """A client verifies actual loaded model/A identity before touching incoming B."""
    for key in ("contract_hash", "common_a_hash", "model_revision"):
        if getattr(request, key) != identity.get(key):
            raise ValueError(f"request_{key}_mismatch")
    if not finite_state(state) or hash_state(state) != request.base_b_hash:
        raise ValueError("request_base_b_hash_mismatch")
    if (time.time() if now is None else now) >= request.deadline_unix:
        raise ValueError("request_deadline_expired")


class RoundCoordinator:
    def __init__(self, *, run_dir: str | Path, run_id: str, initial_state: TensorState,
                 identity: dict, clients: Mapping[str, int], quorum: int = 3,
                 strategy: str = "equal_clipped", seed: int = 42,
                 deadline_seconds: float = 300, epsilon: float = 1e-12,
                 config_extra: dict | None = None, resume: bool = False):
        if not finite_state(initial_state):
            raise ValueError("Initial B must be finite and nonempty")
        if (not clients or not isinstance(quorum, int) or isinstance(quorum, bool)
                or not 1 <= quorum <= len(clients)):
            raise ValueError("Quorum must lie within known client count")
        if any(not isinstance(c, str) or not isinstance(n, int) or isinstance(n, bool)
               or n <= 0 for c, n in clients.items()):
            raise ValueError("Clients require string IDs and positive manifest sizes")
        if not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
            raise ValueError("Deadline must be finite and positive")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Clipping epsilon must be finite and positive")
        if strategy not in {"sample", "equal", "equal_clipped"}:
            raise ValueError("Unknown aggregation strategy")
        if any(not identity.get(key) for key in ("contract_hash", "common_a_hash", "model_revision")):
            raise ValueError("Incomplete model/adapter identity")
        self.run_dir, self.run_id = Path(run_dir), run_id
        self.clients, self.quorum = dict(clients), quorum
        self.identity, self.strategy = dict(identity), strategy
        self.seed, self.deadline_seconds, self.epsilon = seed, deadline_seconds, epsilon
        self.store, self.events = CheckpointStore(run_dir), EventLog(run_dir, run_id)
        self.config = {"run_id": run_id, "clients": dict(clients), "quorum": quorum,
                       "strategy": strategy, "seed": seed, "deadline_seconds": deadline_seconds,
                       "epsilon": epsilon, "identity": identity}
        if set(config_extra or {}) & self.config.keys():
            raise ValueError("Extra configuration cannot override protocol contract fields")
        self.config.update(config_extra or {})
        self.config_hash = canonical_hash(self.config)
        self.state, self.version, self.attempt_id = dict(initial_state), 0, 0
        self.results: list[dict] = []
        if resume:
            loaded = self.store.load()
            if (loaded["run_id"] != run_id or loaded["identity"] != identity
                    or loaded["config_hash"] != self.config_hash):
                raise ValueError("Resume contract/configuration mismatch")
            self.state, self.version, self.attempt_id = (
                loaded["state"], loaded["version"], loaded["attempt_id"])
            result_path = self.run_dir / "round_results.json"
            if result_path.exists():
                self.results = json.loads(result_path.read_text(encoding="utf-8"))
            metadata = loaded.get("metadata", {})
            if (metadata.get("status") in {"committed", "skipped"}
                    and metadata.get("attempt_id") == self.attempt_id
                    and not any(row["attempt_id"] == self.attempt_id for row in self.results)):
                # Recover a crash after durable B replacement but before the result
                # index was replaced. The checkpoint itself contains the decision.
                self.results.append({**metadata, "checkpoint_hash": loaded["b_hash"]})
                atomic_json(result_path, self.results)
                self.events.emit("checkpoint_result_recovered", attempt_id=self.attempt_id,
                                 version=self.version, status=metadata["status"])
            elif metadata.get("status") == "attempt_in_progress":
                self.events.emit("unfinished_attempt_recovered", attempt_id=self.attempt_id,
                                 version=self.version, status="skipped",
                                 reason="server_restart_before_commit",
                                 selected=metadata.get("selected", []))
        elif self.store.path.exists():
            raise FileExistsError("Run already has a checkpoint; choose resume or a fresh run directory")
        else:
            self._save({"status": "initialized"})
        atomic_json(self.run_dir / "config.json", self.config)
        self.active = False
        self.rejected: list[dict] = []

    def _save(self, metadata: dict) -> None:
        self.store.save(run_id=self.run_id, attempt_id=self.attempt_id, version=self.version,
                        state=self.state, identity=self.identity, config_hash=self.config_hash,
                        metadata=metadata)

    def begin(self, budgets: Mapping[str, int], *, now: float | None = None,
              manifest_hashes: Mapping[str, str] | None = None) -> dict[str, RoundRequest]:
        if self.active:
            raise RuntimeError("Attempt already active")
        if not budgets or not set(budgets) <= self.clients.keys():
            raise ValueError("Participants must be known clients")
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in budgets.values()):
            raise ValueError("Work budgets must be positive optimizer steps")
        self.attempt_id += 1
        self.started_at = time.time() if now is None else now
        self.started_monotonic = time.monotonic()
        self.deadline_unix = self.started_at + self.deadline_seconds
        self.base = dict(self.state)
        self.requests = {
            c: RoundRequest(self.run_id, self.attempt_id, self.version, hash_state(self.base),
                            self.identity["contract_hash"], self.identity["common_a_hash"],
                            self.identity["model_revision"], c, budget, self.deadline_unix,
                            self.seed, (manifest_hashes or {}).get(c, ""))
            for c, budget in budgets.items()
        }
        self.accepted: dict[str, ClientReply] = {}
        self.received_reply_bytes = 0
        self.seen: set[str] = set()
        self.rejected: list[dict] = []
        self.active = True
        # Persist consumed attempt before any work. A restart cannot reuse its ID.
        self._save({"status": "attempt_in_progress", "selected": sorted(budgets)})
        self.events.emit("attempt_started", attempt_id=self.attempt_id, version=self.version,
                         selected=sorted(budgets), base_b_hash=hash_state(self.base),
                         deadline_unix=self.deadline_unix, budgets=dict(budgets))
        return dict(self.requests)

    def reject(self, authenticated_client_id: str, reason: str) -> bool:
        if self.active and authenticated_client_id in self.requests:
            self.seen.add(authenticated_client_id)
        record = {"client_id": authenticated_client_id, "reason": reason}
        self.rejected.append(record)
        self.events.emit("reply_rejected", attempt_id=self.attempt_id, version=self.version, **record)
        return False

    def record_transport_reply(self, authenticated_client_id: str, serialized_bytes: int) -> None:
        self.received_reply_bytes += serialized_bytes
        self.events.emit("reply_received", attempt_id=self.attempt_id, version=self.version,
                         client_id=authenticated_client_id, serialized_reply_bytes=serialized_bytes)

    def receive(self, reply: ClientReply, *, authenticated_client_id: str,
                received_at: float | None = None) -> bool:
        client_id = authenticated_client_id
        if not self.active:
            return self.reject(client_id, "attempt_closed")
        if client_id not in self.requests:
            return self.reject(client_id, "unknown_or_unselected_client")
        if (time.time() if received_at is None else received_at) >= self.deadline_unix:
            return self.reject(client_id, "late_reply")
        if client_id in self.seen:
            return self.reject(client_id, "duplicate_reply")
        self.seen.add(client_id)
        expected = self.requests[client_id]
        for key in asdict(expected):
            if getattr(reply.request, key) != getattr(expected, key):
                return self.reject(client_id, f"{key}_mismatch")
        if reply.status != "completed":
            return self.reject(client_id, "unsuccessful_work_status")
        steps = completed_steps(reply.telemetry)
        if steps <= 0 or steps > expected.optimizer_steps:
            return self.reject(client_id, "invalid_completed_optimizer_steps")
        if state_schema(reply.delta) != state_schema(self.base):
            return self.reject(client_id, "tensor_schema_mismatch")
        if not finite_state(reply.delta):
            return self.reject(client_id, "nonfinite_delta")
        try:
            json.dumps(reply.telemetry, allow_nan=False)
        except (ValueError, TypeError):
            return self.reject(client_id, "invalid_telemetry")
        self.accepted[client_id] = reply
        self.events.emit("reply_accepted", attempt_id=self.attempt_id, version=self.version,
                         client_id=client_id, telemetry=reply.telemetry,
                         raw_delta_bytes=state_nbytes(reply.delta),
                         serialized_reply_bytes=reply.serialized_reply_bytes)
        return True

    def finish(self, *, now: float | None = None) -> dict:
        if not self.active:
            raise RuntimeError("No active attempt")
        self.active = False
        version_before = self.version
        result = {
            "run_id": self.run_id, "attempt_id": self.attempt_id,
            "version_before": version_before, "version_after": version_before,
            "status": "skipped", "accepted": sorted(self.accepted),
            "selected": sorted(self.requests),
            "absent": sorted(set(self.requests) - self.seen), "rejected": list(self.rejected),
            "quorum": self.quorum, "aggregation": None,
            "collection_seconds": max(0, now - self.started_at) if now is not None
            else time.monotonic() - self.started_monotonic,
            "clients": {c: r.telemetry for c, r in self.accepted.items()},
            "raw_b_bytes": state_nbytes(self.base),
            "serialized_reply_bytes": max(self.received_reply_bytes,
                                           sum(r.serialized_reply_bytes for r in self.accepted.values())),
        }
        aggregation_started = time.monotonic()
        if len(self.accepted) >= self.quorum:
            try:
                aggregated = aggregate(self.base, {c: r.delta for c, r in self.accepted.items()},
                                       strategy=self.strategy, sample_sizes=self.clients,
                                       epsilon=self.epsilon)
            except (ValueError, OverflowError) as error:
                result["aggregate_error"] = str(error)
            else:
                result.update(status="committed", version_after=version_before + 1,
                              aggregation=aggregated.telemetry,
                              aggregation_seconds=time.monotonic() - aggregation_started)
                # Commit durable state before exposing it as the current version.
                self.store.save(run_id=self.run_id, attempt_id=self.attempt_id,
                                version=version_before + 1, state=aggregated.state,
                                identity=self.identity, config_hash=self.config_hash, metadata=result)
                self.state, self.version = aggregated.state, version_before + 1
        if result["status"] == "skipped":
            self._save(result)
        result["checkpoint_hash"] = hash_state(self.state)
        self.results.append(result)
        atomic_json(self.run_dir / "round_results.json", self.results)
        self.events.emit("round_result", **{k: v for k, v in result.items() if k != "run_id"},
                         version=self.version)
        return result
