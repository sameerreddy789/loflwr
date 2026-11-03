"""Run a validation-only CPU frozen-feature head experiment with actual optimization.

Sequential federation is a mathematical/process prototype, not a Flower/network
test. Each local update uses its own feature partition; centralized mode is an
explicit reference. The sealed confirmation set is never loaded here.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fedlora_edge.checkpoint import CheckpointStore, atomic_json
from fedlora_edge.data import INTENTS, content_hash
from fedlora_edge.encoder_data import load_encoder_manifest, load_feature_split
from fedlora_edge.protocol import ClientReply, RoundCoordinator, validate_request
from fedlora_edge.state import hash_state, state_nbytes, state_schema, subtract_states
from fedlora_edge.training import classification_metrics, derive_client_seed


def evaluate(trainer, X, y):
    import numpy as np
    scores = trainer.predict_scores(X)
    predicted = scores.argmax(axis=1)
    result = classification_metrics([INTENTS[int(i)] for i in y], [INTENTS[int(i)] for i in predicted])
    shifted = scores.astype(np.float64) - scores.max(axis=1, keepdims=True)
    result["cross_entropy"] = float((-shifted[np.arange(len(y)), y] + np.log(np.exp(shifted).sum(axis=1))).mean())
    return result


def make_identity(manifest, initial, head_config):
    identity = {"kind": "frozen_mpnet_linear_head", "model_revision": manifest["model_revision"],
                "common_a_hash": manifest["encoder_identity_hash"], "encoder_identity_hash": manifest["encoder_identity_hash"],
                "feature_dim": manifest["feature_dim"], "labels": manifest["labels"], "head_schema": state_schema(initial),
                "head_config": head_config, "manifest_hash": manifest["manifest_hash"],
                "protocol_alias_note": "Historical B fields carry classifier-head state; common_a_hash binds the frozen encoder identity. No LoRA A/B are trained in this variant."}
    identity["contract_hash"] = content_hash(identity)
    return identity


def run(config):
    from fedlora_edge.encoder_head import HeadConfig, HeadTrainer
    root, output = Path(config["dataset_dir"]), Path(config["run_dir"])
    if output.exists():
        raise FileExistsError("Choose a fresh run directory.")
    if config.get("mode", "federated") not in {"federated", "centralized"}:
        raise ValueError("Invalid experiment mode.")
    manifest = load_encoder_manifest(root)
    head_config = HeadConfig(**config["head_config"])
    for name, default, lower in (("seed", 42, 0), ("attempts", None, 1), ("evaluate_every", 10, 1), ("quorum", 3, 1)):
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}.")
    seed, rounds = config.get("seed", 42), config["attempts"]
    omitted = set(map(str, config.get("omitted_clients", [])))
    budgets = {str(k): v for k, v in config["budgets"].items() if str(k) not in omitted}
    if not budgets or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in budgets.values()):
        raise ValueError("Positive local work is required.")
    trainer = HeadTrainer(manifest["feature_dim"], len(INTENTS), seed, head_config)
    initial = trainer.get_state()
    identity = make_identity(manifest, initial, asdict(head_config))
    Xval, yval, _ = load_feature_split(root)
    # Independent local partitions; the sequential simulator schedules their reads.
    clients = {client: load_feature_split(root, client) for client in budgets}
    plan = {**config, "run_id": output.name, "created_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": "sequential_federated_feature_training" if config.get("mode", "federated") == "federated" else "centralized_feature_reference",
            "identity": identity, "initial_state_hash": hash_state(initial), "split": "validation",
            "selection_rule": "Highest validation accuracy, then macro F1, earliest attempt; full horizon still runs.",
            "source_files_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                "tools/run_encoder_study.py", "fedlora_edge/encoder_head.py", "fedlora_edge/encoder_data.py",
                "fedlora_edge/aggregation.py", "fedlora_edge/protocol.py")},
            "interpretation": "Frozen-encoder head training, not LoRA. No final/confirmation test access. Sequential federation is not a Flower transport or physical-device measurement."}
    plan["config_hash"] = content_hash(plan)
    output.mkdir(parents=True)
    atomic_json(output / "plan.json", plan)
    started = time.perf_counter()
    history, work = [], []
    best_key = (-1., -1.)
    best = None
    coordinator = None

    def score(attempt):
        nonlocal best_key, best
        metrics = evaluate(trainer, Xval, yval)
        version = coordinator.version if coordinator else attempt
        row = {"attempt_id": attempt, "version": version, "split": "validation", "state_hash": hash_state(trainer.get_state()), "metrics": metrics}
        history.append(row)
        key = (metrics["accuracy"], metrics["macro_f1"])
        if key > best_key:
            best_key, best = key, row
            CheckpointStore(output / "selected").save(run_id=plan["run_id"], attempt_id=attempt, version=version,
                state=trainer.get_state(), identity=identity, config_hash=plan["config_hash"], metadata={"selection": row})
        atomic_json(output / "progress.json", {"status": "running", "attempt_id": attempt, "attempts": rounds,
                    "latest": row, "selected": best})
        print(f"{output.name}: attempt {attempt}/{rounds}, validation {metrics['accuracy']:.1%}, worst domain {metrics['worst_domain_accuracy']:.1%}", flush=True)

    score(0)
    mode = config.get("mode", "federated")
    if mode == "federated":
        coordinator = RoundCoordinator(run_dir=output / "protocol", run_id=plan["run_id"], initial_state=initial,
            identity=identity, clients={k: manifest["clients"][k]["count"] for k in budgets},
            quorum=int(config.get("quorum", 3)), strategy=config["strategy"], seed=seed,
            deadline_seconds=float(config.get("deadline_seconds", 60)), config_extra={"plan_config_hash": plan["config_hash"]})
    schedule = {(int(e["attempt_id"]), str(e["client_id"])) for e in config.get("simulated_failures", [])}
    for attempt in range(1, rounds + 1):
        if coordinator:
            requests = coordinator.begin(budgets, manifest_hashes={k: manifest["clients"][k]["records_hash"] for k in budgets})
            incoming = coordinator.state
        for client in sorted(budgets):
            X, y, _ = clients[client]
            if coordinator:
                request = requests[client]
                if (attempt, client) in schedule:
                    coordinator.reject(client, "declared_simulated_unavailability")
                    continue
                try:
                    validate_request(request, incoming, identity)
                except ValueError as exc:
                    if "deadline_expired" not in str(exc):
                        raise
                    coordinator.reject(client, "local_schedule_deadline_expired")
                    continue
                trainer.set_state(incoming)
            try:
                telemetry = trainer.train(X, y, budgets[client], derive_client_seed(seed, attempt, client),
                                          deadline_unix=request.deadline_unix if coordinator else None)
            except TimeoutError:
                if not coordinator:
                    raise
                coordinator.reject(client, "local_work_deadline_expired")
                continue
            work.append({"attempt_id": attempt, "client_id": client, **telemetry})
            if coordinator:
                delta = subtract_states(trainer.get_state(), incoming)
                telemetry["transport_bytes_measured"] = False
                coordinator.receive(ClientReply(request, delta, telemetry.get("status", "completed"), telemetry,
                                                0), authenticated_client_id=client)
        if coordinator:
            coordinator.finish()
            trainer.set_state(coordinator.state)
        if attempt % int(config.get("evaluate_every", 10)) == 0 or attempt == rounds:
            score(attempt)
    CheckpointStore(output / "final").save(run_id=plan["run_id"], attempt_id=rounds,
        version=coordinator.version if coordinator else rounds, state=trainer.get_state(),
        identity=identity, config_hash=plan["config_hash"], metadata={"status": "completed"})
    result = {"status": "completed", "plan": plan, "selected": best, "history": history,
              "final": history[-1], "work": work, "raw_update_bytes": state_nbytes(initial),
              "wall_seconds": time.perf_counter() - started,
              "attempts": rounds, "committed_versions": coordinator.version if coordinator else rounds}
    atomic_json(output / "summary.json", result)
    atomic_json(output / "progress.json", {"status": "completed", "attempt_id": rounds, "selected": best, "final": history[-1]})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")))
