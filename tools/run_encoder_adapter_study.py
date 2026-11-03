"""Real validation-only MPNet B-plus-head training; centralized or sequential FL.

All encoder passes use original local text. Cached embeddings are used only for
an initial frozen-encoder parity check, never as adapter-training inputs.
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
from fedlora_edge.data import content_hash
from fedlora_edge.encoder_adapter import EncoderAdapterConfig, EncoderAdapterTrainer
from fedlora_edge.encoder_data import load_encoder_manifest, load_encoder_records, load_feature_split
from fedlora_edge.protocol import ClientReply, RoundCoordinator, validate_request
from fedlora_edge.state import hash_state, state_nbytes, subtract_states
from fedlora_edge.training import derive_client_seed


def run(config):
    import numpy as np
    output = Path(config["run_dir"])
    if output.exists():
        raise FileExistsError("Choose a fresh experiment directory.")
    for name in ("attempts", "evaluate_every", "quorum"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if config["mode"] not in ("centralized", "federated"):
        raise ValueError("Unknown experiment mode.")
    manifest = load_encoder_manifest(config["dataset_dir"])
    training_config = EncoderAdapterConfig(**config["training_config"])
    trainer = EncoderAdapterTrainer(config["asset_dir"], config["adapter_dir"],
                                    device=config["device"], config=training_config)
    if content_hash(manifest["encoder"]["files_sha256"]) != trainer.identity["asset_hash"]:
        raise ValueError("The prepared feature and live encoder assets differ.")
    initial = trainer.get_state()
    if any(v != 0 for name, tensor in initial.items() if ".lora_B." in name for v in tensor.values):
        raise ValueError("This experiment requires fresh zero B.")
    source_head = CheckpointStore(config["initial_head_dir"]).load()
    if (source_head["identity"].get("kind") != "frozen_mpnet_linear_head"
            or source_head["identity"]["encoder_identity_hash"] != manifest["encoder_identity_hash"]
            or source_head["identity"]["manifest_hash"] != manifest["manifest_hash"]):
        raise ValueError("Warm-start head does not belong to the same encoder/data variant.")
    trainer.set_head_state(source_head["state"])
    initial = trainer.get_state()
    clients = {str(k): load_encoder_records(config["dataset_dir"], str(k)) for k in config["budgets"]
               if str(k) not in set(map(str, config.get("omitted_clients", [])))}
    budgets = {k: config["budgets"][k] for k in clients}
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in budgets.values()):
        raise ValueError("Positive integer local step budgets required.")
    validation = load_encoder_records(config["dataset_dir"])
    frozen_before = trainer.frozen_hashes()
    plan = {**config, "run_id": output.name, "created_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": "sequential_encoder_adapter_federation" if config["mode"] == "federated" else "centralized_encoder_adapter_reference",
            "split": "validation", "identity": trainer.identity, "manifest_hash": manifest["manifest_hash"],
            "initial_state_hash": hash_state(initial), "initial_head_run": source_head["run_id"],
            "initial_head_hash": source_head["b_hash"], "initial_head_config_hash": source_head["config_hash"],
            "selection_rule": "Highest validation accuracy, then macro F1, earliest evaluated attempt; full horizon runs.",
            "centralized_schedule": "Pooled eligible training rows; one optimizer reset per round, summed client step budget. Equal work count need not imply identical examples or tokens.",
            "source_files_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                "tools/run_encoder_adapter_study.py", "fedlora_edge/encoder_adapter.py", "fedlora_edge/encoder_data.py",
                "fedlora_edge/protocol.py", "fedlora_edge/aggregation.py")},
            "interpretation": "Real frozen-A encoder B-plus-head optimization, validation only. Sequential simulation does not establish Flower or physical-device adapter transport. Warm head provenance is federated head-training history, not centrally fitted reference weights."}
    plan["config_hash"] = content_hash(plan)
    output.mkdir(parents=True)
    atomic_json(output/"plan.json", plan)
    # Check that switching from the cache to live text/model passes preserves the
    # starting classifier. Only zero-B evaluation may use this equivalence.
    Xval, _, _ = load_feature_split(config["dataset_dir"])
    cached_logits = Xval @ source_head["state"]["head.weight"].to_numpy().T + source_head["state"]["head.bias"].to_numpy()
    live_logits = trainer.predict_scores(validation)
    parity = {"max_abs_logit_difference": float(np.max(np.abs(cached_logits-live_logits))),
              "identical_predictions": bool(np.array_equal(cached_logits.argmax(1), live_logits.argmax(1)))}
    atomic_json(output/"initial_parity.json", parity)
    if not parity["identical_predictions"] or parity["max_abs_logit_difference"] > 1e-3:
        raise ValueError("Live zero-B encoder differs from cached head reference.")
    history, work, best = [], [], None
    coordinator = None
    started = time.perf_counter()

    def evaluate(attempt):
        nonlocal best
        metrics = trainer.evaluate(validation)
        version = coordinator.version if coordinator else attempt
        entry = {"attempt_id": attempt, "version": version, "split": "validation", "metrics": metrics,
                 "state_hash": hash_state(trainer.get_state())}
        history.append(entry)
        if best is None or (metrics["accuracy"], metrics["macro_f1"]) > (best["metrics"]["accuracy"], best["metrics"]["macro_f1"]):
            best = entry
            CheckpointStore(output/"selected").save(run_id=output.name, attempt_id=attempt, version=version,
                state=trainer.get_state(), identity=trainer.identity, config_hash=plan["config_hash"], metadata={"selection": best})
        atomic_json(output/"progress.json", {"status": "running", "attempt_id": attempt, "attempts": config["attempts"], "latest": entry, "selected": best})
        print(f"{output.name}: {attempt}/{config['attempts']} validation={metrics['accuracy']:.1%} worst={metrics['worst_domain_accuracy']:.1%}", flush=True)

    evaluate(0)
    if config["mode"] == "federated":
        coordinator = RoundCoordinator(run_dir=output/"protocol", run_id=output.name, initial_state=initial,
            identity=trainer.identity, clients={k: len(v) for k,v in clients.items()}, quorum=config["quorum"],
            strategy=config["strategy"], seed=config["seed"], deadline_seconds=config["deadline_seconds"],
            config_extra={"plan_config_hash":plan["config_hash"]})
    pooled = [r for rows in clients.values() for r in rows]
    for attempt in range(1, config["attempts"]+1):
        if coordinator:
            requests = coordinator.begin(budgets, manifest_hashes={k: content_hash(v) for k,v in clients.items()})
            incoming = dict(coordinator.state)
            for client, request in requests.items():
                try:
                    validate_request(request, incoming, trainer.identity)
                except ValueError as exc:
                    if "deadline_expired" not in str(exc):
                        raise
                    coordinator.reject(client, "queued_deadline_expired")
                    continue
                trainer.set_state(incoming)
                result = trainer.train(clients[client], budgets[client], derive_client_seed(config["seed"], attempt, client), request.deadline_unix)
                work.append({"attempt_id":attempt,"client_id":client,**result})
                delta = subtract_states(trainer.get_state(), incoming)
                coordinator.receive(ClientReply(request,delta,result["status"],result["telemetry"],0),authenticated_client_id=client)
            coordinator.finish()
            trainer.set_state(coordinator.state)
        else:
            result = trainer.train(pooled,sum(budgets.values()),derive_client_seed(config["seed"],attempt,"centralized"))
            if result["status"] != "completed":
                raise RuntimeError("Centralized block did not complete.")
            work.append({"attempt_id":attempt,"client_id":"centralized",**result})
        if attempt % config["evaluate_every"] == 0 or attempt == config["attempts"]:
            evaluate(attempt)
    frozen_after = trainer.frozen_hashes()
    if frozen_after != frozen_before:
        raise AssertionError("Frozen base or common A changed during experiment.")
    CheckpointStore(output/"final").save(run_id=output.name,attempt_id=config["attempts"],
        version=coordinator.version if coordinator else config["attempts"],state=trainer.get_state(),
        identity=trainer.identity,config_hash=plan["config_hash"],metadata={"status":"completed"})
    summary = {"status":"completed","plan":plan,"selected":best,"final":history[-1],"history":history,"work":work,
               "wall_seconds":time.perf_counter()-started,"frozen_before":frozen_before,"frozen_after":frozen_after,
               "initial_parity":parity,"raw_update_bytes":state_nbytes(initial),
               "committed_versions":coordinator.version if coordinator else config["attempts"]}
    atomic_json(output/"summary.json",summary)
    atomic_json(output/"progress.json",{"status":"completed","attempt_id":config["attempts"],"selected":best,"final":history[-1]})
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")))
