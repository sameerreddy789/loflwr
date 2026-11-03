"""Frozen 33-cell, validation-only CPU head study; mathematical federation, no transport.

Every cell loads its seed's saved initial tensors. Checkpointing every 50 attempts
avoids rewriting a growing protocol log on every step; actual Flower recovery is
a separate experiment. Completed cells are reused; incomplete cells fail closed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fedlora_edge.data import write_json, content_hash
from fedlora_edge.encoder_data import load_encoder_manifest, load_feature_split, sha256
from fedlora_edge.state import state_from_dict, state_to_dict, hash_state, subtract_states, state_nbytes

HEAD = {"learning_rate": 20, "batch_size": 1024, "weight_decay": 3.6231884057971014e-05,
        "optimizer": "sgd", "gradient_clip": 10., "cpu_threads": 1, "class_balance": True}
SOURCES = ["tools/run_encoder_matrix.py", "tools/run_encoder_study.py", "fedlora_edge/encoder_head.py",
           "fedlora_edge/encoder_data.py", "fedlora_edge/aggregation.py", "fedlora_edge/state.py",
           "fedlora_edge/training.py", "fedlora_edge/data.py", "fedlora_edge/checkpoint.py"]


def read(path):
    return json.loads(Path(path).read_text())


def cells():
    result = []
    for seed in (42, 43, 44):
        for regime, work in (("expanded", "equal"), ("expanded", "heterogeneous"), ("imbalanced", "heterogeneous")):
            budgets = dict(zip(map(str, range(5)), [1]*5 if work == "equal" else [1, 4, 2, 2, 1]))
            for strategy in ("sample", "equal", "equal_clipped"):
                result.append({"id": f"s{seed}-{regime}-{work}-{strategy}", "seed": seed, "regime": regime,
                    "work": work, "strategy": strategy, "budgets": budgets, "omitted": [], "unavailable": []})
        result.append({"id": f"s{seed}-expanded-heterogeneous-equal-omit4", "seed": seed,
            "regime": "expanded", "work": "heterogeneous", "strategy": "equal",
            "budgets": dict(zip(map(str, range(5)), [1, 4, 2, 2, 1])), "omitted": ["4"], "unavailable": []})
        result.append({"id": f"s{seed}-imbalanced-heterogeneous-equal_clipped-unavailable1", "seed": seed,
            "regime": "imbalanced", "work": "heterogeneous", "strategy": "equal_clipped",
            "budgets": dict(zip(map(str, range(5)), [1, 4, 2, 2, 1])), "omitted": [],
            "unavailable": [{"client_id": "1", "start": start, "end": start+9} for start in (101, 251, 401)]})
    return result


def prepare(output):
    from fedlora_edge.encoder_head import HeadTrainer, HeadConfig
    import torch
    if output.exists():
        raise FileExistsError("Use a fresh matrix output directory")
    datasets = {"expanded": "assets/encoder-bank20-v1", "imbalanced": "assets/encoder-bank20-imbalance-v1"}
    files = {name: sha256(ROOT / name) for name in SOURCES}
    counts = {}
    for regime, name in datasets.items():
        root = ROOT / name
        manifest = load_encoder_manifest(root)
        files[f"{name}/manifest.json"] = sha256(root / "manifest.json")
        counts[regime] = {c: e["count"] for c, e in manifest["clients"].items()}
        for entry in [manifest["validation"], *manifest["clients"].values()]:
            files[f"{name}/{entry['file']}"] = sha256(root / entry["file"])
    output.mkdir(parents=True)
    initial = {}
    for seed in (42, 43, 44):
        trainer = HeadTrainer(seed=seed, config=HeadConfig(**HEAD))
        state = trainer.get_state()
        path = output / f"initial-{seed}.json"
        write_json(path, {"state": state_to_dict(state), "state_hash": hash_state(state)})
        initial[str(seed)] = {"file": path.name, "sha256": sha256(path), "state_hash": hash_state(state)}
    plan = {"kind": "encoder_matrix_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "cells": cells(), "attempts": 500, "evaluate_every": 50, "quorum": 3, "head_config": HEAD,
        "datasets": datasets, "client_counts": counts, "initial": initial, "files": files,
        "python": sys.version, "torch": torch.__version__, "split": "reused_validation",
        "selection_rule": "highest evaluated validation accuracy, then macro F1, earliest attempt; final reported separately",
        "scope": "Sequential in-memory mathematical federation with real head optimization. No Flower, network, or physical-device claim.",
        "unavailability": "Predeclared skipped work, not actual process termination. Client random streams remain independent."}
    plan["plan_hash"] = content_hash(plan)
    write_json(output / "plan.json", plan)
    return plan


def verify(plan, output):
    import torch
    if content_hash({k:v for k,v in plan.items() if k != "plan_hash"}) != plan["plan_hash"]:
        raise ValueError("Matrix lock changed")
    if plan["python"] != sys.version or plan["torch"] != torch.__version__:
        raise ValueError("Matrix runtime differs from its frozen environment")
    for name, digest in plan["files"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Matrix source or data changed: {name}")
    for entry in plan["initial"].values():
        if sha256(output / entry["file"]) != entry["sha256"]:
            raise ValueError("Initial tensor file changed")


def run_cell(output_string, cell):
    from fedlora_edge.encoder_head import HeadTrainer, HeadConfig
    from fedlora_edge.aggregation import aggregate
    from fedlora_edge.training import derive_client_seed
    from tools.run_encoder_study import evaluate
    output = Path(output_string)
    plan = read(output / "plan.json")
    verify(plan, output)
    run = output / "cells" / cell["id"]
    if (run / "result.json").is_file():
        saved = read(run / "result.json")
        if saved["plan_hash"] != plan["plan_hash"] or saved["cell"] != cell:
            raise ValueError("Completed cell identity mismatch")
        return saved
    if run.exists():
        raise RuntimeError(f"Incomplete cell must be investigated, not overwritten: {cell['id']}")
    run.mkdir(parents=True)
    write_json(run / "started.json", {"cell": cell, "plan_hash": plan["plan_hash"]})
    trainer = HeadTrainer(seed=cell["seed"], config=HeadConfig(**plan["head_config"]))
    envelope = read(output / plan["initial"][str(cell["seed"])]["file"])
    state = state_from_dict(envelope["state"])
    if hash_state(state) != plan["initial"][str(cell["seed"])]["state_hash"]:
        raise ValueError("Wrong initial tensors")
    root = ROOT / plan["datasets"][cell["regime"]]
    manifest = load_encoder_manifest(root)
    Xval, yval, _ = load_feature_split(root)
    clients = {c: load_feature_split(root, c) for c in cell["budgets"] if c not in cell["omitted"]}
    history, rounds, completed, participation = [], [], Counter(), Counter()
    started = time.perf_counter()
    peak, version, selected, best_key = 0, 0, None, (-1., -1.)

    def score(attempt):
        nonlocal selected, best_key
        trainer.set_state(state)
        row = {"attempt_id": attempt, "version": version, "state_hash": hash_state(state),
               "metrics": evaluate(trainer, Xval, yval)}
        history.append(row)
        key = (row["metrics"]["accuracy"], row["metrics"]["macro_f1"])
        if key > best_key:
            selected, best_key = row, key
            write_json(run / "selected.json", {"state": state_to_dict(state), "selection": row})
        write_json(run / "progress.json", {"attempt_id": attempt, "selected": selected, "latest": row})
        print(f"{cell['id']}: {attempt}/{plan['attempts']} validation={row['metrics']['accuracy']:.3f}", flush=True)
    score(0)
    for attempt in range(1, plan["attempts"]+1):
        updates, work = {}, {}
        for c in sorted(clients):
            if any(entry["client_id"] == c and entry["start"] <= attempt <= entry["end"] for entry in cell["unavailable"]):
                continue
            trainer.set_state(state)
            X, y, _ = clients[c]
            telemetry = trainer.train(X, y, cell["budgets"][c], derive_client_seed(cell["seed"], attempt, c))
            updates[c] = subtract_states(trainer.get_state(), state)
            work[c] = telemetry
            completed[c] += telemetry["optimizer_steps"]
            participation[c] += 1
            peak = max(peak, telemetry.get("peak_rss_bytes") or 0)
        if len(updates) >= plan["quorum"]:
            result = aggregate(state, updates, strategy=cell["strategy"],
                               sample_sizes={c: manifest["clients"][c]["count"] for c in clients})
            state, version = result.state, version+1
            aggregation = result.telemetry
        else:
            aggregation = None
        rounds.append({"attempt_id": attempt, "version": version, "accepted": sorted(updates),
            "aggregation": aggregation, "completed_steps": {c:t["optimizer_steps"] for c,t in work.items()}})
        if attempt % plan["evaluate_every"] == 0:
            score(attempt)
    result = {"status": "completed", "plan_hash": plan["plan_hash"], "cell": cell,
        "initial_state_hash": envelope["state_hash"], "selected": selected, "final": history[-1],
        "history": history, "attempts": plan["attempts"], "committed_versions": version,
        "completed_steps": dict(completed), "participation": dict(participation), "peak_sampled_rss_bytes": peak,
        "raw_update_bytes_per_direction": state_nbytes(state), "wall_seconds": time.perf_counter()-started,
        "scope": plan["scope"]}
    write_json(run / "rounds.json", rounds)
    write_json(run / "final.json", {"state": state_to_dict(state), "state_hash": hash_state(state)})
    write_json(run / "result.json", result)
    return result


def report(output):
    plan = read(output / "plan.json")
    results, missing, groups = [], [], {}
    for cell in plan["cells"]:
        path = output / "cells" / cell["id"] / "result.json"
        if not path.exists():
            missing.append(cell["id"])
            continue
        result = read(path)
        if result["plan_hash"] != plan["plan_hash"] or result["cell"] != cell:
            raise ValueError("Mixed matrix results")
        results.append(result)
        groups.setdefault(cell["id"].split("-", 1)[1], []).append(result)
    stats = {}
    for name, group in groups.items():
        stats[name] = {"seeds": [r["cell"]["seed"] for r in group]}
        for endpoint in ("final", "selected"):
            stats[name][endpoint] = {}
            for metric in ("accuracy", "macro_f1", "worst_domain_accuracy", "domain_gap", "cross_entropy"):
                values = [r[endpoint]["metrics"][metric] for r in group]
                stats[name][endpoint][metric] = {"mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values) if len(values)>1 else None, "values": values}
    paired = {}
    indexed = {r["cell"]["id"]:r for r in results}
    for suffix, reference in (("expanded-heterogeneous-equal-omit4", "expanded-heterogeneous-equal"),
                              ("imbalanced-heterogeneous-equal_clipped-unavailable1", "imbalanced-heterogeneous-equal_clipped")):
        paired[suffix] = [{"seed": s, "final_accuracy_difference_control_minus_variant":
            indexed[f"s{s}-{reference}"]["final"]["metrics"]["accuracy"] - indexed[f"s{s}-{suffix}"]["final"]["metrics"]["accuracy"]}
            for s in (42,43,44) if f"s{s}-{reference}" in indexed and f"s{s}-{suffix}" in indexed]
    result = {"status": "completed" if not missing else "incomplete", "plan_hash": plan["plan_hash"],
        "completed":len(results), "expected":len(plan["cells"]), "missing":missing,
        "groups":stats, "paired":paired, "results":results, "scope":plan["scope"],
        "interpretation":"Three development seeds describe observed variability, not an independent accuracy confirmation or significance guarantee."}
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "report"])
    parser.add_argument("--output", type=Path, default=ROOT/"runs/encoder-matrix-v1")
    parser.add_argument("--workers", type=int, choices=[1,2], default=2)
    args=parser.parse_args()
    output=args.output.resolve()
    if args.action == "prepare":
        print(json.dumps({"plan_hash":prepare(output)["plan_hash"]}))
    elif args.action == "run":
        plan=read(output/"plan.json")
        verify(plan,output)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending=[pool.submit(run_cell,str(output),cell) for cell in plan["cells"]]
            for task in as_completed(pending):
                result=task.result()
                print("COMPLETED " + result["cell"]["id"],flush=True)
        print(json.dumps({"status":report(output)["status"]}))
    else:
        result=report(output)
        print(json.dumps({"status":result["status"],"completed":result["completed"]}))


if __name__ == "__main__":
    main()
