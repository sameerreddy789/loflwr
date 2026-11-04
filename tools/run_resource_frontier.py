"""Fixed-work, validation-only cached MPNet head accuracy/resource study.

Run prepare before run. Each cell runs in a fresh Linux CPU process, serially.
This is numerical federation, not Flower, physical devices, LoRA, or test scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fedlora_edge.data import content_hash, write_json
from fedlora_edge.encoder_data import load_encoder_manifest, load_feature_split, sha256
from fedlora_edge.state import hash_state, state_from_dict, state_nbytes, state_to_dict, subtract_states
from tools.run_encoder_matrix import HEAD

DATASET = "assets/encoder-bank20-v1"
PUBLIC = ROOT / "research/results/resource-frontier-v1"
OUTPUT = ROOT / "runs/resource-frontier-v1"
SEEDS = (42, 43, 44)
LOCAL_STEPS = (1, 2, 4)
STEPS_PER_CLIENT = 500
CHECKPOINTS = (0, 100, 200, 300, 400, 500)
CLIENTS = tuple(map(str, range(5)))
THREAD_ENV = {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
SOURCES = (
    "tools/run_resource_frontier.py", "tests/test_resource_frontier.py",
    "tools/run_encoder_matrix.py", "tools/run_encoder_study.py",
    "fedlora_edge/encoder_head.py", "fedlora_edge/encoder_data.py",
    "fedlora_edge/aggregation.py", "fedlora_edge/state.py", "fedlora_edge/training.py",
    "fedlora_edge/data.py", "fedlora_edge/checkpoint.py", "fedlora_edge/protocol.py",
)
SCOPE = ("Real CPU optimization of a linear classifier on cached frozen MPNet features; "
         "five sequential numerical clients on one host. Reused 200-query validation only. "
         "No LoRA, Flower runtime, physical-device, fresh test, or wire-traffic claim.")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def cells():
    # Rotate order across seeds to avoid assigning the same schedule to each
    # position. This is predeclared, not reordered after observing timings.
    orders = ((1, 2, 4), (2, 4, 1), (4, 1, 2))
    return [{"id": f"s{seed}-local{steps}", "seed": seed, "local_steps": steps,
             "rounds": STEPS_PER_CLIENT // steps}
            for seed, order in zip(SEEDS, orders) for steps in order]


def payload_bytes(raw_head_bytes, rounds, clients=5):
    """Arithmetic FP32 payload estimate: one head down and one delta up/client/round."""
    positive_int(raw_head_bytes, "raw_head_bytes")
    positive_int(rounds, "rounds", 0)
    positive_int(clients, "clients")
    one_direction = raw_head_bytes * rounds * clients
    return {"estimated_download_bytes": one_direction, "estimated_upload_bytes": one_direction,
            "estimated_total_bytes": 2 * one_direction}


def validate_design(plan):
    if (plan["kind"] != "resource_frontier_v1" or plan["cells"] != cells()
            or plan["head_config"] != HEAD or plan["head_config"]["cpu_threads"] != 1
            or plan["dataset"] != DATASET or plan["strategy"] != "equal"
            or plan["checkpoints_steps_per_client"] != list(CHECKPOINTS)
            or plan["steps_per_client"] != STEPS_PER_CLIENT
            or plan["split"] != "reused_validation" or plan["workers"] != 1):
        raise ValueError("The frozen resource-frontier design changed")
    counts = plan["client_counts"]
    if set(counts) != set(CLIENTS):
        raise ValueError("Exactly the five declared clients are required")
    for client, count in counts.items():
        positive_int(count, f"client {client} count")
        if count > HEAD["batch_size"]:
            raise ValueError("This matched-example design requires full local batches")
    for cell in plan["cells"]:
        for name in ("seed", "local_steps", "rounds"):
            positive_int(cell[name], name)
        if cell["rounds"] * cell["local_steps"] != STEPS_PER_CLIENT:
            raise ValueError("Unequal completed-step budget")
    if set(plan["initial"]) != set(map(str, SEEDS)):
        raise ValueError("All three saved seed initializations are required")


def runtime_identity():
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "cpu": cpu, "logical_cpus": os.cpu_count(),
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "psutil")}}


def prepare(output, public):
    from fedlora_edge.encoder_head import HeadConfig, HeadTrainer
    if output.exists() or (public / "plan.json").exists():
        raise FileExistsError("Use a fresh output and public plan; existing evidence is never overwritten")
    if sys.platform != "linux":
        raise RuntimeError("Use the pinned WSL Linux environment for measured process RSS")
    manifest = load_encoder_manifest(ROOT / DATASET)
    files = {name: sha256(ROOT / name) for name in SOURCES}
    files[f"{DATASET}/manifest.json"] = sha256(ROOT / DATASET / "manifest.json")
    source_ids = set()
    for client in (*CLIENTS, None):
        _, _, ids = load_feature_split(ROOT / DATASET, client)
        if source_ids.intersection(ids.tolist()):
            raise ValueError("Training/validation source IDs overlap")
        source_ids.update(ids.tolist())
        entry = manifest["validation"] if client is None else manifest["clients"][client]
        files[f"{DATASET}/{entry['file']}"] = sha256(ROOT / DATASET / entry["file"])
    output.mkdir(parents=True)
    public.mkdir(parents=True, exist_ok=True)
    initial = {}
    for seed in SEEDS:
        state = HeadTrainer(seed=seed, config=HeadConfig(**HEAD)).get_state()
        path = output / f"initial-{seed}.json"
        write_json(path, {"state": state_to_dict(state), "state_hash": hash_state(state)})
        initial[str(seed)] = {"file": path.name, "sha256": sha256(path), "state_hash": hash_state(state)}
    baseline = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    plan = {"kind": "resource_frontier_v1", "run_id": "resource-frontier-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(), "baseline_commit": baseline,
            "cells": cells(), "head_config": HEAD, "strategy": "equal", "workers": 1,
            "dataset": DATASET, "manifest_hash": manifest["manifest_hash"],
            "encoder_identity_hash": manifest["encoder_identity_hash"], "model_revision": manifest["model_revision"],
            "client_counts": {c: manifest["clients"][c]["count"] for c in CLIENTS},
            "validation_count": manifest["validation"]["count"], "split": "reused_validation",
            "steps_per_client": STEPS_PER_CLIENT, "checkpoints_steps_per_client": list(CHECKPOINTS),
            "initial": initial, "files_sha256": files, "runtime": runtime_identity(), "thread_environment": THREAD_ENV,
            "scope": SCOPE, "selection_rule": "Report all six checkpoints and final cells. No selected checkpoint or deployment change.",
            "randomness": "Saved initial tensors by seed; derive_client_seed(seed, round, client). Full batches do not sample rows.",
            "optimizer_reset": "Existing HeadTrainer resets SGD at every local call; no momentum. Local epochs between averages vary.",
            "timing": "Fresh serial subprocesses, CPU threads 1. Process wall includes startup and exit. Cell wall includes source/data verification, loading, optimization, aggregation, evaluation and artifact writing before result serialization. Local train and optimization times are summed trainer telemetry; aggregation/evaluation timed separately. No encoder inference or feature preparation.",
            "memory": "Linux os.wait4 child rusage.ru_maxrss * 1024, complete child-process lifetime high-water RSS collected by the serial supervisor after exit. Child also records an intermediate getrusage high-water before result serialization. Includes imports and all five feature caches; excludes parent, other processes, encoder preparation and physical devices.",
            "communication": "Calculated raw FP32 payload only: 2 * 5 * rounds * head_bytes. One full head download and delta upload per client/round. Includes initial round head distribution; excludes encoder/data transfer, metadata, transport overhead, retries and any final deployment. No actual serialized/wire traffic.",
            "dominance": "Strict Pareto: accuracy >= and payload <=, at least one strict. Also show observed accuracy/payload/process-wall/RSS frontier; timing/memory are single measurements per seed, not causal hardware claims.",
            "interpretation": "Three development seeds and reused validation cannot establish fresh test accuracy or statistical significance. Equal optimizer steps do not equal arithmetic/communication work: aggregation and trainer-call overhead vary."}
    validate_design(plan)
    plan["plan_hash"] = content_hash(plan)
    write_json(output / "plan.json", plan)
    write_json(public / "plan.json", plan)
    return plan


def verify(plan, output, public, *, root=ROOT, check_runtime=True):
    if content_hash({k: v for k, v in plan.items() if k != "plan_hash"}) != plan["plan_hash"]:
        raise ValueError("Resource plan hash mismatch")
    validate_design(plan)
    if read(public / "plan.json") != plan:
        raise ValueError("Public plan differs from the pre-run frozen plan")
    if check_runtime and runtime_identity() != plan["runtime"]:
        raise ValueError("Frozen runtime changed")
    if not set(SOURCES).issubset(plan["files_sha256"]):
        raise ValueError("Required source identities are missing")
    for name, digest in plan["files_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
            raise ValueError(f"Source or validation/training data changed: {name}")
    for entry in plan["initial"].values():
        path = output / entry["file"]
        if not path.resolve().is_relative_to(output.resolve()) or sha256(path) != entry["sha256"]:
            raise ValueError("Initial tensor file changed")
        envelope = read(path)
        if envelope["state_hash"] != entry["state_hash"] or hash_state(state_from_dict(envelope["state"])) != entry["state_hash"]:
            raise ValueError("Initial tensor identity mismatch")


def validate_work(telemetry, steps, examples):
    if (telemetry["optimizer_steps"] != steps or telemetry["completed_optimizer_steps"] != steps
            or telemetry["examples_processed"] != steps * examples
            or telemetry["manifest_examples"] != examples):
        raise ValueError("Measured work differs from the fixed full-batch budget")


def validate_result(result, plan, cell):
    if (result["status"] != "completed" or result["plan_hash"] != plan["plan_hash"] or result["cell"] != cell
            or result["initial_state_hash"] != plan["initial"][str(cell["seed"])]["state_hash"]):
        raise ValueError("Mixed or incomplete cell identity")
    if (result["completed_steps"] != {c: STEPS_PER_CLIENT for c in CLIENTS}
            or result["examples_processed"] != {c: STEPS_PER_CLIENT * plan["client_counts"][c] for c in CLIENTS}
            or result["participation"] != {c: cell["rounds"] for c in CLIENTS}
            or [r["steps_per_client"] for r in result["history"]] != list(CHECKPOINTS)):
        raise ValueError("Cell did not complete the matched work/trajectory")


def run_cell(output, public, cell_id):
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    from fedlora_edge.encoder_head import HeadConfig, HeadTrainer
    from fedlora_edge.aggregation import aggregate
    from fedlora_edge.training import derive_client_seed
    from tools.run_encoder_study import evaluate
    import resource
    plan = read(output / "plan.json")
    verify(plan, output, public)
    matches = [c for c in plan["cells"] if c["id"] == cell_id]
    if len(matches) != 1:
        raise ValueError("Unknown frozen cell")
    cell = matches[0]
    run = output / "cells" / cell_id
    if run.exists():
        raise FileExistsError("A fresh cell directory is required; investigate incomplete runs")
    run.mkdir(parents=True)
    write_json(run / "started.json", {"cell": cell, "plan_hash": plan["plan_hash"], "started_utc": started_utc, "pid": os.getpid()})
    trainer = HeadTrainer(seed=cell["seed"], config=HeadConfig(**plan["head_config"]))
    envelope = read(output / plan["initial"][str(cell["seed"])]["file"])
    state = state_from_dict(envelope["state"])
    root = ROOT / plan["dataset"]
    Xval, yval, _ = load_feature_split(root)
    clients = {c: load_feature_split(root, c) for c in CLIENTS}
    raw_bytes = state_nbytes(state)
    totals, examples, participation = Counter(), Counter(), Counter()
    timings = {name: 0.0 for name in ("local_train_seconds", "optimization_seconds", "aggregation_seconds", "evaluation_seconds")}
    history = []

    def score(round_id):
        tick = time.perf_counter()
        trainer.set_state(state)
        metrics = evaluate(trainer, Xval, yval)
        timings["evaluation_seconds"] += time.perf_counter() - tick
        row = {"round": round_id, "steps_per_client": round_id * cell["local_steps"],
               "total_optimizer_steps": sum(totals.values()), "total_examples_processed": sum(examples.values()),
               "state_hash": hash_state(state), "metrics": metrics, "cell_elapsed_seconds": time.perf_counter() - started,
               "cumulative_timings": dict(timings), **payload_bytes(raw_bytes, round_id)}
        history.append(row)
        write_json(run / "progress.json", row)
        print(f"{cell_id}: {row['steps_per_client']}/500 steps/client, validation={metrics['accuracy']:.3f}", flush=True)

    score(0)
    with (run / "rounds.jsonl").open("w", encoding="utf-8") as log:
        for round_id in range(1, cell["rounds"] + 1):
            updates = {}
            for client in CLIENTS:
                trainer.set_state(state)
                X, y, _ = clients[client]
                telemetry = trainer.train(X, y, cell["local_steps"], derive_client_seed(cell["seed"], round_id, client))
                validate_work(telemetry, cell["local_steps"], len(X))
                updates[client] = subtract_states(trainer.get_state(), state)
                totals[client] += telemetry["optimizer_steps"]
                examples[client] += telemetry["examples_processed"]
                participation[client] += 1
                for name in ("local_train_seconds", "optimization_seconds"):
                    timings[name] += telemetry[name]
            tick = time.perf_counter()
            combined = aggregate(state, updates, strategy="equal", sample_sizes=plan["client_counts"])
            timings["aggregation_seconds"] += time.perf_counter() - tick
            state = combined.state
            log.write(json.dumps({"round": round_id, "steps_per_client": round_id * cell["local_steps"],
                                  "aggregation": combined.telemetry}, separators=(",", ":")) + "\n")
            if round_id * cell["local_steps"] in CHECKPOINTS:
                score(round_id)
    write_json(run / "final-state.json", {"state_hash": hash_state(state), "state": state_to_dict(state)})
    result = {"status": "completed", "plan_hash": plan["plan_hash"], "cell": cell,
              "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
              "initial_state_hash": envelope["state_hash"], "history": history,
              "completed_steps": dict(totals), "examples_processed": dict(examples), "participation": dict(participation),
              "raw_head_bytes": raw_bytes, "peak_process_rss_through_final_state_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
              "cell_wall_seconds": time.perf_counter() - started, **timings,
              **payload_bytes(raw_bytes, cell["rounds"]), "scope": SCOPE}
    validate_result(result, plan, cell)
    write_json(run / "result.json", result)
    return result


def stats(values):
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("Finite nonempty observations are required")
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None, "values": values}


def dominates(left, right, maximize, minimize):
    weak = all(left[k] >= right[k] for k in maximize) and all(left[k] <= right[k] for k in minimize)
    strict = any(left[k] > right[k] for k in maximize) or any(left[k] < right[k] for k in minimize)
    return weak and strict


def summarize(results):
    groups = {}
    quality_names = ("accuracy", "macro_f1", "worst_domain_accuracy", "domain_gap", "cross_entropy")
    resources = ("raw_head_bytes", "estimated_total_bytes", "peak_process_rss_bytes", "process_wall_seconds",
                 "cell_wall_seconds", "local_train_seconds", "optimization_seconds", "aggregation_seconds", "evaluation_seconds")
    for steps in LOCAL_STEPS:
        rows = sorted((r for r in results if r["cell"]["local_steps"] == steps), key=lambda r: r["cell"]["seed"])
        group = {"seeds": [r["cell"]["seed"] for r in rows], "rounds": STEPS_PER_CLIENT // steps,
                 "resources": {name: stats([r[name] for r in rows]) for name in resources}, "trajectory": []}
        for index, completed in enumerate(CHECKPOINTS):
            points = [r["history"][index] for r in rows]
            group["trajectory"].append({"steps_per_client": completed,
                "metrics": {name: stats([r["metrics"][name] for r in points]) for name in quality_names},
                "per_domain_accuracy": {name: stats([r["metrics"]["per_domain_accuracy"][name] for r in points])
                                        for name in points[0]["metrics"]["per_domain_accuracy"]},
                "estimated_total_bytes": points[0]["estimated_total_bytes"]})
        groups[str(steps)] = group
    points = {key: {"accuracy": g["trajectory"][-1]["metrics"]["accuracy"]["mean"],
                    **{name: g["resources"][name]["mean"] for name in ("estimated_total_bytes", "process_wall_seconds", "peak_process_rss_bytes")}}
              for key, g in groups.items()}
    frontiers = {}
    for name, costs in (("accuracy_payload", ("estimated_total_bytes",)),
                        ("accuracy_payload_wall_rss", ("estimated_total_bytes", "process_wall_seconds", "peak_process_rss_bytes"))):
        frontiers[name] = {key: {"dominated_by": [other for other in points if other != key and dominates(points[other], point, ("accuracy",), costs)]}
                           for key, point in points.items()}
    indexed = {(r["cell"]["seed"], r["cell"]["local_steps"]): r for r in results}
    paired = {str(steps): {metric: stats([indexed[(s, steps)]["history"][-1]["metrics"][metric] - indexed[(s, 1)]["history"][-1]["metrics"][metric] for s in SEEDS])
                           for metric in quality_names} for steps in (2, 4)}
    return {"groups": groups, "pareto_group_means": frontiers, "paired_final_difference_from_local1": paired}


def report(output, public):
    plan = read(output / "plan.json")
    verify(plan, output, public)
    results = []
    previous_end = None
    for cell in plan["cells"]:
        run = output / "cells" / cell["id"]
        result = read(run / "result.json")
        validate_result(result, plan, cell)
        process = read(run / "process.json")
        if process["result_sha256"] != sha256(run / "result.json") or process["cell_id"] != cell["id"]:
            raise ValueError("Process measurement/result identity mismatch")
        if previous_end and process["started_utc"] < previous_end:
            raise ValueError("Timed subprocesses overlapped")
        previous_end = process["finished_utc"]
        results.append({**result, "process_wall_seconds": process["process_wall_seconds"],
                        "peak_process_rss_bytes": process["peak_process_rss_bytes"]})
    summary = {"status": "completed", "plan_hash": plan["plan_hash"], "completed_cells": len(results),
               "scope": SCOPE, "results": results, **summarize(results)}
    write_json(output / "summary.json", summary)
    write_json(public / "summary.json", summary)
    return summary


def run_all(output, public):
    plan = read(output / "plan.json")
    verify(plan, output, public)
    for cell in plan["cells"]:
        run = output / "cells" / cell["id"]
        if run.exists():
            # Never rerun a timed cell to obtain a preferred measurement.
            saved = read(run / "result.json")
            validate_result(saved, plan, cell)
            if not (run / "process.json").exists():
                raise RuntimeError("Existing cell has no complete process timing; investigate")
            continue
        tick = time.perf_counter()
        start = datetime.now(timezone.utc).isoformat()
        if sys.platform != "linux":
            raise RuntimeError("Measured cells require the pinned WSL Linux environment")
        command = [sys.executable, str(Path(__file__).resolve()), "cell", "--output", str(output),
                   "--public", str(public), "--cell-id", cell["id"]]
        child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **plan["thread_environment"]})
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, command)
        elapsed = time.perf_counter() - tick
        write_json(run / "process.json", {"cell_id": cell["id"], "started_utc": start,
                   "finished_utc": datetime.now(timezone.utc).isoformat(), "process_wall_seconds": elapsed,
                   "peak_process_rss_bytes": int(usage.ru_maxrss * 1024),
                   "result_sha256": sha256(run / "result.json")})
        print(f"COMPLETED {cell['id']} process_wall={elapsed:.2f}s", flush=True)
    return report(output, public)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "cell", "report"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--public", type=Path, default=PUBLIC)
    parser.add_argument("--cell-id")
    args = parser.parse_args()
    output, public = args.output.resolve(), args.public.resolve()
    if args.action == "prepare":
        print(json.dumps({"plan_hash": prepare(output, public)["plan_hash"]}))
    elif args.action == "cell":
        if not args.cell_id:
            parser.error("cell requires --cell-id")
        run_cell(output, public, args.cell_id)
    else:
        result = run_all(output, public) if args.action == "run" else report(output, public)
        print(json.dumps({"status": result["status"], "completed_cells": result["completed_cells"]}))


if __name__ == "__main__":
    main()
