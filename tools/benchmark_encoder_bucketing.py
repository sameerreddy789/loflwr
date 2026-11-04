"""Locked development comparison of original and bucketed local CPU encoding.

Two fixed cohorts: all 200 reused validation rows and all 408 weak-client rows.
Three paired repetitions in fresh processes; no held-out test inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import evaluate_fresh_bank20 as base

DATASET = "assets/encoder-bank20-v1"
KIND = "encoder_bucketing_benchmark_v1"
THREAD_ENV = {key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
SOURCES = ("tools/benchmark_encoder_bucketing.py", "tests/test_encoder_bucketing_benchmark.py",
           "fedlora_edge/encoder_bucketing.py", "tools/prepare_encoder_bucketed.py",
           "tests/test_encoder_bucketing.py", "fedlora_edge/encoder_local_data.py")
DESIGN = {"cohorts": {"server": 200, "4": 408}, "repeats": 3, "batch_size": 8,
          "max_length": 384, "cpu_threads": 1, "dtype": "float32", "feature_absolute_tolerance": 1e-5,
          "timing": "Resident encoder; includes tokenization, sorting, padding, pooling, restoration and forward-shape hooks; excludes model load and quality checks",
          "warmup": "First eight rows through each backend before timing; no scores selected",
          "scope": "Single-laptop development parity/timing; reused validation and training only; no new accuracy, energy, wire traffic or physical-laptop claim"}


def cells():
    return [{"id": f"{role}-r{repeat}", "role": role, "repeat": repeat,
             "order": ["original", "bucketed"] if (position + repeat) % 2 == 0 else ["bucketed", "original"]}
            for repeat in range(3) for position, role in enumerate(("server", "4"))]


def runtime():
    cpu = platform.processor()
    if sys.platform == "linux":
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return {**base.runtime(), "os": platform.platform(), "machine": platform.machine(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV}}


def input_files(model):
    from fedlora_edge.encoder_data import load_encoder_manifest
    manifest = load_encoder_manifest(ROOT / DATASET)
    files = {**model["files"], **{name: base.sha(ROOT / name) for name in SOURCES},
             f"{DATASET}/manifest.json": base.sha(ROOT / DATASET / "manifest.json")}
    for role, count in DESIGN["cohorts"].items():
        entry = manifest["validation"] if role == "server" else manifest["clients"][role]
        if entry["count"] != count:
            raise ValueError("Fixed full-cohort row count changed")
        for field in ("records_file", "file"):
            name = f"{DATASET}/{entry[field]}"
            files[name] = base.sha(ROOT / name)
    return files


def prepare(output, public):
    output, public = base.private(output), Path(public).resolve()
    if not public.is_relative_to(ROOT / "research/results") or output.exists() or public.exists():
        raise ValueError("Use new private runs/assets and public research/results directories")
    model = base.model_contract()
    files = input_files(model)
    plan = {"kind": KIND, "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": DATASET,
            "output": base.relative(output), "public": base.relative(public), "model": model,
            "design": DESIGN, "cells": cells(), "files": files, "runtime": runtime()}
    plan["lock_hash"] = base.digest(plan)
    output.mkdir(parents=True)
    public.mkdir(parents=True)
    base.write_new(output / "plan.json", plan)
    base.write_new(public / "plan.json", plan)
    return plan


def verify(output):
    output = base.private(output)
    plan = base.read(output / "plan.json")
    if (plan.get("kind") != KIND or plan.get("dataset") != DATASET or plan.get("design") != DESIGN
            or plan.get("cells") != cells() or plan.get("output") != base.relative(output)
            or base.digest({k: v for k, v in plan.items() if k != "lock_hash"}) != plan.get("lock_hash")
            or runtime() != plan.get("runtime")):
        raise ValueError("Fixed benchmark design or environment changed")
    public = (ROOT / plan["public"]).resolve()
    if not public.is_relative_to(ROOT / "research/results") or base.read(public / "plan.json") != plan:
        raise ValueError("Public benchmark lock differs")
    if plan["model"] != base.model_contract() or plan["files"] != input_files(plan["model"]):
        raise ValueError("Required model/source/data contract changed")
    if any(base.relative(ROOT / name) != name or base.sha(ROOT / name) != value for name, value in plan["files"].items()):
        raise ValueError("A frozen benchmark input changed")
    return plan


def worker(output, cell_id):
    import numpy as np
    from fedlora_edge.encoder_bucketing import encode_bucketed
    from fedlora_edge.encoder_local_data import encode_batches
    from fedlora_edge.encoder_data import load_encoder_records
    from fedlora_edge.encoder_inference import EncoderPredictor
    from fedlora_edge.encoder_head import _RssMonitor
    plan = verify(output)
    cell = next(c for c in plan["cells"] if c["id"] == cell_id)
    target = Path(output) / f"{cell_id}.json"
    if target.exists():
        raise FileExistsError("Never replace a benchmark cell")
    records = load_encoder_records(ROOT / DATASET, None if cell["role"] == "server" else cell["role"])
    if any(r["source_split"] != "train" for r in records) or len(records) != DESIGN["cohorts"][cell["role"]]:
        raise ValueError("Benchmark requires the declared original-training cohorts")
    queries = [r["query"] for r in records]
    model = plan["model"]
    start = time.perf_counter()
    predictor = EncoderPredictor(ROOT / model["asset_dir"], ROOT / model["checkpoint_dir"], cpu_threads=1)
    loading = time.perf_counter() - start
    encode_batches(predictor.encoder, predictor.tokenizer, queries[:8], 8)
    encode_bucketed(predictor.encoder, predictor.tokenizer, queries[:8], 8)
    features, truncations, observations = {}, {}, {}
    work = None
    for backend in cell["order"]:
        shapes = []
        def hook(_module, _args, kwargs):
            shapes.append(list(kwargs["input_ids"].shape))
        handle = predictor.encoder.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with _RssMonitor() as memory:
                start = time.perf_counter()
                if backend == "original":
                    X, truncated = encode_batches(predictor.encoder, predictor.tokenizer, queries, 8)
                else:
                    X, truncated, work = encode_bucketed(predictor.encoder, predictor.tokenizer, queries, 8)
                seconds = time.perf_counter() - start
            features[backend], truncations[backend] = X, int(truncated)
            observations[backend] = {"seconds": seconds, "queries_per_second": len(queries) / seconds,
                "peak_process_rss_bytes": memory.peak, "rss_sample_interval_seconds": memory.interval,
                "padded_token_slots": sum(n*l for n, l in shapes),
                "attention_cells": sum(n*l*l for n, l in shapes), "forward_calls": len(shapes)}
        finally:
            handle.remove()
    old_scores = predictor.head.predict_scores(features["original"])
    new_scores = predictor.head.predict_scores(features["bucketed"])
    fixture_rows = [{"query_id": r["source_id"], "disposition": "supported"} for r in records]
    old = base.predictions_from_scores(fixture_rows, old_scores.tolist())
    new = base.predictions_from_scores(fixture_rows, new_scores.tolist())
    difference = float(np.abs(features["original"] - features["bucketed"]).max())
    changed = sum(a["prediction"] != b["prediction"] for a, b in zip(old, new))
    changed_answers = sum(a["answered"] != b["answered"] for a, b in zip(old, new))
    arithmetic = all(observations[backend][key] == work[f"{prefix}_{key}"]
        for backend, prefix in (("original", "baseline"), ("bucketed", "candidate"))
        for key in ("padded_token_slots", "attention_cells"))
    result = {"cell": cell, "lock_hash": plan["lock_hash"], "count": len(records),
        "source_ids_hash": base.digest([r["source_id"] for r in records]),
        "head_hash": predictor.saved["b_hash"], "effective_torch_threads": predictor.torch.get_num_threads(),
        "loading_seconds_including_asset_verification": loading, "observations": observations, "work": work,
        "parity": {"max_feature_absolute_difference": difference, "max_logit_absolute_difference": float(np.abs(old_scores-new_scores).max()),
            "prediction_changes": changed, "strict_margin_answer_changes": changed_answers,
            "truncations": truncations, "observed_work_matches_arithmetic": arithmetic,
            "passed": difference <= DESIGN["feature_absolute_tolerance"] and changed == changed_answers == 0
                and truncations["original"] == truncations["bucketed"] and arithmetic},
        "memory_limit": "Sampled current-process RSS includes encoder/runtime and retained outputs; warm allocator state makes pairwise RSS comparisons order-dependent"}
    verify(output)
    base.write_new(target, result)
    print(json.dumps({"cell": cell_id, "seconds": {k: v["seconds"] for k,v in observations.items()}, "parity": result["parity"]}), flush=True)
    return result


def validate_parity(row):
    count, parity, work = row["count"], row["parity"], row["work"]
    if work.get("query_count") != count or work.get("batch_size") != 8:
        raise ValueError("Work telemetry differs from the declared cohort and batch")
    for key in ("max_feature_absolute_difference", "max_logit_absolute_difference"):
        value = parity[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid finite parity difference")
    for value in (parity["prediction_changes"], parity["strict_margin_answer_changes"], *parity["truncations"].values()):
        if type(value) is not int or not 0 <= value <= count:
            raise ValueError("Invalid parity change/truncation count")
    if set(parity["truncations"]) != {"original", "bucketed"}:
        raise ValueError("Missing truncation arm")
    for backend, prefix in (("original", "baseline"), ("bucketed", "candidate")):
        for key in ("padded_token_slots", "attention_cells"):
            observed, expected = row["observations"][backend][key], work[f"{prefix}_{key}"]
            if type(observed) is not int or type(expected) is not int or min(observed, expected) < count:
                raise ValueError("Invalid padding work counters")
    arithmetic = all(row["observations"][backend][key] == work[f"{prefix}_{key}"]
        for backend, prefix in (("original", "baseline"), ("bucketed", "candidate"))
        for key in ("padded_token_slots", "attention_cells"))
    passed = (parity["max_feature_absolute_difference"] <= DESIGN["feature_absolute_tolerance"]
        and parity["prediction_changes"] == parity["strict_margin_answer_changes"] == 0
        and parity["truncations"]["original"] == parity["truncations"]["bucketed"] and arithmetic)
    if (type(parity["passed"]) is not bool or parity["passed"] != passed
            or type(parity["observed_work_matches_arithmetic"]) is not bool
            or parity["observed_work_matches_arithmetic"] != arithmetic):
        raise ValueError("Reported parity flag contradicts primitive measurements")
    return passed


def summarize(plan, rows):
    if len(rows) != len(plan["cells"]) or {r["cell"]["id"] for r in rows} != {c["id"] for c in plan["cells"]}:
        raise ValueError("Every declared paired cell is required, including unfavorable outcomes")
    for row in rows:
        if (row["lock_hash"] != plan["lock_hash"] or row["cell"] not in plan["cells"]
                or row["count"] != DESIGN["cohorts"][row["cell"]["role"]]
                or row["head_hash"] != plan["model"]["head_hash"] or row["effective_torch_threads"] != 1):
            raise ValueError("Mixed benchmark identities/counts")
        for backend in ("original", "bucketed"):
            duration = row["observations"][backend]["seconds"]
            if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
                raise ValueError("Timing must be finite and positive")
        validate_parity(row)
    cohorts = {}
    for role in DESIGN["cohorts"]:
        group = [r for r in rows if r["cell"]["role"] == role]
        if (len({r["source_ids_hash"] for r in group}) != 1
                or any(r["work"] != group[0]["work"] for r in group)):
            raise ValueError("Cohort ordering or padding work changed between repetitions")
        times = {b: [r["observations"][b]["seconds"] for r in group] for b in ("original", "bucketed")}
        medians = {b: statistics.median(values) for b, values in times.items()}
        cohorts[role] = {"count": DESIGN["cohorts"][role], "seconds": times, "median_seconds": medians,
            "median_time_reduction_fraction": 1 - medians["bucketed"] / medians["original"],
            "ratio_of_median_times": medians["original"] / medians["bucketed"],
            "paired_speedups": [r["observations"]["original"]["seconds"] / r["observations"]["bucketed"]["seconds"] for r in group],
            "work": group[0]["work"], "all_parity_checks_passed": all(r["parity"]["passed"] for r in group)}
    return {"kind": KIND, "lock_hash": plan["lock_hash"], "status": "completed", "cohorts": cohorts,
            "cells": rows, "scope": DESIGN["scope"],
            "limits": ["Three timing repetitions are not three training seeds or new accuracy tests.",
                "One laptop with warm operating-system caches; recorded dashboard/background OS processes may remain.",
                "This optimizes batched preparation, not single-query latency, network bytes, or training steps.",
                "Attention-cell and token-slot counts are work proxies, not measured energy or FLOPs."]}


def run(output):
    plan = verify(output)
    base.write_new(Path(output) / "started.json", {"lock_hash": plan["lock_hash"], "utc": datetime.now(timezone.utc).isoformat()})
    for cell in plan["cells"]:
        log = Path(output) / f"{cell['id']}.log"
        with log.open("xb") as handle:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "_worker", "--output", str(output),
                "--cell", cell["id"]], stdout=handle, stderr=subprocess.STDOUT,
                env={**os.environ, **THREAD_ENV, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}, timeout=600)
        if result.returncode:
            raise RuntimeError(f"Cell {cell['id']} failed; preserve its log and plan")
        print(log.read_text(encoding="utf-8")[-1400:], flush=True)
    return report(output)


def report(output):
    plan = verify(output)
    rows = [base.read(Path(output) / f"{c['id']}.json") for c in plan["cells"]]
    result = summarize(plan, rows)
    result["result_hash"] = base.digest(result)
    path = ROOT / plan["public"] / "summary.json"
    if path.exists():
        if base.read(path) != result:
            raise ValueError("Never replace different measured outcomes")
    else:
        base.write_new(path, result)
    return result


def main():
    os.environ.update(THREAD_ENV)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report", "_worker"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--public", type=Path)
    parser.add_argument("--cell")
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.public:
            parser.error("prepare requires --public")
        result = prepare(args.output, args.public)
    elif args.action == "_worker":
        result = worker(args.output, args.cell)
    else:
        result = run(args.output) if args.action == "run" else report(args.output)
    print(json.dumps({k: result[k] for k in ("kind", "lock_hash", "status") if k in result}))


if __name__ == "__main__":
    main()
