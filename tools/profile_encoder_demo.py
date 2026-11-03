"""Measure offline CPU inference costs on authored timing probes, not accuracy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fedlora_edge.data import content_hash, write_json
from fedlora_edge.encoder_data import sha256

# Authored performance inputs, deliberately unrelated to evaluation selection.
# No expected labels: this command cannot calculate task accuracy.
QUERIES = (
    "My card payment is still pending.",
    "How can I verify my identity?",
    "I sent a transfer yesterday and the recipient has not received it yet.",
    "I requested a new card last week. Please explain how to check when it will arrive.",
    "The cash machine gave me a different amount from the amount I requested. What should I do next?",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, default=ROOT / "assets/reference-mpnet")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/encoder-flower-balanced-v2/selected")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh profile output")
    if args.repeats < 2 or args.cpu_threads < 1:
        raise ValueError("Use at least two repeats and one CPU thread")

    import numpy as np
    import torch
    import transformers
    from fedlora_edge.encoder_inference import EncoderPredictor
    from fedlora_edge.training import _MemoryMonitor

    sources = {name: sha256(ROOT / name) for name in (
        "tools/profile_encoder_demo.py", "fedlora_edge/encoder_inference.py",
        "fedlora_edge/encoder_head.py", "fedlora_edge/encoder_adapter.py")}
    with _MemoryMonitor(torch.device("cpu")) as memory:
        started = time.perf_counter()
        predictor = EncoderPredictor(args.asset_dir, args.checkpoint, cpu_threads=args.cpu_threads)
        loading = time.perf_counter() - started
        first = predictor.query(QUERIES[0])
        observations = []
        for repeat in range(args.repeats):
            for index, text in enumerate(QUERIES):
                result = predictor.query(text)
                observations.append({"repeat": repeat, "probe": index,
                                     "seconds": result["seconds"], "truncated": result["truncated"]})
    durations = [row["seconds"] for row in observations]
    result = {
        "status": "completed", "run_id": args.output.stem,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {"device": "cpu", "dtype": "float32", "cpu_threads": args.cpu_threads,
                          "repeats": args.repeats, "query_count": len(QUERIES), "batch_size": 1},
        "probe_hash": content_hash(QUERIES), "source_hashes": sources,
        "head_hash": predictor.saved["b_hash"], "source_run_id": predictor.saved["run_id"],
        "model_loading_seconds_including_asset_verification": loading,
        "first_query_seconds": first["seconds"], "measured_queries": len(durations),
        "warm_seconds": {"median": statistics.median(durations),
                         "p95_linear_interpolation": float(np.percentile(durations, 95)),
                         "min": min(durations), "max": max(durations)},
        "memory": memory.metrics(), "observations": observations,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "transformers": transformers.__version__},
        "limits": ["Authored timing probes; no accuracy or coverage measured.",
                   "Fresh Python process, warm OS file cache possible; imports precede load timer.",
                   "Single laptop with concurrent experiment work; no isolated hardware comparison.",
                   "Process RSS includes encoder, head and runtime, but excludes browser and other processes."]}
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("status", "measured_queries", "warm_seconds", "memory")}))


if __name__ == "__main__":
    main()
