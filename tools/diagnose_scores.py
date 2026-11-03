"""Validation-only likelihood diagnostics; never changes the deployed scorer.

Compare three predeclared scoring rules on a snapshot of one B checkpoint.
The output is exploratory validation evidence, not a final-test result or a
calibrated confidence estimate. No training or asset downloads occur here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedlora_edge.data import INTENTS, content_hash, load_manifest, verify_manifests, write_json  # noqa: E402
from fedlora_edge.state import hash_state  # noqa: E402
from fedlora_edge.training import LocalTrainer, TrainingConfig, classification_metrics, load_b_state  # noqa: E402

RULES = ("sum_including_eos", "mean_including_eos", "content_free_centered_sum")
CONTENT_FREE_QUERY = "N/A"


def scoring_variants(scores: Sequence[float], lengths: Sequence[int],
                     null_scores: Sequence[float]) -> dict[str, list[float]]:
    """Retain EOS for all rules; centering subtracts one content-free log score."""
    if not scores or len(scores) != len(lengths) or len(scores) != len(null_scores):
        raise ValueError("Scores, lengths and content-free scores must align.")
    if any(not math.isfinite(value) for value in [*scores, *null_scores]):
        raise ValueError("All scores must be finite.")
    if any(isinstance(length, bool) or not isinstance(length, int) or length < 1
           for length in lengths):
        raise ValueError("Completion lengths must be positive integers.")
    return {
        RULES[0]: list(scores),
        RULES[1]: [score / length for score, length in zip(scores, lengths)],
        RULES[2]: [score - null for score, null in zip(scores, null_scores)],
    }


def summarize(rows: Sequence[dict], lengths: Sequence[int], null_scores: Sequence[float]) -> dict:
    if not rows or len(lengths) != len(INTENTS):
        raise ValueError("A summary requires validation rows and twenty lengths.")
    predictions = {rule: [] for rule in RULES}
    truth = [row["label"] for row in rows]
    for row in rows:
        if len(row["scores"]) != len(INTENTS):
            raise ValueError("A row must contain all twenty candidate scores.")
        for rule, scores in scoring_variants(row["scores"], lengths, null_scores).items():
            predictions[rule].append(INTENTS[max(range(len(INTENTS)), key=scores.__getitem__)])
    base = predictions[RULES[0]]
    result = {}
    for rule, predicted in predictions.items():
        errors = Counter((actual, chosen) for actual, chosen in zip(truth, predicted)
                         if actual != chosen)
        result[rule] = {
            "metrics": classification_metrics(truth, predicted),
            "prediction_counts": dict(Counter(predicted)),
            "top_confusions": [{"actual": pair[0], "predicted": pair[1], "count": count}
                               for pair, count in errors.most_common(20)],
            "paired_against_sum": {
                "changed": sum(old != new for old, new in zip(base, predicted)),
                "corrected": sum(old != actual and new == actual
                                 for actual, old, new in zip(truth, base, predicted)),
                "harmed": sum(old == actual and new != actual
                              for actual, old, new in zip(truth, base, predicted)),
            },
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("assets/banking77"))
    parser.add_argument("--asset-dir", type=Path, default=Path("assets/model"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("assets/adapter"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Use a new output directory; diagnostic evidence is immutable.")
    verified = verify_manifests(args.data_dir)
    # There is deliberately no --split/test option or arbitrary input manifest.
    validation_path = args.data_dir / "manifests" / "validation.json"
    records = load_manifest(validation_path)
    if len(records) != 200 or Counter(row["label"] for row in records) != Counter({label: 10 for label in INTENTS}):
        raise ValueError("Expected the frozen, balanced 200-query validation manifest.")
    identity = json.loads((args.adapter_dir / "identity.json").read_text(encoding="utf-8"))
    config = TrainingConfig(cpu_threads=args.cpu_threads, eval_batch_size=args.eval_batch_size,
                            max_length=args.max_length)
    snapshot = args.checkpoint.read_bytes()
    checkpoint_meta = json.loads(snapshot)
    args.output_dir.mkdir(parents=True)
    snapshot_path = args.output_dir / "checkpoint.snapshot.json"
    snapshot_path.write_bytes(snapshot)
    b_state = load_b_state(snapshot_path, identity)
    plan = {
        "run_id": args.output_dir.name,
        "kind": "exploratory_validation_scoring_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "validation", "count": len(records), "rules": list(RULES),
        "content_free_query": CONTENT_FREE_QUERY,
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": hashlib.sha256(snapshot).hexdigest(),
        "source_run_id": checkpoint_meta.get("run_id"),
        "source_attempt_id": checkpoint_meta.get("attempt_id"),
        "source_version": checkpoint_meta.get("version"),
        "b_hash": hash_state(b_state), "contract_hash": identity["contract_hash"],
        "model_revision": identity["model_revision"],
        "manifest_set_hash": verified["manifest_set_hash"],
        "validation_records_hash": content_hash(records),
        "training_config": asdict(config), "device": args.device,
        "source_files_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__).resolve(), ROOT / "fedlora_edge/training.py", ROOT / "fedlora_edge/data.py"]
        },
        "interpretation": "All alternatives are exploratory. No test scores, training, active-scorer change, or calibrated confidence claim.",
    }
    plan["config_hash"] = content_hash(plan)
    write_json(args.output_dir / "plan.json", plan)
    started = time.perf_counter()
    trainer = LocalTrainer(args.asset_dir, args.adapter_dir, device=args.device, config=config)
    trainer.set_b_state(b_state)
    lengths = [len(trainer.encoder.completions[label]) for label in INTENTS]
    null_scores = [score for score, _ in trainer._scores_for_examples(
        [trainer.encoder.encode(CONTENT_FREE_QUERY, label) for label in INTENTS])]
    rows = []
    # Rows retain candidate scores and source IDs but never raw query text.
    with (args.output_dir / "validation_scores.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            encoded = [trainer.encoder.encode(record["query"], label) for label in INTENTS]
            values = trainer._scores_for_examples(encoded)
            row = {"source_id": record["source_id"], "label": record["label"],
                   "scores": [score for score, _ in values], "truncated_query": encoded[0]["truncated_query"]}
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if len(rows) % 20 == 0:
                write_json(args.output_dir / "progress.json", {"status": "running", "scored": len(rows), "total": len(records)})
                print(f"Scored {len(rows)}/{len(records)} validation queries", flush=True)
    result = {"plan": plan, "status": "completed", "completion_lengths": dict(zip(INTENTS, lengths)),
              "content_free_scores": dict(zip(INTENTS, null_scores)),
              "truncated_queries": sum(row["truncated_query"] for row in rows),
              "wall_seconds": time.perf_counter() - started, "variants": summarize(rows, lengths, null_scores)}
    write_json(args.output_dir / "summary.json", result)
    write_json(args.output_dir / "progress.json", {"status": "completed", "scored": len(rows), "total": len(records)})
    print(json.dumps({rule: value["metrics"]["accuracy"] for rule, value in result["variants"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
