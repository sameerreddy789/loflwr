"""Freeze a completed head run, then evaluate its sealed confirmation split.

The freeze command reads manifest metadata only. Evaluation verifies the frozen
checkpoint, source files and manifest before loading any confirmation queries.
Never use confirmation results to select or revise this experiment's model.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fedlora_edge.checkpoint import atomic_json
from fedlora_edge.data import INTENTS, content_hash, normalize_query
from fedlora_edge.encoder_data import load_encoder_manifest
from fedlora_edge.encoder_inference import EncoderPredictor, load_head_checkpoint, verify_inference_assets
from fedlora_edge.training import classification_metrics


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_lock(plan):
    if content_hash({k: v for k, v in plan.items() if k != "lock_hash"}) != plan["lock_hash"]:
        raise ValueError("Confirmation lock was modified.")
    for name, digest in plan["locked_files"].items():
        if file_hash(name) != digest:
            raise ValueError(f"Frozen file changed: {name}")


def selective_result(scores, truth, threshold):
    """Keep accuracy on answered queries separate from all-query coverage."""
    import numpy as np
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Margin threshold must be finite and nonnegative.")
    scores, truth = np.asarray(scores), np.asarray(truth)
    if (scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2
            or truth.shape != (len(scores),) or not np.isfinite(scores).all()):
        raise ValueError("Scores and truth must be finite aligned nonempty arrays.")
    order = np.sort(scores, axis=1)
    accepted = order[:, -1] - order[:, -2] >= threshold
    answered = int(accepted.sum())
    correct_answered = int(((scores.argmax(axis=1) == truth) & accepted).sum())
    return {"confirmation_answered": answered, "confirmation_coverage": answered / len(truth),
            "confirmation_correct_answered": correct_answered,
            "confirmation_accuracy_on_answered": correct_answered / answered if answered else None,
            "confirmation_abstained": len(truth) - answered}


def freeze(run_dir, dataset_dir, asset_dir, output, reason, policy_path=None):
    run_dir, dataset_dir, asset_dir, output = (Path(p).resolve() for p in (run_dir, dataset_dir, asset_dir, output))
    if output.exists():
        raise FileExistsError("Choose a fresh confirmation directory; locks are immutable.")
    manifest = load_encoder_manifest(dataset_dir)
    assets = verify_inference_assets(asset_dir)
    if assets["files_sha256"] != manifest["encoder"]["files_sha256"]:
        raise ValueError("Model assets differ from the training manifest.")
    checkpoint = run_dir / "selected/checkpoint.json"
    saved = load_head_checkpoint(checkpoint.parent)
    source = run_dir / "final_results.json"
    if not source.exists():
        source = run_dir / "summary.json"
    evidence = json.loads(source.read_text(encoding="utf-8"))
    selected = evidence["selected"]
    expected = selected.get("state_hash") or selected["metrics"].get("head_hash")
    if (selected["split"] != "validation" or saved["b_hash"] != expected
            or saved["version"] != selected["version"]
            or saved["identity"]["manifest_hash"] != manifest["manifest_hash"]):
        raise ValueError("Selected checkpoint and validation evidence do not agree.")
    sources = [ROOT / name for name in ("tools/confirm_encoder.py", "fedlora_edge/encoder_inference.py",
        "fedlora_edge/encoder_head.py", "fedlora_edge/encoder_adapter.py", "fedlora_edge/encoder_data.py",
        "fedlora_edge/training.py", "fedlora_edge/data.py", "fedlora_edge/checkpoint.py", "fedlora_edge/state.py")]
    plan = {"kind": "locked_mpnet_head_confirmation", "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir), "dataset_dir": str(dataset_dir), "asset_dir": str(asset_dir),
        "run_id": saved["run_id"], "head_hash": saved["b_hash"], "config_hash": saved["config_hash"],
        "manifest_hash": manifest["manifest_hash"], "confirmation": manifest["confirmation"],
        "selected_validation": selected, "selection_reason": reason,
        "scoring": "Pinned offline MPNet, masked mean + L2, fixed 20-way head argmax, length 384, FP32 CPU",
        "locked_files": {str(p): file_hash(p) for p in [checkpoint, source, dataset_dir / "manifest.json",
            *sources, *(asset_dir / name for name in assets["files_sha256"])]},
        "interpretation": "Single selected model. No confirmation-based tuning, relabeling or exclusions."}
    if policy_path is not None:
        policy_path = Path(policy_path).resolve()
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if (policy.get("kind") != "validation_calibrated_margin_policies"
                or policy.get("head_hash") != saved["b_hash"]
                or policy.get("manifest_hash") != manifest["manifest_hash"]
                or policy.get("validation_records_hash") != manifest["validation"]["records_hash"]):
            raise ValueError("Margin calibration belongs to a different model or dataset.")
        for point in policy["policies"]:
            value = point["threshold"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Calibration contains an invalid threshold.")
        plan["selective_policies"] = policy
        plan["locked_files"][str(policy_path)] = file_hash(policy_path)
        calibration_source = ROOT / "tools/calibrate_encoder_policy.py"
        plan["locked_files"][str(calibration_source)] = file_hash(calibration_source)
    plan["lock_hash"] = content_hash(plan)
    atomic_json(output / "plan.json", plan)
    return plan


def evaluate(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    verify_lock(plan)
    result_file = output / "result.json"
    if result_file.exists():
        result = json.loads(result_file.read_text(encoding="utf-8"))
        if result["lock_hash"] != plan["lock_hash"]:
            raise ValueError("Stored confirmation result belongs to a different lock.")
        return result
    started = time.perf_counter()
    predictor = EncoderPredictor(plan["asset_dir"], Path(plan["run_dir"]) / "selected")
    verify_lock(plan)
    atomic_json(output / "evaluation_started.json", {"lock_hash": plan["lock_hash"], "started_unix": time.time()})
    entry = plan["confirmation"]
    root = Path(plan["dataset_dir"])
    path = (root / entry["records_file"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Confirmation path escapes its dataset directory.")
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"]
    if (len(records) != entry["count"] or len(records) != 300
            or content_hash(records) != entry["records_hash"]
            or data["records_hash"] != entry["records_hash"]
            or len({r["source_id"] for r in records}) != len(records)
            or len({normalize_query(r["query"]) for r in records}) != len(records)
            or any(r["label"] not in INTENTS or r["source_split"] != "test" for r in records)):
        raise ValueError("Confirmation records do not match the frozen manifest.")
    scores, truncated = predictor.scores([r["query"] for r in records])
    predicted = [INTENTS[int(i)] for i in scores.argmax(1)]
    metrics = classification_metrics([r["label"] for r in records], predicted)
    correct = sum(r["label"] == p for r, p in zip(records, predicted))
    n, z = len(records), 1.959963984540054
    rate = correct / n
    centre = (rate + z*z/(2*n)) / (1 + z*z/n)
    radius = z*((rate*(1-rate)/n + z*z/(4*n*n)) ** .5) / (1+z*z/n)
    verify_lock(plan)
    result = {"status": "completed", "split": "previously_unscored_confirmation", "run_id": plan["run_id"],
        "lock_hash": plan["lock_hash"], "head_hash": plan["head_hash"], "config_hash": plan["config_hash"],
        "records_hash": entry["records_hash"], "metrics": metrics, "correct": correct, "errors": n-correct,
        "nominal_wilson_95": [centre-radius, centre+radius], "truncated": truncated,
        "wall_seconds_including_model_loading": time.perf_counter()-started,
        "predictions": [{"source_id": r["source_id"], "label": r["label"], "prediction": p}
                        for r, p in zip(records, predicted)],
        "interpretation": "Report all 300. Do not tune against this result; a finite sample is not a population guarantee."}
    import numpy as np
    truth = np.array([INTENTS.index(r["label"]) for r in records])
    shifted = scores.astype(np.float64) - scores.max(axis=1, keepdims=True)
    result["metrics"]["cross_entropy"] = float((-shifted[np.arange(n), truth] + np.log(np.exp(shifted).sum(axis=1))).mean())
    order = scores.argsort(axis=1)
    result["top2_candidate_recall"] = float((order[:, -2:] == truth[:, None]).any(axis=1).mean())
    result["selective_results"] = []
    for policy in plan.get("selective_policies", {}).get("policies", []):
        result["selective_results"].append({**policy, **selective_result(scores, truth, policy["threshold"])})
    atomic_json(result_file, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "evaluate"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dataset-dir", default="assets/encoder-bank20-v1")
    parser.add_argument("--asset-dir", default="assets/reference-mpnet")
    parser.add_argument("--reason")
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        if not args.run_dir or not args.reason:
            parser.error("Freeze requires --run-dir and --reason.")
        result = freeze(args.run_dir, args.dataset_dir, args.asset_dir, args.output, args.reason, args.policy)
        print(json.dumps({k: result[k] for k in ("lock_hash", "run_id", "head_hash")}))
    else:
        result = evaluate(args.output)
        print(json.dumps({k: result[k] for k in ("run_id", "metrics", "correct", "errors", "nominal_wilson_95")}))


if __name__ == "__main__":
    main()
