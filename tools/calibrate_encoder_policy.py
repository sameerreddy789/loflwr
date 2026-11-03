"""Choose margin thresholds using validation only; no confidence guarantee."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedlora_edge.data import INTENTS
from fedlora_edge.encoder_data import load_encoder_manifest, load_feature_split
from fedlora_edge.encoder_inference import load_head_checkpoint
from fedlora_edge.encoder_head import HeadTrainer
from fedlora_edge.checkpoint import atomic_json


def calibrate(run_dir, dataset_dir, output):
    import numpy as np
    if Path(output).exists():
        raise FileExistsError("Choose a fresh calibration output.")
    saved = load_head_checkpoint(Path(run_dir) / "selected")
    manifest = load_encoder_manifest(dataset_dir)
    if saved["identity"]["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("Head and validation dataset identity differ.")
    x, y, _ = load_feature_split(dataset_dir)
    head = HeadTrainer(manifest["feature_dim"], len(INTENTS))
    head.set_state(saved["state"])
    scores = head.predict_scores(x)
    order = np.argsort(scores, axis=1)
    correct = order[:, -1] == y
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    policies = []
    for target in (.98, .99, 1.):
        options = []
        for threshold in sorted(set(margins.tolist())):
            accepted = margins >= threshold
            if correct[accepted].mean() >= target:
                options.append((int(accepted.sum()), -float(threshold), float(correct[accepted].mean())))
        if options:
            count, negative_threshold, accuracy = max(options)
            policies.append({"target_validation_accuracy": target, "threshold": -negative_threshold,
                "validation_answered": count, "validation_coverage": count / len(y),
                "validation_accuracy_on_answered": accuracy})
    result = {"kind": "validation_calibrated_margin_policies", "split": "reused_validation",
        "run_id": saved["run_id"], "head_hash": saved["b_hash"], "manifest_hash": manifest["manifest_hash"],
        "validation_records_hash": manifest["validation"]["records_hash"], "count": len(y),
        "top1_accuracy": float(correct.mean()), "policies": policies,
        "rule": "Answer iff top1 minus top2 raw logit >= threshold; otherwise abstain.",
        "selection": "For each fixed target 98/99/100%, maximize validation coverage; break ties by smaller threshold.",
        "interpretation": "Empirical thresholds from reused validation. No confidence or future-accuracy guarantee. Report coverage and all-query accuracy separately."}
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", default="assets/encoder-bank20-v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(calibrate(args.run_dir, args.dataset_dir, args.output), indent=2))
