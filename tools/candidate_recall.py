"""Post-hoc validation candidate recall from the cached centralized MPNet reference.

Ranks distinct labels by their nearest training example. Recall@k for k > 1 is
not classification accuracy and does not implement a reranker or clarification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedlora_edge.data import INTENTS, content_hash, load_manifest, verify_manifests, write_json  # noqa: E402


def candidate_recall(similarities, training_labels, truth, label_order, ks=(1, 2, 3)):
    """Maximum similarity per label; ties use the declared label order."""
    labels = list(label_order)
    if not truth or len(similarities) != len(truth) or len(set(labels)) != len(labels):
        raise ValueError("Expected aligned nonempty queries and distinct candidate labels.")
    if set(training_labels) != set(labels) or not set(truth).issubset(labels):
        raise ValueError("Training must cover exactly the allowed labels; truth must be supported.")
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(labels) for k in ks):
        raise ValueError("Candidate counts must be positive integers within the label set.")
    correct = {str(k): 0 for k in ks}
    for row, actual in zip(similarities, truth):
        if len(row) != len(training_labels) or any(not math.isfinite(value) for value in row):
            raise ValueError("Similarities must be finite and align with training examples.")
        scores = dict.fromkeys(labels, -math.inf)
        for label, value in zip(training_labels, row):
            scores[label] = max(scores[label], value)
        ranked = sorted(labels, key=scores.__getitem__, reverse=True)
        for k in ks:
            correct[str(k)] += actual in ranked[:k]
    return {"count": len(truth), "correct_within_top_k": correct,
            "recall": {k: value / len(truth) for k, value in correct.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("assets/banking77"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a new output path; previous evidence must be preserved.")
    import numpy as np

    read = lambda name: json.loads((args.reference_dir / name).read_text(encoding="utf-8"))
    summary = read("summary.json")
    plan = summary["plan"]
    if summary.get("status") != "completed" or plan.get("split") != "validation":
        raise ValueError("Requires the completed validation-only reference.")
    if content_hash({key: value for key, value in plan.items() if key != "config_hash"}) != plan["config_hash"]:
        raise ValueError("Reference configuration hash mismatch.")
    verified = verify_manifests(args.data_dir)
    validation_path = args.data_dir / "manifests/validation.json"
    validation = load_manifest(validation_path)
    if (verified["manifest_set_hash"] != plan["manifest_set_hash"]
            or content_hash(validation) != plan["validation_records_hash"]):
        raise ValueError("The frozen validation/manifests differ from the reference.")
    training = read("expanded_training_manifest.json")["records"]
    if content_hash(training) != plan["scenarios"]["all_eligible_training"]["records_hash"]:
        raise ValueError("Expanded training identity mismatch.")
    ids = read("embedding_ids.json")["ids"]
    vectors = np.load(args.reference_dir / "reference_embeddings.npy", allow_pickle=False)
    if (vectors.ndim != 2 or vectors.shape[0] != len(ids) or len(set(ids)) != len(ids)
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)):
        raise ValueError("Expected one finite normalized embedding per distinct source ID.")
    index = {key: i for i, key in enumerate(ids)}
    train_vectors = vectors[[index[row["source_id"]] for row in training]]
    val_vectors = vectors[[index[row["source_id"]] for row in validation]]
    metrics = candidate_recall((val_vectors @ train_vectors.T).tolist(),
                               [row["label"] for row in training],
                               [row["label"] for row in validation], INTENTS)
    source_paths = [args.reference_dir / name for name in (
        "summary.json", "expanded_training_manifest.json", "embedding_ids.json", "reference_embeddings.npy")]
    source_paths += [validation_path, Path(__file__).resolve()]
    result = {"run_id": args.output.stem, "created_utc": datetime.now(timezone.utc).isoformat(),
              "reference_run": plan["run_id"], "reference_config_hash": plan["config_hash"],
              "kind": "posthoc_validation_distinct_label_candidate_recall", "split": "validation",
              "scenario": "all_eligible_training", "candidate_label_count": len(INTENTS),
              "training_count": len(training), "validation_records_hash": content_hash(validation),
              "source_files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
              "rule": "Maximum cosine similarity to any training example for each of twenty distinct labels; ties use INTENTS order.",
              "interpretation": "Recall@2/3 is not top-1 accuracy. A restricted reranker cannot exceed candidate recall on these queries, and no reranker is implemented. Validation-only post-hoc diagnostic.",
              **metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
