"""Cheap task-efficiency context; this is not a federated-adapter experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedlora_edge.data import content_hash, load_manifest, verify_manifests, write_json  # noqa: E402
from fedlora_edge.training import classification_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("assets/banking77"))
    parser.add_argument("--scenario", choices=("balanced", "imbalanced"), default="imbalanced")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    verified = verify_manifests(args.data_dir)
    manifests = args.data_dir / "manifests"
    train = [record for client in range(5)
             for record in load_manifest(manifests / args.scenario / f"client_{client}.json")]
    held_out = load_manifest(manifests / f"{args.split}.json")
    config = {
        "run_id": args.run_id, "kind": "centralized_tfidf_classifier_reference",
        "scenario": args.scenario, "split": args.split,
        "manifest_set_hash": verified["manifest_set_hash"],
        "vectorizer": {"ngram_range": [1, 2], "sublinear_tf": True},
        "classifier": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced", "random_state": 42},
        "scikit_learn_version": importlib.metadata.version("scikit-learn"),
        "training_records_hash": content_hash(train),
        "evaluation_records_hash": content_hash(held_out),
    }
    config["config_hash"] = content_hash(config)
    # Hyperparameters are fixed in source, not selected on the held-out split.
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                             LogisticRegression(C=1.0, max_iter=1000,
                                                class_weight="balanced", random_state=42))
    start = time.perf_counter()
    pipeline.fit([row["query"] for row in train], [row["label"] for row in train])
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predicted = pipeline.predict([row["query"] for row in held_out]).tolist()
    predict_seconds = time.perf_counter() - start
    metrics = classification_metrics([row["label"] for row in held_out], predicted)
    result = {
        "config": config, "metrics": metrics, "training_examples": len(train),
        "fit_seconds": fit_seconds, "prediction_seconds": predict_seconds,
        "interpretation": "Centralized public-data task reference with inverse-class-frequency loss weights. No LoRA, federation, or dropout claim.",
        "predictions": [{"source_id": row["source_id"], "label": row["label"], "prediction": prediction}
                        for row, prediction in zip(held_out, predicted)],
    }
    write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
