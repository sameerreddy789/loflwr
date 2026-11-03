"""Validation-only task references for assessing the 99% aspiration.

These are centralized public-data classifiers, not federated LoRA experiments
or estimates of an accuracy ceiling. No final-test predictions are produced.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedlora_edge.data import (  # noqa: E402
    INTENTS, SOURCE_SHA256, _unique, content_hash, load_manifest, normalize_query,
    read_source, verify_manifests, write_json,
)
from fedlora_edge.training import classification_metrics  # noqa: E402

MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
MODEL_WEIGHTS_SHA256 = "78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658"
METHODS = ["word_tfidf_logreg", "word_char_tfidf_svm", "mpnet_logreg", "mpnet_nearest_example"]


def expanded_training(train, official_test, validation):
    """Use only original-train rows, excluding all official-test/validation matches."""
    if any(row["source_split"] != "train" for row in train):
        raise ValueError("Expanded training can only contain original training rows.")
    unique, duplicates = _unique(train)
    # Comparing text for leakage exclusion does not score the official test.
    _unique([*train, *official_test])
    forbidden = {normalize_query(row["query"]) for row in [*official_test, *validation]}
    selected = [row for row in unique if row["label"] in INTENTS
                and normalize_query(row["query"]) not in forbidden]
    if not selected or set(row["label"] for row in selected) != set(INTENTS):
        raise ValueError("Expanded data does not cover all twenty labels.")
    return selected, duplicates


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if not 0 <= correct <= total or total < 1:
        raise ValueError("Invalid binomial counts.")
    proportion = correct / total
    scale = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / scale
    half = z * ((proportion * (1 - proportion) / total + z * z / (4 * total * total)) ** 0.5) / scale
    return [max(0, center - half), min(1, center + half)]


def encode(texts, model_dir: Path, threads: int):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).eval()
    result = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(texts), 8):
            batch = tokenizer(texts[start:start + 8], padding=True, truncation=True,
                              max_length=384, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors = torch.nn.functional.normalize(pooled, p=2, dim=1)
            result.append(vectors.numpy())
            if start % 256 == 0:
                print(f"Encoded {min(start + 8, len(texts))}/{len(texts)} reference queries (CPU)", flush=True)
    return np.concatenate(result), time.perf_counter() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("assets/banking77"))
    parser.add_argument("--model-dir", type=Path, default=Path("assets/reference-mpnet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Choose a new output directory to preserve prior evidence.")
    if args.cpu_threads < 1:
        parser.error("CPU threads must be positive.")
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, make_pipeline
    from sklearn.svm import LinearSVC
    from threadpoolctl import threadpool_limits

    verified = verify_manifests(args.data_dir)
    for name in ("train.csv", "test.csv"):
        if hashlib.sha256((args.data_dir / "source" / name).read_bytes()).hexdigest() != SOURCE_SHA256[name]:
            raise ValueError(f"Original source changed: {name}")
    validation = load_manifest(args.data_dir / "manifests/validation.json")
    scenarios = {scenario: list(itertools.chain.from_iterable(
        load_manifest(args.data_dir / "manifests" / scenario / f"client_{client}.json")
        for client in range(5))) for scenario in ("imbalanced", "balanced")}
    expanded, duplicates = expanded_training(
        read_source(args.data_dir / "source/train.csv", "train"),
        read_source(args.data_dir / "source/test.csv", "test"), validation)
    scenarios["all_eligible_training"] = expanded
    # One fixed classifier setting per method, declared before validation scoring.
    model_files = {str(path.relative_to(args.model_dir)).replace("\\", "/"):
                   hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(args.model_dir.rglob("*"))
                   if path.is_file() and ".cache" not in path.parts}
    if "model.safetensors" not in model_files:
        raise ValueError("Cache the pinned MPNet safetensors before this offline run.")
    if model_files["model.safetensors"] != MODEL_WEIGHTS_SHA256:
        raise ValueError("MPNet weights do not match the Hub LFS hash at the pinned revision.")
    for relative in model_files:
        if relative == "model.safetensors":
            continue
        metadata = args.model_dir / ".cache/huggingface/download" / (relative + ".metadata")
        if not metadata.is_file() or metadata.read_text(encoding="utf-8").splitlines()[0] != MODEL_REVISION:
            raise ValueError(f"Unverified reference asset revision: {relative}")
    plan = {"run_id": args.output_dir.name, "kind": "centralized_accuracy_feasibility_references",
            "created_utc": datetime.now(timezone.utc).isoformat(), "split": "validation",
            "methods": METHODS, "classifier_settings": {"C": 1.0, "class_weight": "balanced", "random_state": 42, "max_iter": 5000},
            "word_features": {"ngram_range": [1, 2], "sublinear_tf": True},
            "extra_char_features": {"analyzer": "char_wb", "ngram_range": [3, 5], "sublinear_tf": True},
            "encoder": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "license": "apache-2.0",
                        "files_sha256": model_files, "pooling": "attention_mask_mean_then_l2_normalize",
                        "max_length": 384, "batch_size": 8, "device": "cpu", "threads": args.cpu_threads},
            "manifest_set_hash": verified["manifest_set_hash"], "validation_records_hash": content_hash(validation),
            "scenarios": {name: {"count": len(rows), "counts_per_intent": dict(Counter(row["label"] for row in rows)),
                                 "records_hash": content_hash(rows)} for name, rows in scenarios.items()},
            "original_training_duplicate_count": duplicates,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "scikit-learn", "numpy")},
            "interpretation": "Validation-only centralized references. Not an accuracy ceiling, federated result, or final-test evaluation."}
    plan["config_hash"] = content_hash(plan)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "plan.json", plan)
    write_json(args.output_dir / "expanded_training_manifest.json", {"records": expanded, "records_hash": content_hash(expanded)})
    # Embed each unique source row once. Training/validation embeddings use no labels.
    all_rows = {row["source_id"]: row for rows in [*scenarios.values(), validation] for row in rows}
    ids = list(all_rows)
    embeddings, encoding_seconds = encode([all_rows[key]["query"] for key in ids], args.model_dir, args.cpu_threads)
    index = {key: i for i, key in enumerate(ids)}
    np.save(args.output_dir / "reference_embeddings.npy", embeddings)
    write_json(args.output_dir / "embedding_ids.json", {"ids": ids, "model_files_sha256": model_files})
    val_vectors = embeddings[[index[row["source_id"]] for row in validation]]
    truth = [row["label"] for row in validation]
    output = {"plan": plan, "encoding_seconds": encoding_seconds, "results": [], "status": "running"}
    for scenario, rows in scenarios.items():
        queries = [row["query"] for row in rows]
        labels = [row["label"] for row in rows]
        vectors = embeddings[[index[row["source_id"]] for row in rows]]
        for method in METHODS:
            started = time.perf_counter()
            with threadpool_limits(limits=args.cpu_threads):
                if method == "word_tfidf_logreg":
                    classifier = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                               LogisticRegression(C=1, class_weight="balanced", random_state=42, max_iter=5000))
                    classifier.fit(queries, labels)
                    predictions = classifier.predict([row["query"] for row in validation]).tolist()
                elif method == "word_char_tfidf_svm":
                    features = FeatureUnion([("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                                             ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))])
                    classifier = make_pipeline(features, LinearSVC(C=1, class_weight="balanced", random_state=42, max_iter=5000))
                    classifier.fit(queries, labels)
                    predictions = classifier.predict([row["query"] for row in validation]).tolist()
                elif method == "mpnet_logreg":
                    classifier = LogisticRegression(C=1, class_weight="balanced", random_state=42, max_iter=5000)
                    classifier.fit(vectors, labels)
                    predictions = classifier.predict(val_vectors).tolist()
                else:
                    neighbors = (val_vectors @ vectors.T).argmax(axis=1)
                    predictions = [labels[int(i)] for i in neighbors]
            metrics = classification_metrics(truth, predictions)
            correct = sum(actual == predicted for actual, predicted in zip(truth, predictions))
            errors = Counter((actual, predicted) for actual, predicted in zip(truth, predictions) if actual != predicted)
            result = {"scenario": scenario, "method": method, "metrics": metrics, "correct": correct,
                      "errors": len(truth) - correct, "nominal_wilson_95": wilson_interval(correct, len(truth)),
                      "fit_predict_seconds": time.perf_counter() - started,
                      "top_confusions": [{"actual": pair[0], "predicted": pair[1], "count": count} for pair, count in errors.most_common(12)],
                      "predictions": [{"source_id": row["source_id"], "label": row["label"], "prediction": pred}
                                      for row, pred in zip(validation, predictions)]}
            output["results"].append(result)
            write_json(args.output_dir / "summary.json", output)
            print(f"{scenario}/{method}: {correct}/{len(truth)} = {100 * metrics['accuracy']:.1f}%", flush=True)
    output["status"] = "completed"
    output["uncertainty_note"] = "Nominal Wilson intervals assume independent representative examples; they do not account for validation reuse, multiple methods or sample-selection effects."
    write_json(args.output_dir / "summary.json", output)


if __name__ == "__main__":
    main()
