"""Fixed validation-only centralized classifiers on the frozen MPNet cache.

This is task-efficiency context, not federated learning or LoRA. The eight
configurations are frozen in plan.json before fitting or scoring. Only the five
training feature partitions and reused development-validation features are read;
there is no test/confirmation loader, score-based search, or winner selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import warnings

# Set before NumPy/SciPy/sklearn imports; threadpoolctl also limits loaded pools.
THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                      "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREAD_ENVIRONMENT:
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedlora_edge.data import INTENTS, LABEL_CLIENT, content_hash, write_json  # noqa: E402
from fedlora_edge.encoder_data import load_encoder_manifest, load_feature_split  # noqa: E402
from fedlora_edge.training import classification_metrics  # noqa: E402


SOURCE_FILES = ("tools/encoder_reference_ablation.py", "fedlora_edge/encoder_data.py",
                "fedlora_edge/data.py", "fedlora_edge/training.py", "pyproject.toml")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _classifiers():
    """Return the complete fixed sweep; do not alter it based on held-out scores."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    methods = [(f"logistic_c{value:g}", LogisticRegression(
        C=value, class_weight="balanced", max_iter=5000, random_state=42,
        solver="lbfgs", tol=1e-4)) for value in (0.1, 1.0, 10.0, 100.0)]
    methods.extend((f"svc_rbf_c{value:g}", SVC(
        C=value, kernel="rbf", gamma="scale", class_weight=None,
        probability=False, cache_size=200, random_state=42))
        for value in (1.0, 10.0, 100.0))
    methods.append(("mlp_128_early_stop", MLPClassifier(
        hidden_layer_sizes=(128,), activation="relu", solver="adam", alpha=1e-4,
        batch_size=128, learning_rate_init=1e-3, max_iter=500, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=15, tol=1e-4, random_state=42)))
    return methods


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _split_identity(entry, ids, labels):
    return {"file": entry["file"], "sha256": entry["sha256"],
            "records_hash": entry["records_hash"], "count": len(labels),
            "source_ids_hash": content_hash(ids.tolist()),
            "label_counts": dict(Counter(INTENTS[int(i)] for i in labels))}


def _load_data(dataset_dir):
    import numpy as np

    manifest = load_encoder_manifest(dataset_dir)
    if set(manifest["clients"]) != {str(i) for i in range(5)}:
        raise ValueError("The reference requires exactly the five frozen client partitions.")
    clients = [load_feature_split(dataset_dir, client_id=i) for i in range(5)]
    X = np.concatenate([part[0] for part in clients])
    y = np.concatenate([part[1] for part in clients])
    ids = np.concatenate([part[2] for part in clients])
    Xval, yval, val_ids = load_feature_split(dataset_dir, client_id=None)
    if len(X) != 2760 or len(Xval) != 200:
        raise ValueError("Expected the frozen 2,760 training and 200 validation examples.")
    if (len(set(ids.tolist())) != len(ids) or set(ids.tolist()) & set(val_ids.tolist())
            or any(not source_id.startswith("train:") for source_id in [*ids, *val_ids])):
        raise ValueError("Training/development source IDs are duplicated, overlapping, or nontraining.")
    if Counter(yval.tolist()) != Counter({label: 10 for label in range(len(INTENTS))}):
        raise ValueError("Expected ten development-validation examples per intent.")
    for client_id, (_, labels, _) in enumerate(clients):
        if set(labels.tolist()) != {i for i, label in enumerate(INTENTS)
                                    if LABEL_CLIENT[label] == client_id}:
            raise ValueError("Client label ownership differs from the frozen domain partition.")
    return manifest, clients, (X, y, ids), (Xval, yval, val_ids)


def run(dataset_dir: Path, output: Path):
    import numpy as np
    from sklearn.metrics import log_loss
    from threadpoolctl import threadpool_info, threadpool_limits

    dataset_dir, output = dataset_dir.resolve(), output.resolve()
    run_root = (ROOT / "runs").resolve()
    if not output.is_relative_to(run_root) or output == run_root:
        raise ValueError("Reference outputs, including source IDs, must stay in ignored runs/.")
    if output.exists():
        raise FileExistsError("Choose a new run directory to preserve the frozen run identity.")
    manifest, clients, train, validation = _load_data(dataset_dir)
    X, y, ids = train
    Xval, yval, val_ids = validation
    methods = _classifiers()
    source_bytes = {name: (ROOT / name).read_bytes() for name in SOURCE_FILES}
    source_hashes = {name: hashlib.sha256(value).hexdigest() for name, value in source_bytes.items()}
    manifest_file_hash = _sha256(dataset_dir / "manifest.json")
    split_identities = {str(i): _split_identity(manifest["clients"][str(i)], part[2], part[1])
                        for i, part in enumerate(clients)}
    split_identities["validation"] = _split_identity(manifest["validation"], val_ids, yval)
    with threadpool_limits(limits=1):
        plan = {
            "format_version": 1, "run_id": output.name,
            "kind": "centralized_frozen_mpnet_reference_ablation", "created_utc": _utc(),
            "split": "validation", "training_examples": len(X), "validation_examples": len(Xval),
            "dataset_dir": str(dataset_dir), "manifest_hash": manifest["manifest_hash"],
            "manifest_file_sha256": manifest_file_hash, "feature_dim": manifest["feature_dim"],
            "labels": list(INTENTS), "model_revision": manifest["model_revision"],
            "encoder_identity_hash": manifest["encoder_identity_hash"], "encoder": manifest["encoder"],
            "source_reference_run": manifest["source_reference_run"],
            "source_reference_config_hash": manifest["source_reference_config_hash"],
            "source_manifest_set_hash": manifest["source_manifest_set_hash"],
            "source_cache_sha256": manifest["source_cache_sha256"],
            "split_identities": split_identities, "training_source_ids_hash": content_hash(ids.tolist()),
            "source_git_revision": _git_revision(), "source_files_sha256": source_hashes,
            "methods": [{"method_id": name, "estimator": type(model).__name__,
                         "parameters": model.get_params(deep=False)} for name, model in methods],
            "method_count": len(methods), "cpu_threads": 1,
            "runtime": {"python": sys.version, "python_executable": sys.executable,
                        "platform": platform.platform(), "machine": platform.machine(),
                        "packages": {name: importlib.metadata.version(name)
                                     for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
                        "thread_environment": {name: os.environ[name] for name in THREAD_ENVIRONMENT},
                        "threadpools": threadpool_info()},
            "scoring_rule": "Twenty-class predict(); shared project accuracy/macro-F1/domain metrics.",
            "selection_rule": "No automatic selection; score all eight fixed configurations once and retain all outcomes.",
            "mlp_stopping_rule": "Sklearn stratified 10% internal split from the 2,760 training rows only; seed 42. The 200 development rows are never passed to fit().",
            "data_access": "Only manifest.json, clients/client_0..4.npz, and validation.npz. No test or confirmation records are opened.",
            "interpretation": "Centralized public-data task references on reused development validation, not independent confirmation, federated training, or LoRA. SVC support-vector dependence is not transferable to federation by claim.",
        }
        plan["config_hash"] = content_hash(plan)
        output.mkdir(parents=True)
        for name, value in source_bytes.items():
            snapshot = output / "source_snapshot" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(value)
        write_json(output / "source_ids.json", {"training": ids.tolist(), "validation": val_ids.tolist()})
        # This durable plan is written before the first fit(), predict(), or score.
        write_json(output / "plan.json", plan)
        plan_file_hash = _sha256(output / "plan.json")
        print(json.dumps({"status": "plan_frozen", "run_id": output.name,
                          "config_hash": plan["config_hash"], "method_count": len(methods)}), flush=True)
        results = []
        started = time.perf_counter()
        for name, model in methods:
            if _sha256(output / "plan.json") != plan_file_hash:
                raise ValueError("The frozen plan changed during the run.")
            result = {"method_id": name, "config_hash": plan["config_hash"],
                      "split": "validation", "started_utc": _utc(), "status": "running"}
            write_json(output / "progress.json", {"status": "running", "current_method": name,
                                                   "completed_methods": len(results), "method_count": len(methods)})
            captured = []
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    stage_start = time.perf_counter()
                    model.fit(X, y)
                    result["fit_seconds"] = time.perf_counter() - stage_start
                    stage_start = time.perf_counter()
                    predictions = model.predict(Xval)
                    result["predict_seconds"] = time.perf_counter() - stage_start
                    result["metrics"] = classification_metrics(
                        [INTENTS[int(i)] for i in yval], [INTENTS[int(i)] for i in predictions])
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(Xval)
                        result["metrics"]["cross_entropy"] = float(log_loss(yval, probabilities, labels=np.arange(len(INTENTS))))
                    else:
                        result["metrics"]["cross_entropy"] = None
                        result["probability_note"] = "SVC probability=False; no probability calibration or comparable log loss is claimed."
                    result["predictions"] = [
                        {"source_id": str(source_id), "label": INTENTS[int(actual)],
                         "prediction": INTENTS[int(predicted)]}
                        for source_id, actual, predicted in zip(val_ids, yval, predictions)]
                    result["fit_iterations"] = np.asarray(model.n_iter_).tolist()
                    if hasattr(model, "n_support_"):
                        result["support_vectors_per_class"] = model.n_support_.tolist()
                        result["support_vector_count"] = int(model.n_support_.sum())
                    if hasattr(model, "validation_scores_"):
                        result["internal_training_holdout_accuracy_history"] = model.validation_scores_
                        result["internal_training_holdout_best_accuracy"] = float(model.best_validation_score_)
                        result["training_loss_history"] = model.loss_curve_
                    result["status"] = "completed"
            except Exception as exc:
                result.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                               "traceback": traceback.format_exc()})
            result["warnings"] = [{"category": type(warning.message).__name__, "message": str(warning.message)}
                                  for warning in captured]
            result["finished_utc"] = _utc()
            write_json(output / "methods" / f"{name}.json", result)
            results.append(result)
            print(json.dumps({key: result[key] for key in ("method_id", "status", "metrics", "error") if key in result}), flush=True)
        source_unchanged = all(_sha256(ROOT / name) == digest for name, digest in source_hashes.items())
        inputs_unchanged = (_sha256(dataset_dir / "manifest.json") == manifest_file_hash
                            and all(_sha256(dataset_dir / entry["file"]) == entry["sha256"]
                                    for entry in split_identities.values()))
        status = ("completed" if all(row["status"] == "completed" for row in results)
                  and source_unchanged and inputs_unchanged else "completed_with_failures")
        summary = {"status": status, "plan": plan, "plan_file_sha256": plan_file_hash,
                   "results": results, "method_count": len(methods), "finished_utc": _utc(),
                   "wall_seconds": time.perf_counter() - started,
                   "source_unchanged_during_run": source_unchanged,
                   "input_files_unchanged_during_run": inputs_unchanged,
                   "test_or_confirmation_scored": False}
        write_json(output / "summary.json", summary)
        write_json(output / "progress.json", {"status": status, "completed_methods": len(results),
                                               "method_count": len(methods), "config_hash": plan["config_hash"]})
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "assets/encoder-bank20-v1")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/encoder-reference-ablation-v1")
    args = parser.parse_args()
    result = run(args.dataset_dir, args.run_dir)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
