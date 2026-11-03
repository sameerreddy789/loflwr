"""Identity-checked, local frozen-encoder features for a separate BANKING20 variant.

The original public-reference cache is partitioned once for a one-host simulation.
Neither the server nor a client needs another client's feature archive at runtime.
Confirmation queries remain text-only and are not exposed by feature loaders.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .data import (INTENTS, LABEL_CLIENT, SOURCE_SHA256, content_hash, load_manifest,
                   normalize_query, read_source, verify_manifests, write_json)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_encoder_manifest(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("kind") != "mpnet_local_feature_dataset":
        raise ValueError("Unsupported encoder dataset.")
    if content_hash({k: v for k, v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("Encoder manifest hash mismatch.")
    if manifest["labels"] != list(INTENTS) or manifest["feature_dim"] != 768:
        raise ValueError("Expected the declared twenty labels and 768 features.")
    return manifest


def _local_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Dataset path escapes its root.")
    return path


def load_feature_split(root, client_id=None):
    """Load only one client's training features, or server validation features."""
    import numpy as np
    manifest = load_encoder_manifest(root)
    entry = manifest["validation"] if client_id is None else manifest["clients"][str(client_id)]
    path = _local_path(root, entry["file"])
    if sha256(path) != entry["sha256"]:
        raise ValueError("Local feature file checksum mismatch.")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"X", "y", "source_ids"}:
            raise ValueError("Unexpected feature archive members.")
        X, y, ids = data["X"], data["y"], data["source_ids"]
    if (X.shape != (entry["count"], manifest["feature_dim"])
            or y.shape != (len(X),) or ids.shape != (len(X),)
            or X.dtype != np.float32 or y.dtype.kind not in "iu" or ids.dtype.kind not in "US"
            or not np.isfinite(X).all() or (y < 0).any() or (y >= len(INTENTS)).any()
            or len(set(ids.tolist())) != len(ids)
            or not np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)):
        raise ValueError("Invalid local feature values, shapes, labels or normalization.")
    return X, y, ids


def load_encoder_records(root, client_id=None):
    """Load one identified training/validation text split for live encoder passes."""
    manifest = load_encoder_manifest(root)
    entry = manifest["validation"] if client_id is None else manifest["clients"][str(client_id)]
    data = json.loads(_local_path(root, entry["records_file"]).read_text(encoding="utf-8"))
    records = data["records"]
    if (len(records) != entry["count"] or content_hash(records) != entry["records_hash"]
            or data["records_hash"] != entry["records_hash"]):
        raise ValueError("Encoder text records do not match the frozen manifest.")
    return records


def prepare_encoder_dataset(reference_dir, data_dir, output_dir):
    import numpy as np
    reference_dir, data_dir, output_dir = map(Path, (reference_dir, data_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError("Use a new dataset directory to preserve experiment identities.")
    verified = verify_manifests(data_dir)
    summary = json.loads((reference_dir / "summary.json").read_text(encoding="utf-8"))
    plan = summary["plan"]
    if (summary["status"] != "completed" or plan["split"] != "validation"
            or content_hash({k: v for k, v in plan.items() if k != "config_hash"}) != plan["config_hash"]
            or plan["manifest_set_hash"] != verified["manifest_set_hash"]):
        raise ValueError("Reference or original manifest identity mismatch.")
    train = json.loads((reference_dir / "expanded_training_manifest.json").read_text(encoding="utf-8"))["records"]
    validation = load_manifest(data_dir / "manifests/validation.json")
    historical_test = load_manifest(data_dir / "manifests/test.json")
    if (content_hash(train) != plan["scenarios"]["all_eligible_training"]["records_hash"]
            or content_hash(validation) != plan["validation_records_hash"]):
        raise ValueError("Training/validation records changed after reference preparation.")
    for name in ("train.csv", "test.csv"):
        if sha256(data_dir / "source" / name) != SOURCE_SHA256[name]:
            raise ValueError("Original source checksum mismatch.")
    official_test = read_source(data_dir / "source/test.csv", "test")
    forbidden = {normalize_query(r["query"]) for r in [*official_test, *validation]}
    if any(r["source_split"] != "train" or normalize_query(r["query"]) in forbidden for r in train):
        raise ValueError("Held-out leakage in expanded training.")
    if len({normalize_query(r["query"]) for r in train}) != len(train):
        raise ValueError("Duplicate training queries.")
    used_test = {normalize_query(r["query"]) for r in historical_test}
    confirmation = [r for r in official_test if r["label"] in INTENTS
                    and normalize_query(r["query"]) not in used_test]
    if (Counter(r["label"] for r in confirmation) != Counter({label: 15 for label in INTENTS})
            or len({normalize_query(r["query"]) for r in confirmation}) != 300):
        raise ValueError("Expected exactly 300 disjoint previously unscored confirmation queries.")
    metadata = json.loads((reference_dir / "embedding_ids.json").read_text(encoding="utf-8"))
    ids = metadata["ids"]
    vectors = np.load(reference_dir / "reference_embeddings.npy", allow_pickle=False)
    if len(set(ids)) != len(ids) or vectors.shape != (len(ids), 768) or not np.isfinite(vectors).all():
        raise ValueError("Reference embedding cache does not align with source identifiers.")
    if metadata["model_files_sha256"] != plan["encoder"]["files_sha256"]:
        raise ValueError("Reference encoder asset identity mismatch.")
    index = {key: i for i, key in enumerate(ids)}
    output_dir.mkdir(parents=True)
    (output_dir / "clients").mkdir()

    def save_split(rows, stem):
        feature_path = output_dir / (stem + ".npz")
        record_path = output_dir / (stem + ".json")
        X = vectors[[index[r["source_id"]] for r in rows]].astype(np.float32)
        y = np.asarray([INTENTS.index(r["label"]) for r in rows], dtype=np.int64)
        np.savez(feature_path, X=X, y=y, source_ids=np.asarray([r["source_id"] for r in rows]))
        write_json(record_path, {"records": rows, "records_hash": content_hash(rows)})
        return {"file": feature_path.relative_to(output_dir).as_posix(), "sha256": sha256(feature_path),
                "records_file": record_path.relative_to(output_dir).as_posix(),
                "records_hash": content_hash(rows), "count": len(rows),
                "label_counts": dict(Counter(r["label"] for r in rows))}

    manifest = {"format_version": 1, "kind": "mpnet_local_feature_dataset", "feature_dim": 768,
                "labels": list(INTENTS), "model_revision": plan["encoder"]["revision"],
                "encoder": plan["encoder"], "encoder_identity_hash": content_hash(plan["encoder"]),
                "source_reference_run": plan["run_id"], "source_reference_config_hash": plan["config_hash"],
                "source_cache_sha256": sha256(reference_dir / "reference_embeddings.npy"),
                "source_manifest_set_hash": verified["manifest_set_hash"], "clients": {},
                "preparation_note": "Public benchmark simulation prepared from the centralized frozen reference cache. Runtime clients load only their own features; the server receives parameter updates. This does not demonstrate distributed feature preparation or physical-device isolation."}
    for client in range(5):
        manifest["clients"][str(client)] = save_split([r for r in train if LABEL_CLIENT[r["label"]] == client], f"clients/client_{client}")
    manifest["validation"] = save_split(validation, "validation")
    write_json(output_dir / "confirmation.json", {"split": "confirmation_unscored", "records": confirmation,
               "records_hash": content_hash(confirmation), "selection_rule": "All selected-intent official-test queries outside the historical 500-query test; fifteen per intent. Do not evaluate until model/config selection is frozen."})
    manifest["confirmation"] = {"records_file": "confirmation.json", "count": len(confirmation),
                                "records_hash": content_hash(confirmation), "status_at_preparation": "unscored"}
    manifest["manifest_hash"] = content_hash(manifest)
    write_json(output_dir / "manifest.json", manifest)
    for client in range(5):
        load_feature_split(output_dir, client)
    load_feature_split(output_dir)
    return manifest
