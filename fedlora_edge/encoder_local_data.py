"""Independently prepare frozen-MPNet features from one participant's records.

The coordinator assembles metadata only. A participant receives the common plan,
its own pre-partitioned record bundles, and the pinned encoder. No preparation
operation reads the historical test, confirmation, or another client's files.
"""
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from .data import INTENTS, LABEL_CLIENT, content_hash, normalize_query, write_json
from .encoder_adapter import MODEL_ID, MODEL_REVISION, POOLING
from .encoder_data import sha256

LAYOUTS = {"five": {str(i): [str(i)] for i in range(5)},
           "three": {"0": ["0", "1"], "1": ["2", "3"], "2": ["4"]}}
FEATURE_SETTINGS = {"dtype": "float32", "device": "cpu", "max_length": 384,
                    "pooling": POOLING, "batch_size": 8, "cpu_threads": 1}


def _check_hash(value, field):
    if not isinstance(value, dict) or content_hash({k: v for k, v in value.items() if k != field}) != value.get(field):
        raise ValueError(f"Invalid {field}.")


def _counts(value):
    if (not isinstance(value, dict) or not value or not set(value) <= set(INTENTS)
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in value.values())):
        raise ValueError("Invalid label counts.")
    return value


def _part(entry, original_client):
    counts = _counts(entry["label_counts"])
    if sum(counts.values()) != entry["count"]:
        raise ValueError("Record count differs from label counts.")
    if original_client is not None and any(str(LABEL_CLIENT[label]) != original_client for label in counts):
        raise ValueError("Source client owns labels from another domain.")
    return {"original_client": original_client, "records_hash": entry["records_hash"],
            "count": entry["count"], "label_counts": dict(counts)}


def build_local_plan(source_manifest, layout="five"):
    """Read a frozen manifest's metadata; never load any source query/archive."""
    _check_hash(source_manifest, "manifest_hash")
    if (layout not in LAYOUTS or source_manifest.get("kind") != "mpnet_local_feature_dataset"
            or source_manifest.get("format_version") != 1 or source_manifest.get("feature_dim") != 768
            or source_manifest.get("labels") != list(INTENTS)
            or set(source_manifest["clients"]) != set(map(str, range(5)))):
        raise ValueError("Expected the frozen five-domain encoder source and a known layout.")
    encoder = source_manifest["encoder"]
    if (encoder.get("model_id") != MODEL_ID or encoder.get("revision") != MODEL_REVISION
            or encoder.get("pooling") != POOLING or encoder.get("max_length") != 384
            or source_manifest.get("model_revision") != MODEL_REVISION
            or source_manifest.get("encoder_identity_hash") != content_hash(encoder)):
        raise ValueError("Unexpected encoder identity or feature recipe.")
    participants = {}
    for participant, originals in LAYOUTS[layout].items():
        parts = [_part(source_manifest["clients"][client], client) for client in originals]
        participants[participant] = {"role": "client", "parts": parts}
    participants["server"] = {"role": "validation", "parts": [_part(source_manifest["validation"], None)]}
    plan = {"kind": "mpnet_participant_local_preparation_v1", "format_version": 1,
            "layout": layout, "source_manifest_hash": source_manifest["manifest_hash"],
            "source_manifest_set_hash": source_manifest.get("source_manifest_set_hash"),
            "encoder": copy.deepcopy(encoder), "encoder_identity_hash": content_hash(encoder),
            "model_revision": MODEL_REVISION, "labels": list(INTENTS), "feature_dim": 768,
            "feature_settings": dict(FEATURE_SETTINGS), "participants": participants,
            "boundary": "Approved public record hashes are fixed upstream. Only each participant's own bundles are encoded locally; registration contains metadata only."}
    plan["plan_hash"] = content_hash(plan)
    return plan


def validate_plan(plan):
    _check_hash(plan, "plan_hash")
    if (plan.get("kind") != "mpnet_participant_local_preparation_v1" or plan.get("format_version") != 1
            or plan.get("layout") not in LAYOUTS or plan.get("feature_settings") != FEATURE_SETTINGS
            or plan.get("feature_dim") != 768 or plan.get("labels") != list(INTENTS)
            or plan.get("model_revision") != MODEL_REVISION
            or plan.get("encoder_identity_hash") != content_hash(plan["encoder"])
            or set(plan["participants"]) != {*LAYOUTS[plan["layout"]], "server"}):
        raise ValueError("Invalid local preparation contract.")
    for participant, descriptor in plan["participants"].items():
        expected_originals = [None] if participant == "server" else LAYOUTS[plan["layout"]][participant]
        if (descriptor["role"] != ("validation" if participant == "server" else "client")
                or [part["original_client"] for part in descriptor["parts"]] != expected_originals):
            raise ValueError("Participant role or source ownership was modified.")
        for part in descriptor["parts"]:
            if _part(part, part["original_client"]) != part:
                raise ValueError("Invalid participant source metadata.")
    return plan


def load_own_records(plan, participant, record_files):
    """Open only explicitly supplied bundles; reorder them by frozen part order."""
    validate_plan(plan)
    participant = str(participant)
    if participant not in plan["participants"]:
        raise ValueError("Unknown participant.")
    expected = plan["participants"][participant]["parts"]
    if len(record_files) != len(expected):
        raise ValueError("Supply exactly this participant's record bundles.")
    supplied = {}
    for name in record_files:
        bundle = json.loads(Path(name).read_text(encoding="utf-8"))
        records = bundle.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("Expected a nonempty local records bundle.")
        digest = content_hash(records)
        if digest != bundle.get("records_hash") or digest in supplied:
            raise ValueError("Local record hash mismatch or duplicate bundle.")
        supplied[digest] = records
    if set(supplied) != {part["records_hash"] for part in expected}:
        raise ValueError("Record bundles do not belong to this participant's frozen plan.")
    result, seen_ids, seen_queries = [], set(), set()
    for part in expected:
        records = supplied[part["records_hash"]]
        if len(records) != part["count"]:
            raise ValueError("Local source count mismatch.")
        for row in records:
            query, label, source_id = row.get("query"), row.get("label"), row.get("source_id")
            if (not isinstance(query, str) or not normalize_query(query) or label not in INTENTS
                    or row.get("source_split") != "train" or not isinstance(source_id, str)
                    or re.fullmatch(r"train:[1-9][0-9]*", source_id) is None):
                raise ValueError("Expected approved original-training rows with valid labels and IDs.")
            key = normalize_query(query)
            if source_id in seen_ids or key in seen_queries:
                raise ValueError("Duplicate local source ID or normalized query.")
            if part["original_client"] is not None and str(LABEL_CLIENT[label]) != part["original_client"]:
                raise ValueError("A local row belongs to another original domain.")
            seen_ids.add(source_id)
            seen_queries.add(key)
        if dict(Counter(row["label"] for row in records)) != part["label_counts"]:
            raise ValueError("Local source label counts mismatch.")
        result.extend(records)
    return result


def encode_batches(model, tokenizer, queries, batch_size=8):
    """The frozen inference recipe, independent of head/checkpoint loading."""
    import numpy as np
    import torch
    if (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
            or not queries or any(not isinstance(q, str) or not q.strip() for q in queries)):
        raise ValueError("Expected nonempty queries and positive batch size.")
    vectors, truncated = [], 0
    with torch.inference_mode():
        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            lengths = tokenizer(batch, padding=False, truncation=False, return_length=True)["length"]
            truncated += sum(length > 384 for length in lengths)
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=384, return_tensors="pt")
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
    X = np.concatenate(vectors).astype(np.float32, copy=False)
    if X.shape != (len(queries), 768) or not np.isfinite(X).all() or not np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5):
        raise ValueError("Encoder produced invalid or unnormalized features.")
    return X, truncated


def encode_local_records(records, asset_dir, expected_encoder):
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    from .encoder_inference import verify_inference_assets
    from .training import _MemoryMonitor
    torch.set_num_threads(FEATURE_SETTINGS["cpu_threads"])
    started = time.perf_counter()
    with _MemoryMonitor(torch.device("cpu")) as memory:
        assets = verify_inference_assets(asset_dir)
        if assets["files_sha256"] != expected_encoder["files_sha256"]:
            raise ValueError("Local encoder assets differ from the common plan.")
        tokenizer = AutoTokenizer.from_pretrained(str(asset_dir), local_files_only=True)
        tokenizer.padding_side = "right"
        model = AutoModel.from_pretrained(str(asset_dir), local_files_only=True, dtype=torch.float32).cpu().eval()
        model.requires_grad_(False)
        X, truncated = encode_batches(model, tokenizer, [row["query"] for row in records], FEATURE_SETTINGS["batch_size"])
    return X, {"encoding_seconds_including_asset_verification_and_model_loading": time.perf_counter()-started,
               "truncated": int(truncated), "memory": memory.metrics(), "torch": torch.__version__,
               "transformers": transformers.__version__, "feature_settings": dict(FEATURE_SETTINGS),
               "memory_scope": "Sampled current preparation process RSS; encoder is in this process, no child process is launched."}


def _entry_paths(participant):
    stem = "validation" if participant == "server" else f"clients/client_{participant}"
    return {"file": f"{stem}.npz", "records_file": f"{stem}.json"}


def prepare_local_features(plan, record_files, output_dir, asset_dir, participant):
    import numpy as np
    participant = str(participant)
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError("Use a new participant output directory.")
    records = load_own_records(plan, participant, record_files)
    # Hash/ownership validation precedes model loading or creation of output files.
    X, telemetry = encode_local_records(records, asset_dir, plan["encoder"])
    if X.dtype != np.float32 or X.shape != (len(records), 768) or not np.isfinite(X).all():
        raise ValueError("Invalid prepared feature matrix.")
    paths = _entry_paths(participant)
    feature = root / paths["file"]
    feature.parent.mkdir(parents=True, exist_ok=False)
    np.savez(feature, X=X, y=np.asarray([INTENTS.index(r["label"]) for r in records], dtype=np.int64),
             source_ids=np.asarray([r["source_id"] for r in records]))
    write_json(root / paths["records_file"], {"records": records, "records_hash": content_hash(records)})
    entry = {**paths, "sha256": sha256(feature), "count": len(records), "records_hash": content_hash(records),
             "label_counts": dict(Counter(r["label"] for r in records))}
    registration = {"kind": "mpnet_local_registration_v1", "plan_hash": plan["plan_hash"],
                    "participant": participant, "role": plan["participants"][participant]["role"],
                    "encoder_identity_hash": plan["encoder_identity_hash"], "entry": entry}
    registration["registration_hash"] = content_hash(registration)
    write_json(root / "registration.json", registration)
    write_json(root / "preparation.json", {"status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
               "participant": participant, "plan_hash": plan["plan_hash"], "registration_hash": registration["registration_hash"],
               "count": len(records), "raw_feature_bytes": int(X.nbytes), "archive_bytes": feature.stat().st_size,
               "source_record_bundle_hashes": [part["records_hash"] for part in plan["participants"][participant]["parts"]],
               "telemetry": telemetry, "interpretation": "Local offline feature preparation, not model training or an accuracy result."})
    return registration


def validate_registration(plan, registration):
    validate_plan(plan)
    _check_hash(registration, "registration_hash")
    fields = {"kind", "plan_hash", "participant", "role", "encoder_identity_hash", "entry", "registration_hash"}
    if set(registration) != fields or registration["participant"] not in plan["participants"]:
        raise ValueError("Unknown registration fields or participant; metadata only is permitted.")
    participant = registration["participant"]
    descriptor = plan["participants"][participant]
    expected_counts = Counter()
    for part in descriptor["parts"]:
        expected_counts.update(part["label_counts"])
    entry = registration["entry"]
    if (registration["kind"] != "mpnet_local_registration_v1" or registration["plan_hash"] != plan["plan_hash"]
            or registration["encoder_identity_hash"] != plan["encoder_identity_hash"] or registration["role"] != descriptor["role"]
            or set(entry) != {"file", "records_file", "sha256", "count", "records_hash", "label_counts"}
            or {k: entry[k] for k in _entry_paths(participant)} != _entry_paths(participant)
            or entry["count"] != sum(expected_counts.values()) or entry["label_counts"] != dict(expected_counts)
            or any(not isinstance(entry[k], str) or re.fullmatch(r"[0-9a-f]{64}", entry[k]) is None for k in ("sha256", "records_hash"))):
        raise ValueError("Registration does not match the local preparation plan.")
    if len(descriptor["parts"]) == 1 and entry["records_hash"] != descriptor["parts"][0]["records_hash"]:
        raise ValueError("Single-source record identity changed.")
    return registration


def assemble_local_manifest(plan, registrations):
    """Combine metadata registrations; no feature or query path is opened."""
    validate_plan(plan)
    entries = {}
    for registration in registrations:
        validate_registration(plan, registration)
        participant = registration["participant"]
        if participant in entries:
            raise ValueError("Duplicate participant registration.")
        entries[participant] = copy.deepcopy(registration["entry"])
    if set(entries) != set(plan["participants"]):
        raise ValueError("A participant registration is missing.")
    manifest = {"format_version": 1, "kind": "mpnet_local_feature_dataset", "feature_dim": 768,
                "labels": list(INTENTS), "model_revision": plan["model_revision"], "encoder": copy.deepcopy(plan["encoder"]),
                "encoder_identity_hash": plan["encoder_identity_hash"], "source_manifest_set_hash": plan["source_manifest_set_hash"],
                "source_manifest_hash": plan["source_manifest_hash"], "local_preparation_plan_hash": plan["plan_hash"],
                "layout": plan["layout"], "clients": {participant: entries[participant] for participant in LAYOUTS[plan["layout"]]},
                "validation": entries["server"], "feature_settings": dict(FEATURE_SETTINGS),
                "preparation_note": "Each participant independently encoded only its approved local records. The server assembles this metadata without client queries or features. Same-host isolation and physical deployment are separate evidence gates."}
    manifest["manifest_hash"] = content_hash(manifest)
    return manifest
