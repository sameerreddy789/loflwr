"""Opt-in length bucketing for identity-checked, participant-local MPNet features.

This is a new preparation recipe. The locked encoder, prior preparation code,
and evaluation artifacts remain unchanged. Features return in source row order.
"""
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
from pathlib import Path
import time

from .data import INTENTS, content_hash, write_json
from .encoder_adapter import MODEL_ID, MODEL_REVISION, POOLING
from .encoder_data import sha256
from .encoder_local_data import (
    FEATURE_SETTINGS as SOURCE_SETTINGS,
    assemble_local_manifest,
    build_local_plan,
    load_own_records,
    validate_plan as validate_source_plan,
    validate_registration as validate_source_registration,
)

PLAN_KIND = "mpnet_bucketed_local_preparation_v1"
REGISTRATION_KIND = "mpnet_bucketed_local_registration_v1"
FEATURE_SETTINGS = {
    **SOURCE_SETTINGS,
    "batching": "stable_exact_token_length_ascending",
    "output_order": "source_record_order",
    "raw_tokenization": "once_without_special_tokens",
    "special_token_preparation": "tokenizer_prepare_for_model",
    "padding_side": "right",
    "truncation_side": "right",
}


def _check_hash(value, field):
    if (not isinstance(value, dict)
            or content_hash({k: v for k, v in value.items() if k != field}) != value.get(field)):
        raise ValueError(f"Invalid {field}.")


def build_plan(source_manifest, layout="five"):
    """Build a new recipe identity using only approved source metadata."""
    plan = {
        "kind": PLAN_KIND,
        "format_version": 1,
        "layout": layout,
        "source_plan": build_local_plan(source_manifest, layout),
        "feature_settings": dict(FEATURE_SETTINGS),
        "recipe_hash": content_hash(FEATURE_SETTINGS),
    }
    plan["plan_hash"] = content_hash(plan)
    return validate_plan(plan)


def validate_plan(plan):
    _check_hash(plan, "plan_hash")
    if (set(plan) != {"kind", "format_version", "layout", "source_plan", "feature_settings",
                      "recipe_hash", "plan_hash"}
            or plan["kind"] != PLAN_KIND or plan["format_version"] != 1
            or plan["feature_settings"] != FEATURE_SETTINGS
            or plan["recipe_hash"] != content_hash(FEATURE_SETTINGS)):
        raise ValueError("Invalid bucketed preparation contract.")
    validate_source_plan(plan["source_plan"])
    if plan["layout"] != plan["source_plan"]["layout"]:
        raise ValueError("Bucketed layout differs from the approved source plan.")
    encoder = plan["source_plan"]["encoder"]
    if (encoder.get("model_id") != MODEL_ID or encoder.get("revision") != MODEL_REVISION
            or encoder.get("pooling") != POOLING or encoder.get("max_length") != 384):
        raise ValueError("Bucketed plan requires the pinned encoder and pooling recipe.")
    return plan


def _batch_work(lengths, order, batch_size):
    padded, attention = 0, 0
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        longest = max(lengths[index] for index in indices)
        padded += len(indices) * longest
        attention += len(indices) * longest * longest
    return padded, attention


def _validate_features(X, count):
    import numpy as np
    if (X.dtype != np.float32 or X.shape != (count, 768) or not np.isfinite(X).all()
            or not np.allclose(np.linalg.norm(X, axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError("Encoder produced invalid or unnormalized features.")


def encode_bucketed(model, tokenizer, queries, batch_size=8):
    """Encode once-tokenized strings in length order, restoring original rows.

The helper accepts a batch size for comparison fixtures. The v1 preparation
entry point fixes CPU FP32, one CPU thread, and batches of eight in its identity.
Attention cells are an arithmetic padding proxy, not measured FLOPs or latency.
"""
    import numpy as np
    import torch

    if (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
            or not isinstance(queries, (list, tuple)) or not queries
            or any(not isinstance(query, str) or not query.strip() for query in queries)):
        raise ValueError("Expected nonempty queries and positive batch size.")
    if getattr(model, "training", True):
        raise ValueError("A frozen encoder in evaluation mode is required.")
    if not callable(getattr(model, "parameters", None)) or not callable(getattr(model, "buffers", None)):
        raise ValueError("Expected an inspectable frozen Torch encoder.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("All encoder parameters must be frozen.")
    if any(tensor.device.type != "cpu" or (tensor.is_floating_point() and tensor.dtype != torch.float32)
           for tensor in (*model.parameters(), *model.buffers())):
        raise ValueError("The bucketed recipe requires CPU FP32 encoder tensors.")
    if (getattr(tokenizer, "padding_side", None) != "right"
            or getattr(tokenizer, "truncation_side", None) != "right"):
        raise ValueError("The bucketed recipe requires right padding and truncation.")

    raw = tokenizer(queries, add_special_tokens=False, padding=False,
                    truncation=False, return_attention_mask=False)["input_ids"]
    if (not isinstance(raw, list) or len(raw) != len(queries)
            or any(not isinstance(ids, list) or any(isinstance(i, bool) or not isinstance(i, int) or i < 0
                                                   for i in ids) for ids in raw)):
        raise ValueError("Tokenizer returned malformed token IDs.")
    specials = tokenizer.num_special_tokens_to_add(pair=False)
    if isinstance(specials, bool) or not isinstance(specials, int) or not 0 < specials < 384:
        raise ValueError("Invalid tokenizer special-token count.")
    original_lengths = [len(ids) + specials for ids in raw]
    clipped_lengths = [min(length, 384) for length in original_lengths]
    prepared = []
    for ids, expected_length in zip(raw, clipped_lengths):
        # Let the tokenizer truncate content before adding its boundary tokens.
        # Slicing a tokenized sequence here would silently discard the true EOS.
        encoded = tokenizer.prepare_for_model(
            ids, add_special_tokens=True, truncation=True, max_length=384,
            return_attention_mask=True, padding=False,
        )
        if (len(encoded["input_ids"]) != expected_length
                or encoded["attention_mask"] != [1] * expected_length):
            raise ValueError("Prepared tokens differ from the declared truncation recipe.")
        prepared.append(encoded)
    order = sorted(range(len(queries)), key=lambda index: (clipped_lengths[index], index))
    baseline_padded, baseline_attention = _batch_work(clipped_lengths, list(range(len(queries))), batch_size)
    candidate_padded, candidate_attention = _batch_work(clipped_lengths, order, batch_size)
    X = np.empty((len(queries), 768), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            encoded = tokenizer.pad([prepared[index] for index in indices], padding=True, return_tensors="pt")
            hidden = model(**encoded).last_hidden_state
            if hidden.dtype != torch.float32 or hidden.device.type != "cpu":
                raise ValueError("Encoder outputs must be CPU FP32.")
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            X[indices] = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
    _validate_features(X, len(queries))
    truncated = sum(length > 384 for length in original_lengths)
    work = {
        "query_count": len(queries), "batch_size": batch_size,
        "batch_count": (len(queries) + batch_size - 1) // batch_size,
        "raw_token_count": sum(map(len, raw)),
        "special_tokens_per_query": specials,
        "original_token_count": sum(original_lengths),
        "clipped_token_count": sum(clipped_lengths),
        "baseline_padded_token_slots": baseline_padded,
        "candidate_padded_token_slots": candidate_padded,
        "baseline_attention_cells": baseline_attention,
        "candidate_attention_cells": candidate_attention,
        "truncated": truncated,
        "arithmetic_scope": "Token slots and batch_size times padded_length squared; not measured FLOPs or latency.",
    }
    return X, truncated, work


def _encode_records(records, asset_dir, expected_encoder):
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
            raise ValueError("Local encoder assets differ from the approved source plan.")
        tokenizer = AutoTokenizer.from_pretrained(str(asset_dir), local_files_only=True)
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        model = AutoModel.from_pretrained(str(asset_dir), local_files_only=True, dtype=torch.float32).cpu().eval()
        model.requires_grad_(False)
        X, truncated, work = encode_bucketed(model, tokenizer, [row["query"] for row in records],
                                            FEATURE_SETTINGS["batch_size"])
    return X, {
        "encoding_seconds_including_asset_verification_and_model_loading": time.perf_counter() - started,
        "truncated": truncated, "work": work, "memory": memory.metrics(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "feature_settings": dict(FEATURE_SETTINGS),
        "memory_scope": "Sampled current preparation process RSS; no child process is launched.",
    }


def _source_registration(plan, registration):
    """Validate the new identity before making an explicit old-validator proxy."""
    validate_plan(plan)
    _check_hash(registration, "registration_hash")
    if (set(registration) != {"kind", "plan_hash", "participant", "role", "encoder_identity_hash",
                              "entry", "recipe_hash", "registration_hash"}
            or registration["kind"] != REGISTRATION_KIND
            or registration["plan_hash"] != plan["plan_hash"]
            or registration["recipe_hash"] != plan["recipe_hash"]):
        raise ValueError("Registration does not match the bucketed preparation recipe.")
    proxy = copy.deepcopy(registration)
    del proxy["recipe_hash"]
    del proxy["registration_hash"]
    proxy.update(kind="mpnet_local_registration_v1", plan_hash=plan["source_plan"]["plan_hash"])
    proxy["registration_hash"] = content_hash(proxy)
    validate_source_registration(plan["source_plan"], proxy)
    return proxy


def prepare(plan, record_files, output_dir, asset_dir, participant):
    """Prepare only one participant's approved bundles into a fresh directory."""
    import numpy as np

    validate_plan(plan)
    participant = str(participant)
    root = Path(output_dir)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Use a new participant output directory.")
    records = load_own_records(plan["source_plan"], participant, record_files)
    # Ownership and record hashes are checked before model loading or any output.
    X, telemetry = _encode_records(records, asset_dir, plan["source_plan"]["encoder"])
    _validate_features(X, len(records))
    stem = "validation" if participant == "server" else f"clients/client_{participant}"
    paths = {"file": f"{stem}.npz", "records_file": f"{stem}.json"}
    root.mkdir(parents=True, exist_ok=False)
    feature = root / paths["file"]
    feature.parent.mkdir(parents=True, exist_ok=True)
    np.savez(feature, X=X, y=np.asarray([INTENTS.index(row["label"]) for row in records], dtype=np.int64),
             source_ids=np.asarray([row["source_id"] for row in records]))
    write_json(root / paths["records_file"], {"records": records, "records_hash": content_hash(records)})
    registration = {
        "kind": REGISTRATION_KIND, "plan_hash": plan["plan_hash"], "participant": participant,
        "role": plan["source_plan"]["participants"][participant]["role"],
        "encoder_identity_hash": plan["source_plan"]["encoder_identity_hash"], "recipe_hash": plan["recipe_hash"],
        "entry": {**paths, "sha256": sha256(feature), "count": len(records), "records_hash": content_hash(records),
                  "label_counts": dict(Counter(row["label"] for row in records))},
    }
    registration["registration_hash"] = content_hash(registration)
    _source_registration(plan, registration)
    write_json(root / "registration.json", registration)
    write_json(root / "preparation.json", {
        "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "participant": participant, "plan_hash": plan["plan_hash"], "recipe_hash": plan["recipe_hash"],
        "registration_hash": registration["registration_hash"], "count": len(records),
        "raw_feature_bytes": int(X.nbytes), "archive_bytes": feature.stat().st_size,
        "source_record_bundle_hashes": [part["records_hash"]
                                       for part in plan["source_plan"]["participants"][participant]["parts"]],
        "telemetry": telemetry,
        "interpretation": "Opt-in local frozen feature preparation; no training or new accuracy result.",
    })
    return registration


def assemble(plan, registrations):
    """Assemble metadata only, with a new plan and recipe identity in the manifest."""
    validate_plan(plan)
    proxies = [_source_registration(plan, registration) for registration in registrations]
    manifest = assemble_local_manifest(plan["source_plan"], proxies)
    del manifest["manifest_hash"]
    manifest.update(
        local_preparation_plan_hash=plan["plan_hash"],
        source_preparation_plan_hash=plan["source_plan"]["plan_hash"],
        preparation_kind=PLAN_KIND,
        feature_recipe_hash=plan["recipe_hash"],
        feature_settings=dict(FEATURE_SETTINGS),
        preparation_note="Each participant encoded only its approved bundles using the separately identified bucketed recipe. Metadata assembly reads no feature or query files. Output rows retain source order; no new training, accuracy, or physical deployment result is implied.",
    )
    manifest["manifest_hash"] = content_hash(manifest)
    return manifest
