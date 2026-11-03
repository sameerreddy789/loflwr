"""Prepare, lock, then score a narrowly mapped CLINC150 external stress test.

This is not a twenty-intent confirmation set. No training or policy fitting is
performed. Raw licensed source data and predictions stay in ignored directories.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fedlora_edge.data import INTENTS, SOURCE_SHA256, content_hash, normalize_query, read_source, write_json

REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
REPOSITORY = "clinc/oos-eval"
MAPPING = {"card_declined": "declined_card_payment"}
OOS_GROUPS = {
    "unsupported_banking": ["routing", "pin_change", "interest_rate", "credit_score"],
    "unrelated_topics": ["weather", "play_music", "recipe", "book_flight"],
}
CLASSES = tuple(MAPPING) + tuple(label for labels in OOS_GROUPS.values() for label in labels)
DATA_DIR = ROOT / "assets/external-clinc-v1"
RUN_DIR = ROOT / "runs/external-clinc-v1"
PUBLIC_DIR = ROOT / "research/results/external-clinc-v1"
ORIGINAL_PLAN = ROOT / "research/results/encoder-v2/confirmation-plan.json"
POLICY = ROOT / "research/results/encoder-v2/final-policy.json"
PROTOCOL = ROOT / "research/external-clinc-protocol.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def original_files():
    """Relocate the published original lock without modifying its bytes."""
    plan = read_json(ORIGINAL_PLAN)
    if content_hash({k: v for k, v in plan.items() if k != "lock_hash"}) != plan["lock_hash"]:
        raise ValueError("Original confirmation lock was modified.")
    path_type = PureWindowsPath if PureWindowsPath(plan["run_dir"]).drive else PurePosixPath
    old_root = path_type(plan["run_dir"]).parents[1]
    files = {path_type(name).relative_to(old_root).as_posix(): digest
             for name, digest in plan["locked_files"].items()}
    for name, digest in files.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Original frozen file changed: {name}")
    return plan, files


def group_for(label):
    if label in MAPPING:
        return "mapped_intent"
    for group, labels in OOS_GROUPS.items():
        if label in labels:
            return group
    raise ValueError(f"No declared mapping/group for {label}")


def _rows(payload, split):
    rows = payload.get(split)
    if not isinstance(rows, list):
        raise ValueError(f"Missing CLINC split: {split}")
    for i, row in enumerate(rows):
        if (not isinstance(row, list) or len(row) != 2 or not all(isinstance(v, str) for v in row)
                or not normalize_query(row[0]) or not row[1]):
            raise ValueError(f"Malformed CLINC row {split}:{i}")
    return rows


def build_records(payload, bank_records, expected_per_class=30):
    """Select by predeclared source labels, never by query text or predictions."""
    bank_keys = {normalize_query(r["query"]) for r in bank_records}
    development_keys = {normalize_query(q) for split in ("train", "val") for q, _ in _rows(payload, split)}
    selected = []
    for index, (query, label) in enumerate(_rows(payload, "test")):
        if label in CLASSES:
            selected.append({"source_id": f"clinc:test:{index}", "source_split": "test",
                             "source_label": label, "query": query, "group": group_for(label),
                             "label": MAPPING.get(label)})
    counts = Counter(r["source_label"] for r in selected)
    if dict(counts) != {label: expected_per_class for label in CLASSES}:
        raise ValueError(f"Unexpected pre-exclusion class counts: {dict(counts)}")
    # Label conflicts fail preparation even when a conflicting row would be excluded.
    seen_labels = {}
    for r in selected:
        key = normalize_query(r["query"])
        if key in seen_labels and seen_labels[key] != r["source_label"]:
            raise ValueError("Conflicting source labels for duplicate selected text.")
        seen_labels[key] = r["source_label"]
    retained, exclusions, seen = [], [], set()
    for r in selected:
        key = normalize_query(r["query"])
        reason = "matches_original_banking77" if key in bank_keys else "duplicate_selected_test" if key in seen else None
        seen.add(key)
        if reason:
            exclusions.append({"source_id": r["source_id"], "source_label": r["source_label"], "reason": reason})
        else:
            retained.append(r)
    if set(r["source_label"] for r in retained) != set(CLASSES):
        raise ValueError("A predeclared class became empty after deduplication.")
    validate_records(retained)
    return retained, {"candidates": len(selected), "retained": len(retained), "exclusions": exclusions,
        "candidate_counts": dict(counts), "retained_counts": dict(Counter(r["source_label"] for r in retained)),
        "selected_test_also_in_clinc_train_val": sum(normalize_query(r["query"]) in development_keys for r in retained)}


def validate_records(records):
    if not records:
        raise ValueError("Empty external evaluation.")
    ids, queries = set(), set()
    for r in records:
        label = r["source_label"]
        key = normalize_query(r["query"])
        if (r["source_split"] != "test" or not r["source_id"].startswith("clinc:test:")
                or r["source_id"] in ids or not key or key in queries
                or r["group"] != group_for(label) or r["label"] != MAPPING.get(label)):
            raise ValueError("Invalid external row identity, label, group, or duplicate.")
        ids.add(r["source_id"])
        queries.add(key)


def near_duplicate_audit(records, bank_records):
    """Token-set Jaccard diagnostic only; it never removes or relabels a row."""
    tokenize = lambda text: set(re.findall(r"\w+", normalize_query(text)))
    bank_tokens = [tokenize(r["query"]) for r in bank_records]
    inverted = defaultdict(set)
    for i, tokens in enumerate(bank_tokens):
        for token in tokens:
            inverted[token].add(i)
    matches = []
    for r in records:
        tokens = tokenize(r["query"])
        candidates = set().union(*(inverted[t] for t in tokens)) if tokens else set()
        best = max(((len(tokens & bank_tokens[i]) / len(tokens | bank_tokens[i]), i)
                    for i in candidates), default=(0.0, -1))
        if best[0] >= 0.8:
            matches.append({"source_id": r["source_id"], "banking77_source_id": bank_records[best[1]]["source_id"],
                            "token_jaccard": best[0]})
    return {"method": "NFKC/casefold, Unicode word token sets, Jaccard >= 0.8; diagnostic only",
            "flagged_count": len(matches), "matches": matches,
            "limitation": "Not an exhaustive semantic duplication or pretraining-contamination audit."}


def download_sources():
    directory = DATA_DIR / "source"
    directory.mkdir(parents=True, exist_ok=True)
    paths = ["data/data_full.json", "data/domains.json", "README.md", "LICENSE"]
    tree = json.loads(subprocess.check_output([
        "gh", "api", f"repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1"], timeout=90))
    if tree.get("truncated"):
        raise ValueError("Source Git tree is incomplete.")
    entries = {item["path"]: item for item in tree["tree"]}
    provenance = {}
    for remote_path in paths:
        path = directory / Path(remote_path).name
        metadata = entries[remote_path]
        endpoint = f"repos/{REPOSITORY}/git/blobs/{metadata['sha']}"
        if not path.exists():
            payload = subprocess.check_output(["gh", "api", endpoint, "-H", "Accept: application/vnd.github.raw+json"], timeout=90)
            path.write_bytes(payload)
        payload = path.read_bytes()
        blob_sha = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if blob_sha != metadata["sha"]:
            raise ValueError(f"Pinned Git blob mismatch for {remote_path}")
        provenance[remote_path] = {"sha256": sha(path), "git_blob_sha1": blob_sha, "bytes": len(payload),
            "url": f"https://github.com/{REPOSITORY}/blob/{REVISION}/{remote_path}"}
    return provenance


def prepare():
    lock_path = PUBLIC_DIR / "plan.json"
    if lock_path.exists() or (DATA_DIR / "records.json").exists():
        raise FileExistsError("External v1 is immutable; do not overwrite a prepared evaluation.")
    original, files = original_files()
    policy = read_json(POLICY)
    if policy != original["selective_policies"]:
        raise ValueError("Published policy differs from the original frozen policies.")
    source = download_sources()
    bank = []
    for split in ("train", "test"):
        path = ROOT / f"assets/banking77/source/{split}.csv"
        if sha(path) != SOURCE_SHA256[f"{split}.csv"]:
            raise ValueError("Original BANKING77 source bytes changed.")
        files[path.relative_to(ROOT).as_posix()] = sha(path)
        bank.extend(read_source(path, split))
    records, audit = build_records(read_json(DATA_DIR / "source/data_full.json"), bank)
    audit["near_duplicate_diagnostic"] = near_duplicate_audit(records, bank)
    record_path = DATA_DIR / "records.json"
    write_json(record_path, records)
    for path in [record_path, PROTOCOL, POLICY, ORIGINAL_PLAN, Path(__file__),
                 ROOT / "tests/test_external_clinc.py", *(DATA_DIR / "source").iterdir()]:
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    plan = {"kind": "clinc150_narrow_external_stress_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": REPOSITORY, "source_revision": REVISION, "source_files": source,
        "source_license": "CC BY 3.0 Unported", "source_split": "test", "mapping": MAPPING,
        "unsupported_groups": OOS_GROUPS, "supported_target_labels": 1, "total_target_labels": 20,
        "records_file": record_path.relative_to(ROOT).as_posix(), "records_hash": content_hash(records),
        "audit": audit, "original_confirmation_lock": original["lock_hash"],
        "run_id": "external-clinc-v1", "model_run_id": original["run_id"], "head_hash": original["head_hash"],
        "checkpoint_dir": "runs/encoder-flower-balanced-v2/selected", "asset_dir": "assets/reference-mpnet",
        "policies": policy["policies"], "scoring": "Original offline CPU FP32 20-way argmax; no candidate restriction",
        "locked_files": files, "interpretation": "One-class transfer and two unsupported-topic stress groups, not 20-intent accuracy."}
    plan["lock_hash"] = content_hash(plan)
    write_json(lock_path, plan)
    return {"lock_hash": plan["lock_hash"], "audit": audit}


def verify_plan(plan, root=ROOT):
    if content_hash({k: v for k, v in plan.items() if k != "lock_hash"}) != plan["lock_hash"]:
        raise ValueError("External evaluation lock was modified.")
    root = Path(root).resolve()
    for name, digest in plan["locked_files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError(f"External frozen file changed or escaped workspace: {name}")


def wilson(correct, n):
    if n == 0:
        return None
    z = 1.959963984540054
    p, divisor = correct / n, 1 + z*z/n
    centre = (p + z*z/(2*n)) / divisor
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / divisor
    return [centre-radius, centre+radius]


def summarize(records, scores, policies):
    import numpy as np
    validate_records(records)
    scores = np.asarray(scores)
    if scores.shape != (len(records), len(INTENTS)) or not np.isfinite(scores).all():
        raise ValueError("Expected finite scores for every row and all 20 labels.")
    predictions = scores.argmax(axis=1)
    ordered = np.sort(scores, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    mapped = np.array([r["group"] == "mapped_intent" for r in records])
    truth = np.array([INTENTS.index(r["label"]) if r["label"] else -1 for r in records])
    correct = (predictions == truth) & mapped
    n, k = int(mapped.sum()), int(correct.sum())
    if not n:
        raise ValueError("Mapped class is missing.")
    result = {"mapped_intent": {"count": n, "correct": k, "accuracy": k/n,
        "nominal_wilson_95": wilson(k, n), "scope": "One mapped class, predicted among all twenty labels"},
        "policies": []}
    for policy in policies:
        threshold = policy["threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid fixed margin threshold.")
        accepted = margins >= threshold
        answered = int((accepted & mapped).sum())
        correct_answered = int((accepted & correct).sum())
        entry = {"threshold": threshold, "target_validation_accuracy": policy["target_validation_accuracy"],
            "mapped_intent": {"count": n, "answered": answered, "correct_answered": correct_answered,
                "wrong_answered": answered-correct_answered, "declined": n-answered, "coverage": answered/n,
                "accuracy_on_answered": correct_answered/answered if answered else None}, "unsupported_groups": {}}
        for group, labels in OOS_GROUPS.items():
            mask = np.array([r["group"] == group for r in records])
            count, false_accepts = int(mask.sum()), int((mask & accepted).sum())
            if not count:
                raise ValueError("An unsupported stress group is missing.")
            entry["unsupported_groups"][group] = {"count": count, "false_accepts": false_accepts,
                "false_accept_rate": false_accepts/count, "declined": count-false_accepts,
                "decline_rate": 1-false_accepts/count,
                "per_class": {label: {"count": sum(r["source_label"] == label for r in records),
                    "false_accepts": sum(bool(a) and r["source_label"] == label for r, a in zip(records, accepted))}
                    for label in labels}}
        result["policies"].append(entry)
    return result


def evaluate():
    plan = read_json(PUBLIC_DIR / "plan.json")
    verify_plan(plan)
    result_path = RUN_DIR / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        if result["lock_hash"] != plan["lock_hash"]:
            raise ValueError("Result belongs to another lock.")
        return result
    # Require a committed protocol before the first inference, and record its SHA.
    command = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    relative_plan = (PUBLIC_DIR / "plan.json").relative_to(ROOT).as_posix()
    tracked_plan = subprocess.check_output([*command, "show", f"HEAD:{relative_plan}"], cwd=ROOT)
    if json.loads(tracked_plan) != plan:
        raise ValueError("Commit the exact prepared plan before evaluating.")
    commit = subprocess.check_output([*command, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    started_path = RUN_DIR / "evaluation_started.json"
    if started_path.exists():
        raise RuntimeError("An incomplete evaluation already started; investigate before any retry.")
    records = read_json(ROOT / plan["records_file"])
    if content_hash(records) != plan["records_hash"]:
        raise ValueError("External records differ from the prepared manifest.")
    validate_records(records)
    from fedlora_edge.encoder_inference import EncoderPredictor
    import numpy as np
    import torch
    import transformers
    started = time.perf_counter()
    predictor = EncoderPredictor(ROOT / plan["asset_dir"], ROOT / plan["checkpoint_dir"])
    if predictor.saved["b_hash"] != plan["head_hash"]:
        raise ValueError("Loaded model differs from the external lock.")
    verify_plan(plan)
    write_json(started_path, {"lock_hash": plan["lock_hash"], "code_commit": commit,
                             "started_utc": datetime.now(timezone.utc).isoformat()})
    scores, truncated = predictor.scores([r["query"] for r in records])
    result = {"status": "completed", "run_id": plan["run_id"], "model_run_id": plan["model_run_id"],
        "head_hash": plan["head_hash"], "lock_hash": plan["lock_hash"], "records_hash": plan["records_hash"],
        "pre_inference_commit": commit, "metrics": summarize(records, scores, plan["policies"]),
        "truncated": truncated, "wall_seconds_including_model_loading": time.perf_counter()-started,
        "environment": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
                        "device": "cpu", "dtype": "float32", "threads": 2},
        "interpretation": plan["interpretation"],
        "pretraining_exposure": "Independent collection from BANKING77; encoder pretraining exposure has not been ruled out."}
    verify_plan(plan)
    np.save(RUN_DIR / "scores.npy", scores, allow_pickle=False)
    write_json(RUN_DIR / "predictions.json", [dict(r, prediction=INTENTS[int(s.argmax())]) for r, s in zip(records, scores)])
    write_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "evaluate"])
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == "prepare" else evaluate(), indent=2))


if __name__ == "__main__":
    main()
