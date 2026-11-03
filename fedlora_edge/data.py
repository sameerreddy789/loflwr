"""Pinned, deterministic BANKING77 selection without a datasets dependency.

Original rows remain in ignored assets/. Manifests retain the original text,
split and 1-based data-row number; normalization is only for leakage checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

DATASET_REPOSITORY = "PolyAI-LDN/task-specific-datasets"
DATASET_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
NORMALIZATION_VERSION = "nfkc-casefold-whitespace-v1"
SELECTION_SEED = 20261020
DOMAINS = {
    0: ("Card lifecycle", ("card_arrival", "card_delivery_estimate", "activate_my_card", "card_about_to_expire")),
    1: ("Card payments", ("pending_card_payment", "declined_card_payment", "card_payment_not_recognised", "transaction_charged_twice")),
    2: ("Transfers", ("pending_transfer", "transfer_not_received_by_recipient", "transfer_fee_charged", "cancel_transfer")),
    3: ("Cash withdrawals", ("pending_cash_withdrawal", "declined_cash_withdrawal", "wrong_amount_of_cash_received", "cash_withdrawal_not_recognised")),
    4: ("Account verification", ("edit_personal_details", "verify_my_identity", "unable_to_verify_identity", "why_verify_identity")),
}
INTENTS = tuple(label for _, labels in DOMAINS.values() for label in labels)
LABEL_CLIENT = {label: client for client, (_, labels) in DOMAINS.items() for label in labels}
SOURCE_FILES = {"train.csv": "banking_data/train.csv", "test.csv": "banking_data/test.csv", "LICENSE": "LICENSE", "README.md": "README.md"}
# Filled from the original pinned bytes; every download is checked against these.
SOURCE_SHA256 = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "LICENSE": "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661",
    "README.md": "7b9ce9069931cc2de89dce8c7ffb53d07806cf413601ccf133b9af5227b3a373",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def normalize_query(query: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", query).casefold().split())


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def download_sources(data_dir: str | Path = "assets/banking77") -> dict[str, Any]:
    root = Path(data_dir) / "source"
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, repository_path in SOURCE_FILES.items():
        url = f"https://raw.githubusercontent.com/{DATASET_REPOSITORY}/{DATASET_REVISION}/{repository_path}"
        path = root / name
        if not path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "FedLoRA-Edge/0.1 (BANKING77 reproducibility)"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in SOURCE_SHA256 and sha != SOURCE_SHA256[name]:
            raise ValueError(f"Source checksum mismatch for {path}; remove the corrupt cache and retry.")
        files[name] = {"sha256": sha, "bytes": path.stat().st_size, "url": url}
    provenance = {"repository": DATASET_REPOSITORY, "revision": DATASET_REVISION, "files": files, "attribution": "BANKING77 by PolyAI; original repository LICENSE and README preserved."}
    write_json(root / "provenance.json", provenance)
    return provenance


def read_source(path: str | Path, split: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(Path(path).read_text(encoding="utf-8-sig")))
    if set(reader.fieldnames or ()) != {"text", "category"}:
        raise ValueError(f"Unexpected BANKING77 CSV schema in {path}: {reader.fieldnames}")
    return [{"query": row["text"], "label": row["category"], "source_split": split, "source_row": index, "source_id": f"{split}:{index}"} for index, row in enumerate(reader, 1)]


def _unique(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    found: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for record in records:
        normalized = normalize_query(record["query"])
        if not normalized:
            raise ValueError(f"Empty normalized query: {record['source_id']}")
        if normalized in found:
            if found[normalized]["label"] != record["label"]:
                raise ValueError(f"Conflicting labels for normalized query: {found[normalized]['source_id']} / {record['source_id']}")
            duplicates += 1
        else:
            found[normalized] = record
    return list(found.values()), duplicates


def build_selections(train: list[dict[str, Any]], test: list[dict[str, Any]], selection_seed: int = SELECTION_SEED) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    # Inspect all original labels before filtering to the twenty selected labels.
    _unique([*test, *train])
    test_unique, test_duplicates = _unique(test)
    train_unique, train_duplicates = _unique(train)
    all_test_keys = {normalize_query(record["query"]) for record in test_unique}
    eligible = [record for record in train_unique if normalize_query(record["query"]) not in all_test_keys]
    selections: dict[str, list[dict[str, Any]]] = {"validation": [], "test": []}
    for scenario in ("balanced", "imbalanced"):
        for client in DOMAINS:
            selections[f"{scenario}/client_{client}"] = []
    available = {}
    for label in INTENTS:
        client = LABEL_CLIENT[label]
        train_rows = [record for record in eligible if record["label"] == label]
        test_rows = [record for record in test_unique if record["label"] == label]
        seed = int(content_hash([selection_seed, label]), 16)
        random.Random(seed).shuffle(train_rows)
        random.Random(seed + 1).shuffle(test_rows)
        needed_train = 100 if client == 1 else 50
        if len(train_rows) < needed_train or len(test_rows) < 25:
            raise ValueError(f"Insufficient unique rows for {label}: train={len(train_rows)} (need {needed_train}), test={len(test_rows)} (need 25)")
        available[label] = {"eligible_train": len(train_rows), "unique_official_test": len(test_rows)}
        def decorate(row: dict[str, Any]) -> dict[str, Any]:
            return {**row, "client_id": client, "domain": DOMAINS[client][0]}
        selections["test"].extend(map(decorate, test_rows[:25]))
        selections["validation"].extend(map(decorate, train_rows[:10]))
        selections[f"balanced/client_{client}"].extend(map(decorate, train_rows[10:50]))
        minority_count = (3, 3, 2, 2)[DOMAINS[client][1].index(label)]
        count = 90 if client == 1 else minority_count
        selections[f"imbalanced/client_{client}"].extend(map(decorate, train_rows[10:10 + count]))
    verify_selections(selections)
    diagnostics = {"original_counts": {"train": len(train), "test": len(test)}, "excluded_duplicates": {"within_train": train_duplicates, "within_test": test_duplicates, "train_matching_official_test": len(train_unique) - len(eligible)}, "available_per_intent": available}
    return selections, diagnostics


def verify_selections(selections: dict[str, list[dict[str, Any]]]) -> None:
    expected = {"validation": 200, "test": 500, **{f"balanced/client_{client}": 160 for client in DOMAINS}, **{f"imbalanced/client_{client}": 360 if client == 1 else 10 for client in DOMAINS}}
    if set(selections) != set(expected):
        raise ValueError("Missing or extra selection manifests.")
    for name, records in selections.items():
        if len(records) != expected[name]:
            raise ValueError(f"Wrong count for {name}: {len(records)}")
        labels = Counter(record["label"] for record in records)
        if name in ("validation", "test"):
            expected_labels = {label: 10 if name == "validation" else 25 for label in INTENTS}
        else:
            client = int(name.rsplit("_", 1)[1])
            counts = (40,) * 4 if name.startswith("balanced/") else ((90,) * 4 if client == 1 else (3, 3, 2, 2))
            expected_labels = dict(zip(DOMAINS[client][1], counts))
        if dict(labels) != expected_labels:
            raise ValueError(f"Wrong per-intent allocation for {name}.")
        for record in records:
            if record["client_id"] != LABEL_CLIENT[record["label"]]:
                raise ValueError(f"Wrong client/domain ownership in {name}.")
            expected_split = "test" if name == "test" else "train"
            if record["source_split"] != expected_split:
                raise ValueError(f"Wrong original split in {name}.")
    for scenario in ("balanced", "imbalanced"):
        seen: set[str] = set()
        for name in ("validation", "test", *(f"{scenario}/client_{client}" for client in DOMAINS)):
            for record in selections[name]:
                key = normalize_query(record["query"])
                if key in seen:
                    raise ValueError(f"Duplicate or held-out leakage in scenario {scenario}, manifest {name}.")
                seen.add(key)


def prepare_manifests(data_dir: str | Path = "assets/banking77", selection_seed: int = SELECTION_SEED) -> dict[str, Any]:
    root = Path(data_dir)
    source = root / "source"
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    if provenance["revision"] != DATASET_REVISION or provenance["repository"] != DATASET_REPOSITORY or set(provenance["files"]) != set(SOURCE_FILES):
        raise ValueError("Unexpected source revision.")
    for name, metadata in provenance["files"].items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != metadata["sha256"] or (name in SOURCE_SHA256 and actual != SOURCE_SHA256[name]):
            raise ValueError(f"Source bytes have changed: {name}")
    selections, diagnostics = build_selections(read_source(source / "train.csv", "train"), read_source(source / "test.csv", "test"), selection_seed)
    common = {"schema_version": 1, "dataset_repository": DATASET_REPOSITORY, "dataset_revision": DATASET_REVISION, "source_hashes": {name: value["sha256"] for name, value in provenance["files"].items()}, "selection_seed": selection_seed, "normalization_version": NORMALIZATION_VERSION}
    index = {**common, **diagnostics, "manifests": {}}
    for name, records in selections.items():
        manifest = {**common, "name": name, "counts_per_intent": dict(sorted(Counter(record["label"] for record in records).items())), "records": records}
        manifest["manifest_hash"] = content_hash(manifest)
        write_json(root / "manifests" / (name + ".json"), manifest)
        index["manifests"][name] = {"path": name + ".json", "hash": manifest["manifest_hash"], "count": len(records)}
    index["manifest_set_hash"] = content_hash(index)
    write_json(root / "manifests" / "index.json", index)
    return index


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed_hash = manifest.pop("manifest_hash")
    if content_hash(manifest) != claimed_hash:
        raise ValueError(f"Manifest hash mismatch: {path}")
    if manifest["dataset_revision"] != DATASET_REVISION or manifest["normalization_version"] != NORMALIZATION_VERSION:
        raise ValueError("Unsupported manifest contract.")
    return manifest["records"]


def verify_manifests(data_dir: str | Path = "assets/banking77") -> dict[str, Any]:
    root = Path(data_dir) / "manifests"
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in index.items() if key != "manifest_set_hash"}
    if content_hash(unsigned) != index["manifest_set_hash"]:
        raise ValueError("Manifest index hash mismatch.")
    selections = {}
    for name, entry in index["manifests"].items():
        path = root / entry["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["manifest_hash"] != entry["hash"]:
            raise ValueError(f"Manifest is not the indexed revision: {name}")
        selections[name] = load_manifest(path)
    verify_selections(selections)
    return {"verified": True, "manifest_set_hash": index["manifest_set_hash"], "counts": {name: len(records) for name, records in selections.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "prepare", "verify"))
    parser.add_argument("--data-dir", default="assets/banking77")
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    args = parser.parse_args()
    if args.command == "download":
        result = download_sources(args.data_dir)
    elif args.command == "prepare":
        result = prepare_manifests(args.data_dir, args.selection_seed)
    else:
        result = verify_manifests(args.data_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
