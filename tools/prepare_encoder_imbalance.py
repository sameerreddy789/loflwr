"""Reuse verified local features for the original 400-query imbalance scenario.

Only training and validation files are opened. The derived dataset has no
confirmation entry and must not replace the sealed expanded-data experiment.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedlora_edge.data import INTENTS, content_hash, load_manifest, write_json
from fedlora_edge.encoder_data import load_encoder_manifest, load_encoder_records, load_feature_split, sha256


def prepare(source, original_manifests, output):
    import numpy as np
    source, original_manifests, output = map(Path, (source, original_manifests, output))
    if output.exists():
        raise FileExistsError("Use a fresh dataset directory.")
    manifest = load_encoder_manifest(source)
    splits = {}
    for client in range(5):
        x, y, ids = load_feature_split(source, client)
        records = {r["source_id"]: r for r in load_encoder_records(source, client)}
        index = {str(key): i for i, key in enumerate(ids)}
        rows = load_manifest(original_manifests / "imbalanced" / f"client_{client}.json")
        if len(rows) != (360 if client == 1 else 10):
            raise ValueError("Original imbalance counts differ from the declared scenario.")
        positions = []
        for row in rows:
            key = row["source_id"]
            if key not in index or key not in records:
                raise ValueError(f"Original training example is not in the verified feature cache: {key}")
            cached = records[key]
            if (row["label"] != cached["label"] or row["query"] != cached["query"]
                    or INTENTS[int(y[index[key]])] != row["label"]):
                raise ValueError("Original row and cached feature label/query disagree.")
            positions.append(index[key])
        splits[f"clients/client_{client}"] = (rows, x[positions], y[positions], ids[positions])
    xv, yv, iv = load_feature_split(source)
    splits["validation"] = (load_encoder_records(source), xv, yv, iv)
    output.mkdir(parents=True)
    (output / "clients").mkdir()
    derived = {k: v for k, v in manifest.items() if k not in ("clients", "validation", "confirmation", "manifest_hash")}
    derived.update(clients={}, parent_manifest_hash=manifest["manifest_hash"],
                   scenario="original_400_query_90_percent_imbalance_validation_only")
    for stem, (rows, x, y, ids) in splits.items():
        np.savez(output / (stem + ".npz"), X=x, y=y, source_ids=ids)
        write_json(output / (stem + ".json"), {"records": rows, "records_hash": content_hash(rows)})
        entry = {"file": stem + ".npz", "sha256": sha256(output / (stem + ".npz")),
                 "records_file": stem + ".json", "records_hash": content_hash(rows), "count": len(rows),
                 "label_counts": dict(Counter(r["label"] for r in rows))}
        if stem == "validation":
            derived["validation"] = entry
        else:
            derived["clients"][stem.rsplit("_", 1)[1]] = entry
    derived["manifest_hash"] = content_hash(derived)
    write_json(output / "manifest.json", derived)
    for client in range(5):
        load_feature_split(output, client)
    load_feature_split(output)
    return derived


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="assets/encoder-bank20-v1")
    parser.add_argument("--original-manifests", default="assets/banking77/manifests")
    parser.add_argument("--output", default="assets/encoder-bank20-imbalance-v1")
    args = parser.parse_args()
    result = prepare(args.source, args.original_manifests, args.output)
    print(json.dumps({"manifest_hash": result["manifest_hash"], "counts": [v["count"] for v in result["clients"].values()]}))
