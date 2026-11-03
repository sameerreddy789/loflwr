"""Partition the frozen public MPNet reference into local training feature files."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedlora_edge.encoder_data import prepare_encoder_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=Path("runs/accuracy-feasibility-20261103"))
    parser.add_argument("--data-dir", type=Path, default=Path("assets/banking77"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets/encoder-bank20-v1"))
    args = parser.parse_args()
    manifest = prepare_encoder_dataset(args.reference_dir, args.data_dir, args.output_dir)
    print(json.dumps({"manifest_hash": manifest["manifest_hash"], "clients": {k: v["count"] for k, v in manifest["clients"].items()},
                      "validation": manifest["validation"]["count"], "confirmation_unscored": manifest["confirmation"]["count"]}, indent=2))
