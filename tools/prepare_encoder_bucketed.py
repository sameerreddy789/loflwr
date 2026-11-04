"""Opt-in bucketed MPNet preparation with separate metadata and feature identities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fedlora_edge.data import write_json
from fedlora_edge.encoder_bucketing import assemble, build_plan, prepare
from fedlora_edge.encoder_data import load_encoder_manifest


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_output(path):
    """Keep generated data below ignored project roots, never overwrite an output."""
    path = Path(path)
    resolved = path.resolve()
    if not any(resolved.is_relative_to((ROOT / name).resolve())
               and resolved != (ROOT / name).resolve() for name in ("assets", "runs")):
        raise ValueError("Output must be inside this project's ignored assets/ or runs/ directory.")
    if path.exists() or path.is_symlink():
        raise FileExistsError("Output already exists; preparation identities are immutable.")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    plan = subs.add_parser("plan")
    plan.add_argument("--source-dataset", required=True)
    plan.add_argument("--layout", choices=("five", "three"), default="five")
    plan.add_argument("--output", type=Path, required=True)
    participant = subs.add_parser("prepare")
    participant.add_argument("--plan", required=True)
    participant.add_argument("--participant", required=True)
    participant.add_argument("--records", nargs="+", required=True)
    participant.add_argument("--asset-dir", required=True)
    participant.add_argument("--output", type=Path, required=True)
    assembly = subs.add_parser("assemble")
    assembly.add_argument("--plan", required=True)
    assembly.add_argument("--registrations", nargs="+", required=True)
    assembly.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_output(args.output)
    if args.action == "plan":
        result = build_plan(load_encoder_manifest(args.source_dataset), args.layout)
        write_json(args.output, result)
        print(json.dumps({"plan_hash": result["plan_hash"], "recipe_hash": result["recipe_hash"]}))
    elif args.action == "prepare":
        result = prepare(_read_json(args.plan), args.records, args.output, args.asset_dir, args.participant)
        print(json.dumps(result))
    else:
        result = assemble(_read_json(args.plan), [_read_json(path) for path in args.registrations])
        write_json(args.output, result)
        print(json.dumps({"manifest_hash": result["manifest_hash"], "clients": list(result["clients"])}))


if __name__ == "__main__":
    main()
