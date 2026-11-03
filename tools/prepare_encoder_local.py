"""Plan local feature preparation, prepare one participant, or assemble metadata."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fedlora_edge.data import write_json
from fedlora_edge.encoder_data import load_encoder_manifest
from fedlora_edge.encoder_local_data import build_local_plan, prepare_local_features, assemble_local_manifest


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    plan = subs.add_parser("plan")
    plan.add_argument("--source-dataset", required=True)
    plan.add_argument("--layout", choices=["five", "three"], default="five")
    plan.add_argument("--output", type=Path, required=True)
    prepare = subs.add_parser("prepare")
    prepare.add_argument("--plan", required=True)
    prepare.add_argument("--participant", required=True)
    prepare.add_argument("--records", nargs="+", required=True)
    prepare.add_argument("--asset-dir", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    assemble = subs.add_parser("assemble")
    assemble.add_argument("--plan", required=True)
    assemble.add_argument("--registrations", nargs="+", required=True)
    assemble.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; prepared identities are immutable.")
    if args.action == "plan":
        result = build_local_plan(load_encoder_manifest(args.source_dataset), args.layout)
        write_json(args.output, result)
        print(json.dumps({"plan_hash": result["plan_hash"], "layout": result["layout"]}))
    elif args.action == "prepare":
        result = prepare_local_features(read_json(args.plan), args.records, args.output, args.asset_dir, args.participant)
        print(json.dumps(result))
    else:
        result = assemble_local_manifest(read_json(args.plan), [read_json(p) for p in args.registrations])
        write_json(args.output, result)
        print(json.dumps({"manifest_hash": result["manifest_hash"], "clients": list(result["clients"])}))


if __name__ == "__main__":
    main()
