"""Run the separate frozen-MPNet/head variant through actual Flower messages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.resume:
        config["resume"] = True
    from fedlora_edge.encoder_flower import simulate
    simulate(config)
    print(Path(config["run_dir"]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
