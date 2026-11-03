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
    if "identity" not in config:
        from dataclasses import asdict
        from fedlora_edge.encoder_data import load_encoder_manifest
        from fedlora_edge.encoder_head import HeadConfig, HeadTrainer
        from tools.run_encoder_study import make_identity
        manifest = load_encoder_manifest(config["dataset_dir"])
        head_config = HeadConfig(**config.get("head_config", {}))
        trainer = HeadTrainer(manifest["feature_dim"], len(manifest["labels"]),
                              config.get("seed", 42), head_config)
        config["head_config"] = asdict(head_config)
        config["identity"] = make_identity(manifest, trainer.get_state(), asdict(head_config))
    if args.resume:
        config["resume"] = True
    from fedlora_edge.encoder_flower import simulate
    simulate(config)
    print(Path(config["run_dir"]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
