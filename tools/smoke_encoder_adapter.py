"""Run an isolated real-model MPNet B-plus-head optimization/frozen-state smoke.

Tiny training-batch loss is an optimization check, never held-out accuracy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedlora_edge.data import content_hash, load_manifest, write_json  # noqa: E402
from fedlora_edge.encoder_adapter import (  # noqa: E402
    EncoderAdapterConfig, EncoderAdapterTrainer, environment_versions,
    prepare_encoder_adapter, save_encoder_state,
)
from fedlora_edge.state import hash_state  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=Path("assets/reference-mpnet"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("assets/encoder-adapter"))
    parser.add_argument("--records", type=Path, default=Path("assets/banking77/manifests/balanced/client_0.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Choose a new output directory; smoke evidence is immutable.")
    if args.steps < 1:
        parser.error("Steps must be positive.")
    args.output_dir.mkdir(parents=True)
    config = EncoderAdapterConfig(cpu_threads=args.cpu_threads, microbatch_size=args.microbatch_size,
                                  max_length=args.max_length, eval_batch_size=args.microbatch_size)
    started = time.perf_counter()
    identity = prepare_encoder_adapter(args.asset_dir, args.adapter_dir)
    trainer = EncoderAdapterTrainer(args.asset_dir, args.adapter_dir, device=args.device, config=config)
    # Only original training records may enter this optimization probe.
    payload = json.loads(args.records.read_text(encoding="utf-8"))
    if "manifest_hash" in payload:
        all_records = load_manifest(args.records)
    else:
        all_records = payload["records"]
        if content_hash(all_records) != payload.get("records_hash"):
            raise ValueError("Encoder record manifest hash mismatch.")
    if any(row.get("source_split") != "train" for row in all_records):
        raise ValueError("Smoke optimization must use original-training records.")
    records = all_records[:4]
    if not records:
        raise ValueError("Training fixture manifest is empty.")
    plan = {"run_id": args.output_dir.name, "kind": "encoder_b_plus_head_optimization_smoke",
            "created_utc": datetime.now(timezone.utc).isoformat(), "device": args.device,
            "config": asdict(config), "steps": args.steps, "identity": identity,
            "records_hash": content_hash(records), "environment": environment_versions(),
            "source_files_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__).resolve(), ROOT / "fedlora_edge/encoder_adapter.py")}}
    plan["config_hash"] = content_hash(plan)
    write_json(args.output_dir / "plan.json", plan)
    initial = trainer.get_state()
    frozen_before = trainer.frozen_hashes()
    before = trainer.evaluate(records)
    training = trainer.train(records, args.steps, 42)
    after = trainer.evaluate(records)
    current = trainer.get_state()
    frozen_after = trainer.frozen_hashes()
    checks = {"base_unchanged": frozen_before["base_hash"] == frozen_after["base_hash"],
              "a_unchanged": frozen_before["common_a_hash"] == frozen_after["common_a_hash"],
              "b_changed": any(initial[name] != current[name] for name in initial if ".lora_B." in name),
              "head_changed": any(initial[name] != current[name] for name in initial if name.startswith("head.")),
              "nonzero_b_gradient": training["telemetry"]["nonzero_b_gradient"],
              "all_steps_completed": training["status"] == "completed"}
    save_encoder_state(args.output_dir / "trained_state.json", current, identity, run_id=args.output_dir.name)
    summary = {"plan": plan, "checks": checks, "passed": all(checks.values()),
               "initial_state_hash": hash_state(initial), "final_state_hash": hash_state(current),
               "frozen_before": frozen_before, "frozen_after": frozen_after,
               "training": training, "training_batch_before": before, "training_batch_after": after,
               "total_seconds": time.perf_counter() - started,
               "interpretation": "Original-training tiny-batch optimization smoke, not held-out accuracy or a federated experiment."}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"checks": checks, "before_loss": before["task_loss"], "after_loss": after["task_loss"],
                      "training_seconds": training["telemetry"]["local_train_seconds"],
                      "trainable_parameters": identity["trainable_parameters"]}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
