"""An isolated training process with the same LocalTrainer used everywhere.

The Flower handler supervises this process so a hung/expired optimizer job cannot
retain a simulation GPU or a deployment client's memory indefinitely.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from queue import Empty, Queue

from .checkpoint import atomic_json, canonical_hash
from .protocol import RoundRequest, validate_request
from .state import state_from_dict, state_to_dict


def worker_profile(config: dict, request: RoundRequest) -> str:
    return canonical_hash({key: config.get(key) for key in
                           ("asset_dir", "adapter_dir", "device", "training_config")} | {
        "run_id": request.run_id, "contract_hash": request.contract_hash,
        "common_a_hash": request.common_a_hash, "model_revision": request.model_revision,
    })


def run_job(input_path: str, output_path: str, progress_path: str, trainer=None):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from .training import LocalTrainer, TrainingConfig

    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    request = RoundRequest.from_dict(payload["request"])
    incoming = state_from_dict(payload["state"])
    config = payload["runtime"]
    reused = trainer is not None
    atomic_json(progress_path, {"pid": os.getpid(), "status": "loading",
                                "optimizer_steps": 0, "timestamp": time.time(),
                                "worker_reused": reused})
    load_started = time.monotonic()
    if trainer is None:
        trainer = LocalTrainer(asset_dir=config["asset_dir"], adapter_dir=config["adapter_dir"],
                               device=config.get("device", "cpu"),
                               config=TrainingConfig(**config.get("training_config", {})))
    load_seconds = 0.0 if reused else time.monotonic() - load_started
    validate_request(request, incoming, trainer.identity)
    atomic_json(progress_path, {"pid": os.getpid(), "status": "training",
                                "optimizer_steps": 0, "timestamp": time.time()})

    def progress(step: int, telemetry: dict) -> None:
        atomic_json(progress_path, {"pid": os.getpid(), "status": "training",
                                    "optimizer_steps": step, "telemetry": telemetry,
                                    "timestamp": time.time()})

    result = trainer.train(
        payload["records"], incoming, optimizer_steps=request.optimizer_steps,
        run_seed=request.seed, attempt_id=request.attempt_id, client_id=request.client_id,
        deadline_unix=request.deadline_unix, progress_callback=progress,
    )
    # Release activation/optimizer allocator blocks between jobs while keeping
    # the shared frozen model resident. train() resets B, seed and AdamW anew.
    import gc
    import torch

    gc.collect()
    if trainer.device.type == "cuda":
        torch.cuda.empty_cache()
    result["telemetry"].update(worker_reused=reused, model_load_seconds=load_seconds)
    atomic_json(output_path, {"status": result["status"], "delta": state_to_dict(result["delta"]),
                              "telemetry": result["telemetry"]})
    return trainer


def serve(parent_pid: int, idle_timeout: float) -> None:
    """Sequential stdin jobs, one model, and bounded lifetime when the actor exits."""
    if parent_pid <= 0 or idle_timeout <= 0:
        raise ValueError("A parent PID and positive idle timeout are required")

    def watch_parent() -> None:
        while True:
            if os.getppid() != parent_pid:
                os._exit(75)
            time.sleep(0.25)

    threading.Thread(target=watch_parent, name="trainer-parent-watchdog", daemon=True).start()
    incoming: Queue = Queue()

    def read_jobs() -> None:
        for line in sys.stdin:
            incoming.put(line)
        incoming.put(None)

    threading.Thread(target=read_jobs, name="trainer-command-reader", daemon=True).start()
    trainer = None
    profile = None
    while True:
        try:
            line = incoming.get(timeout=idle_timeout)
        except Empty:
            return
        if line is None:
            return
        command = json.loads(line)
        try:
            payload = json.loads(Path(command["input"]).read_text(encoding="utf-8"))
            request = RoundRequest.from_dict(payload["request"])
            current_profile = worker_profile(payload["runtime"], request)
            if profile is not None and profile != current_profile:
                raise ValueError("Persistent trainer profile changed")
            profile = current_profile
            trainer = run_job(command["input"], command["output"], command["progress"], trainer)
        except Exception as error:
            traceback.print_exc()
            atomic_json(command["output"], {"status": "failed", "delta": {},
                        "telemetry": {"optimizer_steps": 0,
                                      "worker_error": f"{type(error).__name__}: {error}"}})
            # Never reuse a trainer after an unknown failure state.
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--progress")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--idle-timeout", type=float, default=900)
    args = parser.parse_args()
    if args.serve:
        serve(args.parent_pid, args.idle_timeout)
    elif args.input and args.output and args.progress:
        run_job(args.input, args.output, args.progress)
    else:
        parser.error("Supply --serve or all three --input/--output/--progress paths")


if __name__ == "__main__":
    main()
