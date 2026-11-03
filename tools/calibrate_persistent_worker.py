"""Real-model cold/warm equivalence, interruption, deadline and cost calibration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedlora_edge.checkpoint import EventLog, atomic_json
from fedlora_edge.data import load_manifest
from fedlora_edge.flower_app import _run_training_unlocked
from fedlora_edge.persistent_worker import stop_worker
from fedlora_edge.protocol import RoundRequest
from fedlora_edge.state import hash_state, state_from_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="persistent-worker-calibration-20261102")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    envelope = json.loads((root / "assets/adapter/initial_b.json").read_text())
    initial = state_from_dict(envelope["state"])
    identity = envelope["identity"]
    config = {"run_id": args.run_id, "run_dir": str(output),
              "asset_dir": str(root / "assets/model"), "adapter_dir": str(root / "assets/adapter"),
              "device": args.device, "training_config": {}, "runtime": "flower_ray_one_gpu_worker"}
    request = RoundRequest(args.run_id, 1, 0, hash_state(initial), identity["contract_hash"],
                           identity["common_a_hash"], identity["model_revision"], "0", args.steps,
                           time.time() + 600, 42)
    records = {str(c): load_manifest(root / f"assets/banking77/manifests/balanced/client_{c}.json")
               for c in (0, 1)}
    report = {"kind": "real_model_worker_calibration", "run_id": args.run_id,
              "runtime": "isolated subprocess calibration", "identity": identity,
              "initial_b_hash": hash_state(initial), "jobs": {}}
    log = EventLog(output, args.run_id)

    def job(name, req, persistent, extra=None):
        current = {**config, "client_job_dir": str(output / name), "persistent_worker": persistent,
                   **(extra or {})}
        started = time.monotonic()
        result = _run_training_unlocked(req, initial, current, records[req.client_id], log)
        report["jobs"][name] = {"status": result["status"], "telemetry": result["telemetry"],
                                "total_seconds": time.monotonic() - started,
                                "delta_hash": hash_state(result["delta"]) if result["delta"] else None}
        atomic_json(output / "results.json", report)
        print(json.dumps({"job": name, **report["jobs"][name]}), flush=True)
        return result

    try:
        cold = job("cold", replace(request, deadline_unix=time.time() + 600), False)
        warm_first = job("warm_first", replace(request, deadline_unix=time.time() + 600), True)
        warm_repeat = job("warm_repeat", replace(request, deadline_unix=time.time() + 600), True)
        job("interleaved_client", replace(request, client_id="1", deadline_unix=time.time() + 600), True)
        after_other = job("after_other_client", replace(request, deadline_unix=time.time() + 600), True)
        expected = hash_state(cold["delta"])
        assert cold["status"] == "completed"
        assert all(result["status"] == "completed" and hash_state(result["delta"]) == expected
                   for result in (warm_first, warm_repeat, after_other)), "Cold/warm B deltas differ"
        report["cold_warm_bitwise_equivalent"] = True
        killed = job("interrupted", replace(request, attempt_id=2, client_id="1", optimizer_steps=8,
                                             deadline_unix=time.time() + 600), True,
                     {"process_interruptions": [{"attempt_id": 2, "client_id": "1", "after_steps": 1}]})
        assert killed["status"] == "process_interrupted" and not killed["delta"]
        replacement = job("replacement", replace(request, deadline_unix=time.time() + 600), True)
        assert replacement["status"] == "completed" and hash_state(replacement["delta"]) == expected
        assert replacement["telemetry"]["job_pid"] != killed["telemetry"]["job_pid"]
        report["kill_replacement_bitwise_equivalent"] = True
        expired = job("deadline", replace(request, attempt_id=3, optimizer_steps=100,
                                           deadline_unix=time.time() + 0.75), True)
        assert expired["status"] == "deadline_expired" and not expired["delta"]
        assert report["jobs"]["deadline"]["total_seconds"] < 5
        report["deadline_bounded"] = True
        report["passed"] = True
    finally:
        stop_worker()
        atomic_json(output / "results.json", report)
    print(json.dumps({"passed": report.get("passed", False), "result": str(output / "results.json")}))


if __name__ == "__main__":
    main()
