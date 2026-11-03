"""Fresh cached-model inference in a Linux network namespace with loopback only."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=Path)
    mode.add_argument("--deployment-config", type=Path)
    parser.add_argument("--port", type=int, default=19692)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path to preserve earlier evidence")
    interfaces = [name for _, name in socket.if_nameindex()]
    if interfaces != ["lo"]:
        parser.error("Run under unshare --user --map-root-user --net with loopback enabled")
    routes = Path("/proc/net/route").read_text().splitlines()[1:]
    if routes:
        parser.error("The isolated process unexpectedly has IPv4 routes")
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=0.5):
            raise RuntimeError("Unexpected external connectivity")
    except OSError as exc:
        network_error = {"type": type(exc).__name__, "errno": exc.errno, "message": str(exc)}
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false"})
    if args.deployment_config:
        from fedlora_edge.federation import deploy_local
        config = json.loads(args.deployment_config.read_text())
        started = time.perf_counter()
        deploy_local(config, project_dir=str(ROOT), port=args.port, certificate_dir=None)
        final = json.loads((Path(config["run_dir"]) / "final_results.json").read_text())
        passed = (final["attempts"] == config["attempts"]
                  and all(row["status"] == "committed" for row in final["rounds"]))
        report = {"run_id": args.run_id, "kind": "fresh_offline_deployment_check", "passed": passed,
                  "interfaces": interfaces, "ipv4_routes": routes,
                  "external_connection_error": network_error,
                  "execution_started_after_isolation": True,
                  "seconds": time.perf_counter() - started,
                  "training_run_id": config["run_id"],
                  "attempts": final["attempts"], "committed_versions": final["committed_versions"],
                  "scope": "Fresh TLS/authenticated local Flower deployment in a loopback-only namespace; one physical laptop."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not passed:
            raise RuntimeError("Offline deployment finished without all expected commits")
        return
    # Import and instantiate the model only after proving this fresh process has
    # no external network interface. No dependency installation is attempted.
    from fedlora_edge.training import LocalTrainer, TrainingConfig, load_b_state
    started = time.perf_counter()
    trainer = LocalTrainer(ROOT / "assets/model", ROOT / "assets/adapter", "cpu",
                           TrainingConfig(cpu_threads=4))
    trainer.set_b_state(load_b_state(args.checkpoint, trainer.identity))
    result = trainer.score("How do I activate my new card?")
    report = {"run_id": args.run_id, "kind": "fresh_offline_inference_check", "passed": True,
              "process_id": os.getpid(), "interfaces": interfaces, "ipv4_routes": routes,
              "external_connection_error": network_error, "model_loaded_after_isolation": True,
              "b_hash": result["b_hash"], "contract_hash": result["contract_hash"],
              "model_revision": trainer.identity["model_revision"], "intent": result["intent"],
              "seconds": time.perf_counter() - started,
              "scope": "Fresh CPU inference, Linux network namespace. No physical LAN or deployment claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
