"""Run the real Flower apps: Ray rehearsal or TLS/authenticated same-host runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from .checkpoint import EventLog, atomic_json


def _absolute(path: str) -> str:
    return str(Path(path).resolve())


def create_config(args: argparse.Namespace) -> dict:
    count = 5 if args.layout == "five" else 3
    groups = [[i] for i in range(5)] if count == 5 else [[0, 1], [2, 3], [4]]
    budgets = [int(x) for x in args.budgets.split(",")]
    if len(budgets) == 1:
        budgets *= count
    if len(budgets) != count or any(x <= 0 for x in budgets):
        raise ValueError(f"Specify one positive budget or exactly {count} budgets")
    interruptions = []
    for text in args.interrupt:
        attempt, client, step = text.split(":")
        interruptions.append({"attempt_id": int(attempt), "client_id": client, "after_steps": int(step)})
    return {
        "run_id": args.run_id, "run_dir": _absolute(args.run_dir or f"runs/{args.run_id}"),
        "asset_dir": _absolute(args.asset_dir), "adapter_dir": _absolute(args.adapter_dir),
        "client_manifests": {
            str(c): [_absolute(str(Path(args.data_dir) / "manifests" / args.scenario / f"client_{i}.json"))
                     for i in group] for c, group in enumerate(groups)
        },
        "budgets": {str(c): budget for c, budget in enumerate(budgets)},
        "attempts": args.attempts, "quorum": args.quorum or (3 if count == 5 else 2),
        "strategy": args.strategy, "seed": args.seed, "deadline_seconds": args.deadline_seconds,
        "device": args.device, "training_config": {},
        "runtime": "same_host_flower_deployment" if args.command == "deploy-local"
        else "flower_ray_one_gpu_worker",
        "process_interruptions": interruptions, "omitted_clients": args.omit,
        "resume": args.resume, "scenario": args.scenario,
        **({"validation_manifest": _absolute(args.validation_manifest),
            "evaluate_every": args.evaluate_every, "evaluate_initial": args.evaluate_initial,
            "evaluation_device": args.evaluation_device or args.device}
           if args.validation_manifest else {}),
    }


def _offline_environment(config: dict) -> dict[str, str]:
    return {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
            "VIRTUAL_ENV": sys.prefix,
            "TOKENIZERS_PARALLELISM": "false", "FLWR_TELEMETRY_ENABLED": "0",
            "FLWR_DISABLE_UPDATE_CHECK": "1", "FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION": "1",
            "FEDLORA_CONFIG": str(Path(config["run_dir"]) / "runtime.json")}


def initialize_ray(backend: dict) -> float:
    """Start infrastructure before any bounded ServerApp request is issued.

    Flower's Python entry point starts ServerApp concurrently with Ray startup;
    on a cold machine that would spend request TTL waiting for a head process.
    """
    import ray

    started = time.monotonic()
    if not ray.is_initialized():
        pythonpath = os.pathsep.join(sys.path)
        if os.environ.get("PYTHONPATH"):
            pythonpath += os.pathsep + os.environ["PYTHONPATH"]
        ray.init(runtime_env={"env_vars": {"PYTHONPATH": pythonpath}}, **backend["init_args"])
    return time.monotonic() - started


def simulate(config: dict) -> None:
    config.setdefault("persistent_worker", True)
    atomic_json(Path(config["run_dir"]) / "runtime.json", config)
    os.environ.update(_offline_environment(config))
    from flwr.simulation import run_simulation
    from .flower_app import make_client_app, make_server_app

    cuda = config.get("device", "cpu").startswith("cuda")
    cpus = int(config.get("worker_cpus", 2))
    # One actor owns all reserved CPU/GPU resources; all clients share its queue.
    backend = {"init_args": {"num_cpus": cpus, "num_gpus": int(cuda),
                             "include_dashboard": False, "log_to_driver": True},
               "client_resources": {"num_cpus": cpus, "num_gpus": int(cuda)}}
    startup_seconds = initialize_ray(backend)
    atomic_json(Path(config["run_dir"]) / "runtime_preparation.json", {
        "run_id": config["run_id"], "ray_startup_seconds": startup_seconds,
        "worker_cpus": cpus, "worker_gpus": int(cuda), "workers": 1,
    })
    run_simulation(server_app=make_server_app(config), client_app=make_client_app(config),
                   num_supernodes=len(config["client_manifests"]), backend_config=backend)


def generate_certificates(directory: Path, client_ids: list[str]) -> dict:
    """Create a local CA, localhost certificate and per-node P-384 SSH identities."""
    from datetime import datetime, timedelta, timezone
    from ipaddress import ip_address
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    directory.mkdir(parents=True, exist_ok=False)
    os.chmod(directory, 0o700)
    ca_key = ec.generate_private_key(ec.SECP384R1())
    server_key = ec.generate_private_key(ec.SECP384R1())
    now = datetime.now(timezone.utc)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "FedLoRA local rehearsal CA")])
    ca = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=30))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .sign(ca_key, hashes.SHA256()))
    server = (x509.CertificateBuilder()
              .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
              .issuer_name(name).public_key(server_key.public_key())
              .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=5))
              .not_valid_after(now + timedelta(days=30))
              .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"),
                                                          x509.IPAddress(ip_address("127.0.0.1"))]), critical=False)
              .sign(ca_key, hashes.SHA256()))
    for filename, key in (("ca.key", ca_key), ("server.key", server_key)):
        path = directory / filename
        path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                           serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
        os.chmod(path, 0o600)
    (directory / "ca.crt").write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    (directory / "server.crt").write_bytes(server.public_bytes(serialization.Encoding.PEM))
    for client_id in client_ids:
        key = ec.generate_private_key(ec.SECP384R1())
        path = directory / f"node_{client_id}.key"
        path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                           serialization.PrivateFormat.OpenSSH,
                                           serialization.NoEncryption()))
        os.chmod(path, 0o600)
        (directory / f"node_{client_id}.pub").write_bytes(
            key.public_key().public_bytes(serialization.Encoding.OpenSSH,
                                           serialization.PublicFormat.OpenSSH))
    return {"directory": str(directory), "ca": str(directory / "ca.crt"),
            "certificate": str(directory / "server.crt"), "key": str(directory / "server.key")}


def _binary(name: str) -> str:
    suffix = ".exe" if os.name == "nt" else ""
    return str(Path(sys.executable).parent / (name + suffix))


def _stop_tree(process: subprocess.Popen) -> None:
    """Stop only the process tree created for this runtime invocation."""
    import psutil

    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        parent.terminate()
    except psutil.NoSuchProcess:
        pass
    _, alive = psutil.wait_procs([parent, *children], timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def deploy_local(config: dict, *, project_dir: str, port: int, certificate_dir: str | None) -> None:
    """Real TLS Flower HTTP transport on this computer; not a three-laptop claim."""
    run_dir = Path(config["run_dir"])
    runtime_dir = run_dir / "flower_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    cert_dir = Path(certificate_dir or f"certificates/{config['run_id']}").resolve()
    certs = generate_certificates(cert_dir, sorted(config["client_manifests"]))
    flower_home = runtime_dir / "home"
    flower_home.mkdir(parents=True, exist_ok=True)
    (flower_home / "config.toml").write_text(
        '[superlink]\ndefault = "rehearsal"\n\n[superlink.rehearsal]\n'
        f'address = "127.0.0.1:{port}"\nroot-certificates = {json.dumps(certs["ca"])}\n',
        encoding="utf-8")
    env = {**_offline_environment(config), "FLWR_HOME": str(flower_home)}
    processes: list[subprocess.Popen] = []
    streams = []
    nodes: dict[str, subprocess.Popen] = {}
    node_commands: dict[str, list[str]] = {}
    log = EventLog(run_dir, config["run_id"])

    def start(command: list[str], log_name: str) -> subprocess.Popen:
        stream = (runtime_dir / log_name).open("wb")
        streams.append(stream)
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=project_dir, env=env,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        processes.append(process)
        return process

    try:
        link = start([_binary("flower-superlink"), "--host", "127.0.0.1", "--port", str(port),
                      "--ssl-ca-certfile", certs["ca"], "--ssl-certfile", certs["certificate"],
                      "--ssl-keyfile", certs["key"], "--enable-supernode-auth",
                      "--disable-runtime-dependency-installation",
                      "--database", config.get("flower_database", ":flwr-in-memory:")], "superlink.log")
        ready = False
        startup_seconds = float(config.get("startup_timeout_seconds", 120))
        startup_end = time.monotonic() + startup_seconds
        while time.monotonic() < startup_end:
            if link.poll() is not None:
                raise RuntimeError(f"SuperLink exited; inspect {runtime_dir / 'superlink.log'}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    ready = True
                    break
            except OSError:
                time.sleep(0.1)
        if not ready:
            raise TimeoutError(f"SuperLink did not start within {startup_seconds} seconds")
        node_map = {}
        for offset, client_id in enumerate(sorted(config["client_manifests"])):
            registration = subprocess.run(
                [_binary("flwr"), "supernode", "register", str(cert_dir / f"node_{client_id}.pub"),
                 "rehearsal", "--name", f"client-{client_id}", "--format", "json"],
                cwd=project_dir, env=env, capture_output=True, text=True, timeout=30, check=True)
            (runtime_dir / f"register_{client_id}.log").write_text(registration.stdout + registration.stderr)
            payload = json.loads(registration.stdout)
            node_map[client_id] = int(payload["node-id"])
            node_commands[client_id] = [
                _binary("flower-supernode"), "--superlink", f"127.0.0.1:{port}",
                "--root-certificates", certs["ca"],
                "--auth-supernode-private-key", str(cert_dir / f"node_{client_id}.key"),
                "--host", "127.0.0.1", "--port", str(port + 1 + offset),
                "--node-config", f"partition-id={client_id}"]
            nodes[client_id] = start(node_commands[client_id], f"supernode_{client_id}.log")
        config["node_map"] = node_map
        config["transport_security"] = {
            "tls": True, "authenticated_registered_nodes": True,
            "ca_sha256": hashlib.sha256(Path(certs["ca"]).read_bytes()).hexdigest(),
            "nodes": {client_id: {
                "node_id": node_map[client_id],
                "public_key_sha256": hashlib.sha256(
                    (cert_dir / f"node_{client_id}.pub").read_bytes()).hexdigest(),
            } for client_id in node_map},
            "bind_address": f"127.0.0.1:{port}",
        }
        config["transport_state_backend"] = config.get("flower_database", ":flwr-in-memory:")
        config["runtime"] = "same_host_flower_deployment_tls_authenticated"
        config["serial_training"] = True
        config["persistent_worker"] = False
        config["rediscover_each_attempt"] = True
        atomic_json(run_dir / "runtime.json", config)
        override_path = runtime_dir / "run-overrides.toml"
        override_path.write_text(f'config-path = {json.dumps(str(run_dir / "runtime.json"))}\n')
        process = start([_binary("flwr"), "run", project_dir, "rehearsal", "--run-config",
                         str(override_path), "--stream"], "run.log")
        evaluation_allowance = float(config.get("evaluation_timeout_seconds", 600)) if config.get("validation_manifest") else 0
        overall_timeout = float(config.get("overall_timeout_seconds",
            float(config.get("startup_timeout_seconds", 120)) + (
                float(config["deadline_seconds"]) + evaluation_allowance
                + float(config.get("between_attempt_seconds", 0)) + 30) * int(config["attempts"]) + 120))
        deadline = time.monotonic() + overall_timeout
        interruptions = [dict(entry, triggered=False, restarted=False)
                         for entry in config.get("node_interruptions", [])]
        for entry in interruptions:
            if str(entry["client_id"]) not in nodes:
                raise ValueError("Node-interruption schedule names an unknown client")
        while process.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Flower runtime exceeded its {overall_timeout}-second total limit")
            for entry in interruptions:
                client_id = str(entry["client_id"])
                job_root = Path(config.get("client_job_dir", str(run_dir / "client_jobs")))
                progress_path = (job_root / config["run_id"]
                                 / f"attempt_{int(entry['attempt_id'])}" / client_id / "progress.json")
                if not entry["triggered"] and progress_path.exists():
                    progress = json.loads(progress_path.read_text())
                    if (progress.get("status") == "training"
                            and progress.get("optimizer_steps", 0) >= int(entry["after_steps"])):
                        original_pid = nodes[client_id].pid
                        _stop_tree(nodes[client_id])
                        entry["triggered"] = True
                        entry["restart_at"] = time.monotonic() + float(entry.get("rejoin_after_seconds", 5))
                        log.emit("supernode_process_interrupted", client_id=client_id,
                                 attempt_id=int(entry["attempt_id"]), pid=original_pid,
                                 mechanism="OS termination of SuperNode and descendant processes",
                                 observed_progress=progress)
                if entry["triggered"] and not entry["restarted"] and time.monotonic() >= entry["restart_at"]:
                    nodes[client_id] = start(node_commands[client_id],
                        f"supernode_{client_id}_rejoin_after_{entry['attempt_id']}.log")
                    entry["restarted"] = True
                    log.emit("supernode_process_restarted", client_id=client_id,
                             after_attempt_id=int(entry["attempt_id"]), pid=nodes[client_id].pid,
                             authenticated_node_id=node_map[client_id])
            time.sleep(0.05)
        if process.returncode != 0:
            raise RuntimeError(f"Flower run failed; inspect {runtime_dir / 'run.log'}")
        if not (run_dir / "final_results.json").exists():
            raise RuntimeError(f"No completed result produced; inspect {runtime_dir / 'run.log'}")
    finally:
        for process in reversed(processes):
            _stop_tree(process)
        for stream in streams:
            stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("simulate", "deploy-local", "write-config"))
    parser.add_argument("--config", help="Use an existing runtime JSON configuration")
    parser.add_argument("--run-id", default=time.strftime("federation-%Y%m%d-%H%M%S"))
    parser.add_argument("--run-dir")
    parser.add_argument("--asset-dir", default="assets/model")
    parser.add_argument("--adapter-dir", default="assets/adapter")
    parser.add_argument("--data-dir", default="assets/banking77")
    parser.add_argument("--layout", choices=("five", "three"), default="five")
    parser.add_argument("--scenario", choices=("balanced", "imbalanced"), default="balanced")
    parser.add_argument("--strategy", choices=("sample", "equal", "equal_clipped"), default="equal_clipped")
    parser.add_argument("--budgets", default="2")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--quorum", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadline-seconds", type=float, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validation-manifest")
    parser.add_argument("--evaluate-every", type=int, default=1)
    parser.add_argument("--evaluate-initial", action="store_true")
    parser.add_argument("--evaluation-device")
    parser.add_argument("--interrupt", action="append", default=[], metavar="ATTEMPT:CLIENT:STEP")
    parser.add_argument("--omit", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--port", type=int, default=19092)
    parser.add_argument("--certificate-dir")
    args = parser.parse_args()
    config = (json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config
              else create_config(args))
    if args.resume:
        config["resume"] = True
    atomic_json(Path(config["run_dir"]) / "runtime.json", config)
    if args.command == "simulate":
        simulate(config)
    elif args.command == "deploy-local":
        deploy_local(config, project_dir=_absolute(args.project_dir), port=args.port,
                     certificate_dir=args.certificate_dir)
    print(json.dumps({"run_id": config["run_id"], "run_dir": config["run_dir"],
                      "config": str(Path(config["run_dir"]) / "runtime.json")}))


if __name__ == "__main__":
    main()
