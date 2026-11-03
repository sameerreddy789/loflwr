"""Real Flower transport for a frozen-MPNet, CPU classification-head variant.

Only head tensors cross Flower messages. The server reads validation features;
each client reads its own archive. Legacy protocol b_hash/common_a_hash fields
mean head-state/frozen-encoder hashes here, and do not denote LoRA factors.
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading
import time
import traceback

from .checkpoint import CheckpointStore, EventLog, atomic_json, canonical_hash, read_progress
from .encoder_data import load_encoder_manifest, load_feature_split
from .data import content_hash
from .protocol import ClientReply, RoundCoordinator, RoundRequest, validate_request
from .state import (hash_state, state_from_dict, state_nbytes, state_to_dict,
                    subtract_states)


def validate_config(config: dict) -> dict:
    """Validate metadata without opening any client training feature archive."""
    manifest = load_encoder_manifest(config["dataset_dir"])
    if set(manifest["clients"]) != set(map(str, range(5))):
        raise ValueError("This runtime requires five logical clients")
    identity = config["identity"]
    from .encoder_head import HeadConfig
    head_config = asdict(HeadConfig(**config.get("head_config", {})))
    expected_schema = {"head.weight": {"shape": [20, 768], "dtype": "float32"},
                       "head.bias": {"shape": [20], "dtype": "float32"}}
    if (identity.get("kind") != "frozen_mpnet_linear_head"
            or identity.get("common_a_hash") != manifest["encoder_identity_hash"]
            or identity.get("model_revision") != manifest["model_revision"]
            or identity.get("encoder_identity_hash") != manifest["encoder_identity_hash"]
            or identity.get("feature_dim") != manifest["feature_dim"]
            or identity.get("labels") != manifest["labels"]
            or identity.get("head_schema") != expected_schema
            or identity.get("head_config") != head_config
            or identity.get("manifest_hash") != manifest["manifest_hash"]
            or identity.get("contract_hash") != content_hash({k: v for k, v in identity.items() if k != "contract_hash"})):
        raise ValueError("Frozen encoder identity does not match the dataset")
    if config.get("manifest_hash", manifest["manifest_hash"]) != manifest["manifest_hash"]:
        raise ValueError("Dataset differs from the frozen run manifest")
    if config.get("device", "cpu") != "cpu":
        raise ValueError("Encoder head transport uses CPU only")
    if set(config["budgets"]) != set(manifest["clients"]):
        raise ValueError("Provide one budget for every declared client")
    for name, values in (("attempts", [config["attempts"]]), ("budget", config["budgets"].values())):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
            raise ValueError(f"{name} must contain positive integers")
    if not set(map(str, config.get("omitted_clients", []))) <= set(manifest["clients"]):
        raise ValueError("Unknown omitted client")
    if len(set(map(str, config.get("omitted_clients", [])))) == 5:
        raise ValueError("At least one client must be selected")
    for name, default in (("evaluate_every", 1), ("worker_cpus", 1), ("quorum", 3)):
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.get("quorum", 3) > 5:
        raise ValueError("Quorum exceeds known client count")
    seed = config.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("Seed must be an integer in [0, 2**63)")
    for name, default in (("deadline_seconds", 60), ("startup_timeout_seconds", 120),
                          ("worker_idle_timeout_seconds", 900)):
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("persistent_worker", "resume", "evaluate_initial"):
        if name in config and not isinstance(config[name], bool):
            raise ValueError(f"{name} must be boolean")
    schedules = set()
    for name in ("simulated_failures", "process_interruptions"):
        for entry in config.get(name, []):
            attempt, client = entry["attempt_id"], str(entry["client_id"])
            if (isinstance(attempt, bool) or not isinstance(attempt, int)
                    or not 1 <= attempt <= config["attempts"] or client not in manifest["clients"]
                    or client in set(map(str, config.get("omitted_clients", [])))
                    or (attempt, client) in schedules):
                raise ValueError("Invalid or overlapping failure schedule")
            schedules.add((attempt, client))
            if name == "process_interruptions":
                step = entry["after_steps"]
                if isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= config["budgets"][client]:
                    raise ValueError("Interruption must follow a completed requested step")
    return manifest


def _trainer(config: dict, manifest: dict):
    from .encoder_head import HeadConfig, HeadTrainer
    return HeadTrainer(input_dim=manifest["feature_dim"], num_classes=len(manifest["labels"]),
                       seed=config.get("seed", 42), config=HeadConfig(**config.get("head_config", {})))


def load_initial_checkpoint(config: dict, manifest: dict) -> dict:
    """A server-only warm start; clients still receive the synchronized head."""
    directory = Path(config["initial_checkpoint_dir"])
    if not (directory / "checkpoint.json").is_file():
        raise FileNotFoundError("Warm-start checkpoint is missing")
    payload = CheckpointStore(directory).load()
    digest = hashlib.sha256((directory / "checkpoint.json").read_bytes()).hexdigest()
    expected = config.get("initial_checkpoint_sha256", digest)
    identity = payload["identity"]
    if (digest != expected or identity.get("kind") != "frozen_mpnet_linear_head"
            or identity.get("encoder_identity_hash") != manifest["encoder_identity_hash"]
            or identity.get("manifest_hash") != manifest["manifest_hash"]
            or identity.get("labels") != manifest["labels"]):
        raise ValueError("Warm-start checkpoint identity/hash mismatch")
    return {**payload, "file_sha256": digest}


def client_seed(request: RoundRequest) -> int:
    from .training import derive_client_seed
    return derive_client_seed(request.seed, request.attempt_id, request.client_id)


def _scheduled(config: dict, name: str, request: RoundRequest) -> dict | None:
    return next((item for item in config.get(name, [])
                 if item["attempt_id"] == request.attempt_id
                 and str(item["client_id"]) == request.client_id), None)


def train_worker(input_path: Path, output_path: Path, progress_path: Path) -> None:
    """The child owns only its client features, incoming head, and local optimizer."""
    job = json.loads(input_path.read_text(encoding="utf-8"))
    config, request = job["config"], RoundRequest.from_dict(job["request"])
    manifest = validate_config(config)
    incoming = state_from_dict(job["state"])
    validate_request(request, incoming, config["identity"])
    if request.manifest_hash != canonical_hash(manifest["clients"][request.client_id]):
        raise ValueError("Client archive identity differs from the request")
    X, y, _ = load_feature_split(config["dataset_dir"], request.client_id)
    trainer = _trainer(config, manifest)
    trainer.set_state(incoming)
    EventLog(config["run_dir"], request.run_id).emit("client_synchronized",
        client_id=request.client_id, attempt_id=request.attempt_id,
        version=request.version, base_head_hash=request.base_b_hash)
    interruption = _scheduled(config, "process_interruptions", request)

    def progress(step: int, work: dict) -> None:
        atomic_json(progress_path, {"optimizer_steps": step, "telemetry": work})
        if interruption and step >= interruption["after_steps"]:
            # Make a deterministic kill point visible to the supervising parent.
            # This wait is a failure trigger, not credited optimization work.
            while time.time() < request.deadline_unix:
                time.sleep(0.02)
            raise TimeoutError("Controlled interruption supervisor missed its deadline")

    telemetry = trainer.train(X, y, request.optimizer_steps, client_seed(request),
                              deadline_unix=request.deadline_unix, progress_callback=progress)
    telemetry.update(client_id=request.client_id, client_seed=client_seed(request),
                     manifest_examples=len(y), encoder_identity_hash=manifest["encoder_identity_hash"])
    if telemetry.get("optimizer_steps") != request.optimizer_steps:
        raise ValueError("A completed head reply requires the full requested work budget")
    atomic_json(output_path, {"status": "completed", "telemetry": telemetry,
                             "delta": state_to_dict(subtract_states(trainer.get_state(), incoming))})


def _run_cold_job(request: RoundRequest, incoming: dict, config: dict) -> dict:
    zero = subtract_states(incoming, incoming)
    if _scheduled(config, "simulated_failures", request):
        return {"status": "simulated_failure", "delta": zero,
                "telemetry": {"optimizer_steps": 0, "failure_kind": "simulated_client_failure"}}
    job_dir = Path(config["run_dir"]) / "client_jobs" / f"attempt_{request.attempt_id}" / request.client_id
    job_dir.mkdir(parents=True, exist_ok=False)
    input_path, output_path, progress_path = [job_dir / name for name in ("input.json", "output.json", "progress.json")]
    atomic_json(input_path, {"config": config, "request": request.to_dict(), "state": state_to_dict(incoming)})
    log = EventLog(config["run_dir"], request.run_id)
    interruption = _scheduled(config, "process_interruptions", request)
    status, progress, started = "worker_failed", {}, time.monotonic()
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                   "PYTHONPATH": str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    with (job_dir / "worker.log").open("wb") as stream:
        process = subprocess.Popen([sys.executable, "-m", "fedlora_edge.encoder_flower", "_worker",
                                    "--input", str(input_path), "--output", str(output_path),
                                    "--progress", str(progress_path)], stdout=stream, stderr=subprocess.STDOUT,
                                   env=environment, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        log.emit("head_client_process_started", client_id=request.client_id, attempt_id=request.attempt_id,
                 version=request.version, pid=process.pid)
        try:
            while process.poll() is None:
                progress = read_progress(progress_path, progress)
                if interruption and progress.get("optimizer_steps", 0) >= interruption["after_steps"]:
                    process.kill()
                    status = "process_interrupted"
                    break
                if time.time() >= request.deadline_unix:
                    process.terminate()
                    status = "deadline_exceeded"
                    break
                time.sleep(0.01)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
    if process.returncode == 0 and output_path.is_file() and time.time() < request.deadline_unix:
        result = json.loads(output_path.read_text(encoding="utf-8"))
        result["delta"] = state_from_dict(result["delta"])
    else:
        result = {"status": status, "delta": zero, "telemetry": progress.get("telemetry", {})}
        result["telemetry"].update(optimizer_steps=progress.get("optimizer_steps", 0), failure_kind=status)
        log.emit("head_client_process_stopped", client_id=request.client_id, attempt_id=request.attempt_id,
                 pid=process.pid, reason=status, observed_optimizer_steps=progress.get("optimizer_steps", 0))
    result["telemetry"].update(worker_returncode=process.returncode, job_wall_seconds=time.monotonic() - started)
    return result


_resident = None
_resident_lock = threading.Lock()


def stop_resident(*, kill=False):
    """Terminate only this actor's owned CPU optimizer child."""
    global _resident
    worker, _resident = _resident, None
    if worker is None:
        return
    process = worker["process"]
    if process.poll() is None:
        process.kill() if kill else process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    if process.stdin:
        process.stdin.close()
    worker["stream"].close()


atexit.register(stop_resident)


def _resident_job(request, incoming, config):
    global _resident
    zero = subtract_states(incoming, incoming)
    queued = time.monotonic()
    if not _resident_lock.acquire(timeout=max(0, request.deadline_unix - time.time())):
        return {"status": "queue_deadline_exceeded", "delta": zero,
                "telemetry": {"optimizer_steps": 0}}
    try:
        started = time.monotonic()
        queue_seconds = started - queued
        job_dir = Path(config["run_dir"]) / "client_jobs" / f"attempt_{request.attempt_id}" / request.client_id
        job_dir.mkdir(parents=True, exist_ok=False)
        input_path, output_path, progress_path = [job_dir / name for name in ("input.json", "output.json", "progress.json")]
        atomic_json(input_path, {"config": config, "request": request.to_dict(), "state": state_to_dict(incoming)})
        log = EventLog(config["run_dir"], request.run_id)
        profile = canonical_hash({"identity": config["identity"], "dataset_dir": config["dataset_dir"],
                                  "run_dir": config["run_dir"], "seed": config.get("seed", 42)})
        reused = _resident is not None and _resident["profile"] == profile and _resident["process"].poll() is None
        if not reused:
            stop_resident()
            log_dir = Path(config["run_dir"]) / "client_workers"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"actor_{os.getpid()}_{time.time_ns()}.log"
            stream = log_path.open("wb")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                   "PYTHONPATH": str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", "")}
            try:
                process = subprocess.Popen([sys.executable, "-m", "fedlora_edge.encoder_flower", "_serve",
                    "--parent-pid", str(os.getpid()), "--idle-timeout", str(config.get("worker_idle_timeout_seconds", 900))],
                    stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT, text=True, env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except BaseException:
                stream.close()
                raise
            _resident = {"process": process, "stream": stream, "profile": profile,
                         "log_path": str(log_path), "jobs": 0}
        worker = _resident
        process = worker["process"]
        worker["jobs"] += 1
        log.emit("head_client_process_reused" if reused else "head_client_process_started",
                 client_id=request.client_id, attempt_id=request.attempt_id, version=request.version,
                 pid=process.pid, worker_job_index=worker["jobs"])
        process.stdin.write(json.dumps({"input": str(input_path), "output": str(output_path), "progress": str(progress_path)}) + "\n")
        process.stdin.flush()
        interruption = _scheduled(config, "process_interruptions", request)
        progress = {}
        result = None
        status = "worker_failed"
        while True:
            progress = read_progress(progress_path, progress)
            if interruption and progress.get("optimizer_steps", 0) >= interruption["after_steps"]:
                status = "process_interrupted"
                stop_resident(kill=True)
                break
            if time.time() >= request.deadline_unix:
                status = "deadline_exceeded"
                stop_resident()
                break
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                payload = None
            if payload:
                result = payload
                result["delta"] = state_from_dict(result["delta"])
                if result["status"] != "completed":
                    stop_resident()
                break
            if process.poll() is not None:
                stop_resident()
                break
            time.sleep(0.01)
        if result is None:
            result = {"status": status, "delta": zero, "telemetry": progress.get("telemetry", {})}
            result["telemetry"].update(optimizer_steps=progress.get("optimizer_steps", 0), failure_kind=status)
            log.emit("head_client_process_stopped", client_id=request.client_id, attempt_id=request.attempt_id,
                     version=request.version, pid=process.pid, reason=status, returncode=process.returncode,
                     observed_optimizer_steps=progress.get("optimizer_steps", 0))
        result["telemetry"].update(worker_returncode=process.returncode, job_wall_seconds=time.monotonic() - started,
                                 worker_reused=reused, worker_job_index=worker["jobs"], job_pid=process.pid,
                                 persistent_queue_seconds=queue_seconds, process_log=worker["log_path"])
        return result
    except BaseException:
        stop_resident()
        raise
    finally:
        _resident_lock.release()


def _run_job(request, incoming, config):
    if _scheduled(config, "simulated_failures", request):
        return {"status": "simulated_failure", "delta": subtract_states(incoming, incoming),
                "telemetry": {"optimizer_steps": 0, "failure_kind": "simulated_client_failure"}}
    if config.get("persistent_worker", True):
        return _resident_job(request, incoming, config)
    return _run_cold_job(request, incoming, config)


def serve_worker(parent_pid, idle_timeout):
    """Reuse imports only: every command creates/resynchronizes a fresh head."""
    import psutil
    parent = psutil.Process(parent_pid)
    parent_created = parent.create_time()

    def watch_parent():
        while True:
            try:
                if not parent.is_running() or parent.create_time() != parent_created:
                    os._exit(1)
            except psutil.NoSuchProcess:
                os._exit(1)
            time.sleep(0.5)

    threading.Thread(target=watch_parent, daemon=True).start()
    jobs = Queue()

    def read_jobs():
        for line in sys.stdin:
            jobs.put(line)
        jobs.put(None)

    threading.Thread(target=read_jobs, daemon=True).start()
    while True:
        try:
            line = jobs.get(timeout=idle_timeout)
        except Empty:
            return
        if line is None:
            return
        command = json.loads(line)
        try:
            train_worker(Path(command["input"]), Path(command["output"]), Path(command["progress"]))
        except Exception as error:
            traceback.print_exc()
            atomic_json(command["output"], {"status": "worker_failed", "delta": {},
                "telemetry": {"optimizer_steps": 0, "worker_error": f"{type(error).__name__}: {error}"}})
            return


def make_client_app(config: dict):
    from flwr.app import ConfigRecord, Message, RecordDict
    from flwr.clientapp import ClientApp
    from .flower_app import _array_record, _tensor_state, serialized_message_bytes
    app = ClientApp()

    @app.query("identity")
    def identity(message, context):
        return Message(RecordDict({"metadata": ConfigRecord({"client_id": str(context.node_config["partition-id"])})}), reply_to=message)

    @app.train()
    def train(message, context):
        received = time.time()
        request = RoundRequest.from_dict(json.loads(message.content["request"]["json"]))
        client = str(context.node_config["partition-id"])
        if client != request.client_id or set(message.content) != {"head", "request"}:
            raise ValueError("Wrong client or unexpected request payload")
        manifest = validate_config(config)
        incoming = _tensor_state(message.content["head"])
        validate_request(request, incoming, config["identity"])
        if request.manifest_hash != canonical_hash(manifest["clients"][client]):
            raise ValueError("Requested client manifest differs from local metadata")
        result = _run_job(request, incoming, config)
        result["telemetry"].update(queue_wait_seconds=max(0, received - message.metadata.created_at),
                                   declared_sample_size=manifest["clients"][client]["count"])
        EventLog(config["run_dir"], request.run_id).emit("client_reply_prepared", client_id=client,
            attempt_id=request.attempt_id, version=request.version, base_head_hash=request.base_b_hash,
            serialized_request_bytes=serialized_message_bytes(message))
        return Message(RecordDict({"reply": ConfigRecord({"request": json.dumps(request.to_dict()),
                         "status": result["status"], "telemetry": json.dumps(result["telemetry"], allow_nan=False)}),
                         "delta": _array_record(result["delta"])}), reply_to=message)
    return app


def evaluate_head(trainer, X, y, labels: list[str], state: dict) -> dict:
    import numpy as np
    from .training import classification_metrics
    trainer.set_state(state)
    scores = trainer.predict_scores(X).astype(np.float64)
    maximum = scores.max(axis=1)
    logsumexp = maximum + np.log(np.exp(scores - maximum[:, None]).sum(axis=1))
    return {**classification_metrics([labels[int(i)] for i in y], [labels[int(i)] for i in scores.argmax(axis=1)]),
            "task_loss": float(np.mean(logsumexp - scores[np.arange(len(y)), y])),
            "head_hash": hash_state(state), "b_hash": hash_state(state), "scoring_rule": "linear_head_argmax"}


def make_server_app(config: dict):
    from flwr.app import ConfigRecord, Message, RecordDict
    from flwr.serverapp import ServerApp
    from .flower_app import _array_record, _collect, _discover, _tensor_state, serialized_message_bytes
    app = ServerApp()

    @app.main()
    def main(grid, context):
        started = time.monotonic()
        manifest = validate_config(config)
        trainer = _trainer(config, manifest)
        if config.get("initial_checkpoint_dir"):
            initial = load_initial_checkpoint(config, manifest)
            trainer.set_state(initial["state"])
            EventLog(config["run_dir"], config["run_id"]).emit("warm_start_loaded",
                source_run_id=initial["run_id"], source_version=initial["version"],
                source_head_hash=initial["b_hash"], source_file_sha256=initial["file_sha256"])
        counts = {client: entry["count"] for client, entry in manifest["clients"].items()}
        hashes = {client: canonical_hash(entry) for client, entry in manifest["clients"].items()}
        coordinator = RoundCoordinator(run_dir=config["run_dir"], run_id=config["run_id"],
            initial_state=trainer.get_state(), identity=config["identity"], clients=counts,
            quorum=config.get("quorum", 3), strategy=config.get("strategy", "equal_clipped"),
            seed=config.get("seed", 42), deadline_seconds=config.get("deadline_seconds", 60),
            resume=config.get("resume", False), config_extra={"kind": "frozen_mpnet_linear_head",
            "runtime": {k: v for k, v in config.items() if k != "resume"},
            "manifest_hash": manifest["manifest_hash"], "protocol_aliases": {
                "b_hash": "classifier_head_state_hash", "common_a_hash": "frozen_encoder_identity_hash"}})
        X, y, _ = load_feature_split(config["dataset_dir"])
        evaluation_path = Path(config["run_dir"]) / "evaluations.jsonl"
        evaluated = set()
        evaluation_history = []
        selected_store = CheckpointStore(Path(config["run_dir"]) / "selected")
        selected, best_key = None, (-1.0, -1.0)
        if config.get("resume", False) and selected_store.path.is_file():
            saved = selected_store.load()
            if saved["identity"] != config["identity"] or saved["config_hash"] != coordinator.config_hash:
                raise ValueError("Selected checkpoint belongs to a different run contract")
            selected = saved["metadata"]["selection"]
            best_key = (selected["metrics"]["accuracy"], selected["metrics"]["macro_f1"])
        if evaluation_path.is_file():
            for line in evaluation_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["run_id"] != config["run_id"] or row["split"] != "validation":
                    raise ValueError("Evaluation history belongs to a different run or split")
                evaluation_history.append(row)
                evaluated.add(row["version"])

        def evaluate():
            nonlocal selected, best_key
            record = {"run_id": config["run_id"], "attempt_id": coordinator.attempt_id,
                      "version": coordinator.version, "timestamp": time.time(), "split": "validation",
                      "manifest_hash": canonical_hash(manifest["validation"]),
                      "metrics": evaluate_head(trainer, X, y, manifest["labels"], coordinator.state)}
            with evaluation_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            coordinator.events.emit("evaluation", **{k: v for k, v in record.items() if k != "run_id"})
            evaluated.add(coordinator.version)
            evaluation_history.append(record)
            key = (record["metrics"]["accuracy"], record["metrics"]["macro_f1"])
            if key > best_key:
                selected, best_key = record, key
                selected_store.save(run_id=config["run_id"], attempt_id=coordinator.attempt_id,
                    version=coordinator.version, state=coordinator.state, identity=config["identity"],
                    config_hash=coordinator.config_hash, metadata={"selection": record,
                        "selection_rule": "Highest evaluated validation accuracy, then macro F1, earliest evaluation"})

        coordinator.events.emit("run_started", kind="frozen_mpnet_linear_head", runtime="flower_ray_one_cpu_worker",
            traffic_measurement="Serialized Flower protobuf; no feature/query arrays; excludes transport framing")
        if config.get("evaluate_initial", True) and coordinator.version not in evaluated:
            evaluate()
        mapping, deadline = {}, time.monotonic() + config.get("startup_timeout_seconds", 120)
        while len(mapping) < len(counts) and time.monotonic() < deadline:
            mapping = _discover(grid, set(counts), config)
            if len(mapping) < len(counts):
                time.sleep(0.1)
        if not mapping:
            raise RuntimeError("No Flower clients registered")
        coordinator.events.emit("nodes_registered", node_map=mapping, logical_clients=len(mapping))
        previous_absent = set()
        if coordinator.attempt_id > config["attempts"]:
            raise ValueError("Declared horizon precedes the checkpoint")
        for _ in range(coordinator.attempt_id, config["attempts"]):
            budgets = {c: config["budgets"][c] for c in counts if c not in set(map(str, config.get("omitted_clients", [])))}
            requests = coordinator.begin(budgets, manifest_hashes=hashes)
            messages = [Message(RecordDict({"head": _array_record(coordinator.base),
                        "request": ConfigRecord({"json": json.dumps(request.to_dict())})}),
                        dst_node_id=mapping[c], message_type="train", group_id=str(request.attempt_id),
                        ttl=max(0.001, request.deadline_unix - time.time())) for c, request in requests.items() if c in mapping]
            coordinator.events.emit("requests_sent", attempt_id=coordinator.attempt_id, version=coordinator.version,
                serialized_request_bytes=sum(serialized_message_bytes(m) for m in messages),
                raw_head_bytes_per_client=state_nbytes(coordinator.base), transmitted_arrays=["head.weight", "head.bias"])
            inverse = {node: client for client, node in mapping.items()}
            for message, received_at in _collect(grid, messages, max(0, coordinator.deadline_unix - time.time())):
                client = inverse.get(message.metadata.src_node_id, "unknown")
                coordinator.record_transport_reply(client, serialized_message_bytes(message))
                if message.has_error():
                    coordinator.reject(client, f"flower_error:{message.error.code}:{message.error.reason}")
                    continue
                try:
                    if set(message.content) != {"reply", "delta"}:
                        raise ValueError("Unexpected response payload; only head delta and telemetry allowed")
                    record = message.content["reply"]
                    reply = ClientReply(RoundRequest.from_dict(json.loads(record["request"])),
                        _tensor_state(message.content["delta"]), record["status"], json.loads(record["telemetry"]),
                        serialized_message_bytes(message))
                except (ValueError, KeyError, TypeError, OverflowError) as error:
                    coordinator.reject(client, f"malformed_payload:{type(error).__name__}:{error}")
                    continue
                if coordinator.receive(reply, authenticated_client_id=client, received_at=received_at) and client in previous_absent:
                    coordinator.events.emit("client_rejoined", client_id=client, attempt_id=coordinator.attempt_id,
                        version=coordinator.version, synchronized_head_hash=reply.request.base_b_hash)
            result = coordinator.finish()
            previous_absent = set(requests) - set(coordinator.accepted)
            if result["status"] == "committed" and coordinator.version % config.get("evaluate_every", 1) == 0:
                evaluate()
            print(json.dumps({"attempt": coordinator.attempt_id, "version": coordinator.version,
                              "status": result["status"], "accepted": result["accepted"]}), flush=True)
        if not evaluation_history or evaluation_history[-1]["attempt_id"] != coordinator.attempt_id:
            evaluate()
        atomic_json(Path(config["run_dir"]) / "final_results.json", {
            "run_id": config["run_id"], "kind": "frozen_mpnet_linear_head", "runtime": "flower_ray_one_cpu_worker",
            "attempts": coordinator.attempt_id, "committed_versions": coordinator.version,
            "config_hash": coordinator.config_hash, "head_hash": hash_state(coordinator.state),
            "identity": config["identity"], "total_wall_seconds": time.monotonic() - started,
            "rounds": coordinator.results, "evaluations": evaluation_history, "selected": selected,
            "limits": ["One host with five logical clients; not physical-laptop evidence.",
                "Process interruption kills a local optimizer child; simulated failures are labelled separately.",
                "Frozen encoder plus trained head; not a LoRA or full-model optimization experiment."]})
        coordinator.events.emit("run_completed", attempt_id=coordinator.attempt_id, version=coordinator.version)
    return app


def simulate(config: dict) -> None:
    config = dict(config)
    config["dataset_dir"] = str(Path(config["dataset_dir"]).resolve())
    config["run_dir"] = str(Path(config["run_dir"]).resolve())
    config.setdefault("run_id", Path(config["run_dir"]).name)
    config.setdefault("persistent_worker", True)
    config.setdefault("selection_rule", "Highest evaluated validation accuracy, then macro F1, earliest evaluation")
    manifest = validate_config(config)
    if config.get("initial_checkpoint_dir"):
        config["initial_checkpoint_dir"] = str(Path(config["initial_checkpoint_dir"]).resolve())
        initial = load_initial_checkpoint(config, manifest)
        config["initial_checkpoint_sha256"] = initial["file_sha256"]
        config["initial_head_hash"] = initial["b_hash"]
    config["manifest_hash"] = manifest["manifest_hash"]
    root = Path(__file__).resolve().parents[1]
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        "fedlora_edge/encoder_flower.py", "fedlora_edge/encoder_head.py", "fedlora_edge/encoder_data.py",
        "fedlora_edge/aggregation.py", "fedlora_edge/protocol.py", "fedlora_edge/state.py",
        "fedlora_edge/checkpoint.py", "fedlora_edge/flower_app.py", "fedlora_edge/training.py",
        "tools/run_encoder_flower.py")}
    if config.get("source_files_sha256", sources) != sources:
        raise ValueError("Runtime source differs from the frozen configuration")
    config["source_files_sha256"] = sources
    output = Path(config["run_dir"])
    if output.exists():
        if not config.get("resume", False):
            raise FileExistsError("Use a new run directory; existing evidence is immutable")
        previous = json.loads((output / "runtime.json").read_text(encoding="utf-8"))
        if {k: v for k, v in previous.items() if k != "resume"} != {k: v for k, v in config.items() if k != "resume"}:
            raise ValueError("Resume configuration/source identity changed")
    elif config.get("resume", False):
        raise FileNotFoundError("No existing run to resume")
    os.environ.update({"FLWR_TELEMETRY_ENABLED": "0", "FLWR_DISABLE_UPDATE_CHECK": "1",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION": "1"})
    from flwr.simulation import run_simulation
    from .federation import initialize_ray
    cpus = int(config.get("worker_cpus", 1))
    backend = {"init_args": {"num_cpus": cpus, "num_gpus": 0, "include_dashboard": False},
               "client_resources": {"num_cpus": cpus, "num_gpus": 0}}
    atomic_json(Path(config["run_dir"]) / "runtime.json", config)
    startup = initialize_ray(backend)
    atomic_json(Path(config["run_dir"]) / "runtime_preparation.json", {
        "runtime": "flower_ray_one_cpu_worker", "logical_clients": 5, "workers": 1,
        "worker_cpus": cpus, "worker_gpus": 0, "ray_startup_seconds": startup})
    run_simulation(server_app=make_server_app(config), client_app=make_client_app(config),
                   num_supernodes=5, backend_config=backend)
    if not (Path(config["run_dir"]) / "final_results.json").is_file():
        raise RuntimeError("Flower exited without a completed head-variant result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal CPU head-training subprocess")
    parser.add_argument("mode", choices=["_worker", "_serve"])
    for option in ("input", "output", "progress"):
        parser.add_argument(f"--{option}", type=Path)
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--idle-timeout", type=float, default=900)
    args = parser.parse_args()
    if args.mode == "_serve":
        if args.parent_pid < 1 or not math.isfinite(args.idle_timeout) or args.idle_timeout <= 0:
            parser.error("A positive parent PID and finite idle timeout are required")
        serve_worker(args.parent_pid, args.idle_timeout)
    elif all((args.input, args.output, args.progress)):
        train_worker(args.input, args.output, args.progress)
    else:
        parser.error("Supply --input, --output and --progress for a one-shot worker")
