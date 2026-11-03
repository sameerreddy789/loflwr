"""Flower 1.40 ClientApp/ServerApp using B-only, deadline-bounded messages.

Both Ray simulation and networked Deployment Runtime call these exact handlers.
The optional process-interruption schedule kills a real optimizer subprocess;
it is recorded separately from remote-node/network interruption evidence.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from queue import Empty, Queue
from pathlib import Path
from typing import Any, Iterable

from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.serde import message_to_proto
from flwr.serverapp import Grid, ServerApp

from .checkpoint import EventLog, atomic_json, canonical_hash, read_progress
from .data import load_manifest
from .protocol import ClientReply, RoundCoordinator, RoundRequest, validate_request
from .state import TensorData, TensorState, state_from_dict, state_nbytes, state_to_dict


def _config(context: Context, override: dict | None) -> dict:
    if override is not None:
        return dict(override)
    path = context.run_config.get("config-path", os.environ.get("FEDLORA_CONFIG", ""))
    if not path:
        raise ValueError("Supply config-path in Flower run config or FEDLORA_CONFIG")
    return json.loads(Path(str(path)).read_text(encoding="utf-8"))


def _array_record(state: TensorState) -> ArrayRecord:
    return ArrayRecord({name: Array(tensor.to_numpy()) for name, tensor in state.items()})


def _tensor_state(record: ArrayRecord) -> TensorState:
    result = {}
    for name, array in record.items():
        # Do not silently cast a hostile/wrong dtype to FP32 at the receive boundary.
        if array.dtype != "float32":
            raise ValueError("Wire tensors must have dtype float32")
        value = array.numpy()
        if str(value.dtype) != "float32" or list(value.shape) != list(array.shape):
            raise ValueError("Wire array metadata differs from its payload")
        result[name] = TensorData.from_numpy(value)
    return result


def serialized_message_bytes(message: Message) -> int:
    """Measured Flower protobuf bytes; excludes HTTP/TLS framing and retries."""
    return len(message_to_proto(message).SerializeToString())


def _manifest_bundle(paths: list[str]) -> tuple[list[dict], str]:
    records, hashes = [], []
    for name in paths:
        records.extend(load_manifest(name))
        hashes.append(json.loads(Path(name).read_text(encoding="utf-8"))["manifest_hash"])
    return records, canonical_hash(hashes)


def _collect(grid: Grid, messages: list[Message], timeout: float) -> Iterable[tuple[Message, float]]:
    """Bound collection even if an SDK HTTP call retries or blocks past the TTL.

    The daemon only handles transport; it never owns training compute. Expired
    jobs are independently stopped by each ClientApp's process supervisor.
    """
    end = time.monotonic() + max(0, timeout)
    results: Queue = Queue()
    stop = threading.Event()

    def collect() -> None:
        try:
            pending = set(grid.push_messages(messages))
            while pending and not stop.is_set() and time.monotonic() < end:
                for message in grid.pull_messages(pending):
                    pending.discard(message.metadata.reply_to_message_id)
                    results.put(("message", (message, time.time())))
                if pending:
                    stop.wait(min(0.1, max(0, end - time.monotonic())))
        except Exception as error:
            results.put(("error", error))
        finally:
            results.put(("done", None))

    threading.Thread(target=collect, name="flower-bounded-collection", daemon=True).start()
    try:
        while time.monotonic() < end:
            try:
                kind, value = results.get(timeout=max(0, end - time.monotonic()))
            except Empty:
                break
            if kind == "error":
                raise value
            if kind == "done":
                break
            yield value
    finally:
        stop.set()


def _stop_job(process: subprocess.Popen, *, kill: bool = False) -> None:
    if process.poll() is not None:
        return
    if kill:
        process.kill()
    else:
        process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _run_training_unlocked(request: RoundRequest, incoming: TensorState, config: dict,
                           records: list[dict], log: EventLog) -> dict:
    root = Path(config.get("client_job_dir", str(Path(config["run_dir"]) / "client_jobs")))
    job_dir = root / request.run_id / f"attempt_{request.attempt_id}" / request.client_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path, output_path, progress_path = (job_dir / name for name in
                                               ("input.json", "output.json", "progress.json"))
    if output_path.exists() or progress_path.exists():
        raise ValueError("Refusing to reuse an existing client job/attempt")
    atomic_json(input_path, {"request": request.to_dict(), "state": state_to_dict(incoming),
                             "runtime": config, "records": records})
    if config.get("persistent_worker", False):
        if config.get("runtime") != "flower_ray_one_gpu_worker" or config.get("serial_training", False):
            raise ValueError("Persistent GPU workers require the single-actor Ray runtime")
        from .persistent_worker import run_persistent_job

        return run_persistent_job(request, input_path, output_path, progress_path, config, log)
    interruption = next((entry for entry in config.get("process_interruptions", [])
                         if int(entry["attempt_id"]) == request.attempt_id
                         and str(entry["client_id"]) == request.client_id), None)
    progress: dict[str, Any] = {}
    status = "failed"
    started = time.monotonic()
    remaining = max(0, request.deadline_unix - time.time())
    with (job_dir / "process.log").open("wb") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "fedlora_edge.client_job", "--input", str(input_path),
             "--output", str(output_path), "--progress", str(progress_path)],
            stdout=output, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                 "TOKENIZERS_PARALLELISM": "false",
                 "PYTHONPATH": str(Path(__file__).resolve().parent.parent) + os.pathsep
                 + os.environ.get("PYTHONPATH", "")},
        )
        log.emit("client_process_started", attempt_id=request.attempt_id, version=request.version,
                 client_id=request.client_id, pid=process.pid, job_dir=str(job_dir),
                 base_b_hash=request.base_b_hash)
        try:
            while process.poll() is None:
                progress = read_progress(progress_path, progress)
                if (interruption is not None and progress.get("status") == "training"
                        and progress.get("optimizer_steps", 0) >= int(interruption["after_steps"])):
                    _stop_job(process, kill=True)
                    status = "process_interrupted"
                    log.emit("client_process_interrupted", attempt_id=request.attempt_id,
                             version=request.version, client_id=request.client_id, pid=process.pid,
                             mechanism="OS process kill", observed_progress=progress,
                             returncode=process.returncode)
                    break
                if time.monotonic() - started >= remaining:
                    _stop_job(process)
                    status = "deadline_expired"
                    log.emit("client_process_deadline", attempt_id=request.attempt_id,
                             version=request.version, client_id=request.client_id, pid=process.pid,
                             observed_progress=progress, returncode=process.returncode)
                    break
                time.sleep(0.05)
        finally:
            _stop_job(process)
    elapsed = time.monotonic() - started
    if status == "failed" and process.returncode == 0 and output_path.exists():
        result = json.loads(output_path.read_text(encoding="utf-8"))
        result["delta"] = state_from_dict(result["delta"])
        result["telemetry"].update(job_process_seconds=elapsed, job_pid=process.pid)
        return result
    return {"status": status, "delta": {}, "telemetry": {
        **progress.get("telemetry", {}), "optimizer_steps": progress.get("optimizer_steps", 0),
        "job_process_seconds": elapsed, "job_pid": process.pid,
        "job_returncode": process.returncode, "process_log": str(job_dir / "process.log"),
    }}


def _run_training(request: RoundRequest, incoming: TensorState, config: dict,
                  records: list[dict], log: EventLog) -> dict:
    if not config.get("serial_training", False):
        return _run_training_unlocked(request, incoming, config, records, log)
    from filelock import FileLock, Timeout

    started = time.monotonic()
    lock = FileLock(str(Path(config["run_dir"]) / "training-worker.lock"))
    try:
        with lock.acquire(timeout=max(0, request.deadline_unix - time.time()), poll_interval=0.05):
            queued = time.monotonic() - started
            if time.time() >= request.deadline_unix:
                raise Timeout(str(lock.lock_file))
            result = _run_training_unlocked(request, incoming, config, records, log)
            result["telemetry"]["compute_queue_seconds"] = queued
            return result
    except Timeout:
        return {"status": "queue_deadline_expired", "delta": {},
                "telemetry": {"optimizer_steps": 0,
                              "compute_queue_seconds": time.monotonic() - started}}


def make_client_app(config_override: dict | None = None) -> ClientApp:
    app = ClientApp()

    @app.query("identity")
    def identity(message: Message, context: Context) -> Message:
        client_id = str(context.node_config["partition-id"])
        return Message(RecordDict({"metadata": ConfigRecord({"client_id": client_id})}),
                       reply_to=message)

    @app.train()
    def train(message: Message, context: Context) -> Message:
        receipt = time.time()
        handler_started = time.monotonic()
        config = _config(context, config_override)
        for name in ("asset_dir", "adapter_dir", "client_job_dir", "device"):
            if name.replace("_", "-") in context.node_config:
                config[name] = context.node_config[name.replace("_", "-")]
        request = RoundRequest.from_dict(json.loads(message.content["request"]["json"]))
        client_id = str(context.node_config["partition-id"])
        if client_id != request.client_id:
            raise ValueError("Request addressed to a different logical client")
        incoming = _tensor_state(message.content["b"])
        envelope = json.loads((Path(config["adapter_dir"]) / "initial_b.json").read_text())
        validate_request(request, incoming, envelope["identity"])
        records, manifest_hash = _manifest_bundle(config["client_manifests"][client_id])
        if request.manifest_hash != manifest_hash:
            raise ValueError("Client manifest differs from requested manifest")
        log = EventLog(config["run_dir"], request.run_id)
        log.emit("client_synchronized", attempt_id=request.attempt_id, version=request.version,
                 client_id=client_id, base_b_hash=request.base_b_hash,
                 serialized_request_bytes=serialized_message_bytes(message))
        result = _run_training(request, incoming, config, records, log)
        telemetry = result["telemetry"]
        telemetry["queue_wait_seconds"] = max(0, receipt - float(message.metadata.created_at))
        telemetry["client_processing_seconds"] = time.monotonic() - handler_started
        telemetry["declared_sample_size"] = len(records)
        content = RecordDict({
            "reply": ConfigRecord({"request": json.dumps(request.to_dict()),
                                    "status": result["status"], "telemetry": json.dumps(telemetry)}),
            "delta": _array_record(result["delta"]),
        })
        return Message(content, reply_to=message)

    return app


def _discover(grid: Grid, known_clients: set[str], config: dict) -> dict[str, int]:
    fixed = {str(c): int(n) for c, n in config.get("node_map", {}).items()}
    if fixed:
        if not set(fixed) <= known_clients or len(set(fixed.values())) != len(fixed):
            raise ValueError("Invalid pinned node map")
        available = set(grid.get_node_ids())
        return {client_id: node for client_id, node in fixed.items() if node in available}
    nodes = list(grid.get_node_ids())
    messages = [Message(RecordDict(), dst_node_id=node, message_type="query.identity", ttl=30)
                for node in nodes]
    mapping: dict[str, int] = {}
    for reply, _ in _collect(grid, messages, min(30, config.get("discovery_timeout_seconds", 30))):
        if reply.has_error():
            continue
        client_id = reply.content["metadata"]["client_id"]
        if client_id not in known_clients or client_id in mapping:
            raise ValueError("Unknown or duplicate partition returned by registered nodes")
        mapping[client_id] = reply.metadata.src_node_id
    return mapping


def make_server_app(config_override: dict | None = None) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        run_started = time.monotonic()
        config = _config(context, config_override)
        envelope = json.loads((Path(config["adapter_dir"]) / "initial_b.json").read_text())
        initial = state_from_dict(envelope["state"])
        counts, hashes = {}, {}
        for client_id, paths in config["client_manifests"].items():
            records, digest = _manifest_bundle(paths)
            counts[client_id], hashes[client_id] = len(records), digest
        coordinator = RoundCoordinator(
            run_dir=config["run_dir"], run_id=config["run_id"], initial_state=initial,
            identity=envelope["identity"], clients=counts, quorum=int(config.get("quorum", 3)),
            strategy=config.get("strategy", "equal_clipped"), seed=int(config.get("seed", 42)),
            deadline_seconds=float(config.get("deadline_seconds", 300)),
            epsilon=float(config.get("epsilon", 1e-12)), resume=bool(config.get("resume", False)),
            config_extra={"runtime": {k: v for k, v in config.items() if k != "resume"},
                          "manifest_hashes": hashes},
        )
        coordinator.events.emit("run_started", config_hash=coordinator.config_hash,
                                runtime=config.get("runtime", "unspecified"),
                                traffic_measurement="serialized Flower protobuf; excludes HTTP/TLS framing/retries")
        evaluation_history = []
        evaluation_path = Path(config["run_dir"]) / "evaluations.jsonl"
        if config.get("resume") and evaluation_path.exists():
            evaluation_history = [json.loads(line) for line in evaluation_path.read_text().splitlines() if line]
        evaluator = None
        evaluation_records = None
        evaluation_manifest_hash = None
        if config.get("validation_manifest"):
            manifest = json.loads(Path(config["validation_manifest"]).read_text(encoding="utf-8"))
            if manifest.get("name") != "validation":
                raise ValueError("The per-round evaluation hook accepts the validation manifest only")
            evaluation_records = load_manifest(config["validation_manifest"])
            evaluation_manifest_hash = manifest["manifest_hash"]

        def evaluate_current() -> None:
            nonlocal evaluator
            if evaluation_records is None:
                return
            from .training import LocalTrainer, TrainingConfig

            started = time.monotonic()
            if evaluator is None:
                evaluator = LocalTrainer(
                    asset_dir=config["asset_dir"], adapter_dir=config["adapter_dir"],
                    device=config.get("evaluation_device", config.get("device", "cpu")),
                    config=TrainingConfig(**config.get("training_config", {})),
                )
            metrics = evaluator.evaluate(evaluation_records, b_state=coordinator.state)
            cleanup = {"cuda_cache_released": False}
            if str(evaluator.device).startswith("cuda"):
                import gc
                import torch

                gc.collect()
                torch.cuda.empty_cache()
                cleanup = {"cuda_cache_released": True,
                           "allocated_bytes_after_cleanup": torch.cuda.memory_allocated(evaluator.device),
                           "reserved_bytes_after_cleanup": torch.cuda.memory_reserved(evaluator.device)}
            record = {"run_id": config["run_id"], "timestamp": time.time(), "split": "validation",
                      "attempt_id": coordinator.attempt_id, "version": coordinator.version,
                      "manifest_hash": evaluation_manifest_hash, "metrics": metrics,
                      "evaluation_wall_seconds": time.monotonic() - started,
                      "post_evaluation_cleanup": cleanup}
            with evaluation_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            evaluation_history.append(record)
            coordinator.events.emit("evaluation", **{key: value for key, value in record.items()
                                                       if key not in {"run_id", "timestamp"}})

        if config.get("evaluate_initial", False) and coordinator.version == 0:
            evaluate_current()
        end_discovery = time.monotonic() + float(config.get("startup_timeout_seconds", 120))
        mapping: dict[str, int] = {}
        while len(mapping) < len(counts) and time.monotonic() < end_discovery:
            mapping = _discover(grid, set(counts), config)
            if len(mapping) < len(counts):
                time.sleep(0.5)
        coordinator.events.emit("nodes_registered", node_map=mapping)
        if not mapping:
            raise RuntimeError("No registered Flower clients became available")
        previous_absent: set[str] = set()
        attempts = int(config.get("attempts", 1))
        if attempts < coordinator.attempt_id:
            raise ValueError("Declared horizon precedes the last persisted attempt")
        for _ in range(coordinator.attempt_id, attempts):
            # Rediscovery supports a restarted SuperNode receiving a new transport ID.
            if config.get("rediscover_each_attempt", False):
                mapping.update(_discover(grid, set(counts), config))
            budgets = {c: int(config["budgets"][c]) for c in counts if c in mapping
                       and c not in {str(x) for x in config.get("omitted_clients", [])}}
            if not budgets:
                raise RuntimeError("No selected clients")
            requests = coordinator.begin(budgets, manifest_hashes=hashes)
            messages = [Message(
                RecordDict({"b": _array_record(coordinator.base),
                            "request": ConfigRecord({"json": json.dumps(request.to_dict())})}),
                dst_node_id=mapping[c], message_type="train", group_id=str(request.attempt_id),
                ttl=max(0.001, request.deadline_unix - time.time()),
            ) for c, request in requests.items()]
            coordinator.events.emit("requests_sent", attempt_id=coordinator.attempt_id,
                                    version=coordinator.version,
                                    serialized_request_bytes=sum(serialized_message_bytes(m) for m in messages),
                                    raw_b_bytes_per_client=state_nbytes(coordinator.base))
            inverse = {node: c for c, node in mapping.items()}
            remaining = max(0, coordinator.deadline_unix - time.time())
            for message, received_at in _collect(grid, messages, remaining):
                client_id = inverse.get(message.metadata.src_node_id, "unknown")
                coordinator.record_transport_reply(client_id, serialized_message_bytes(message))
                if message.has_error():
                    coordinator.reject(client_id, f"flower_error:{message.error.code}:{message.error.reason}")
                    continue
                try:
                    record = message.content["reply"]
                    reply = ClientReply(
                        RoundRequest.from_dict(json.loads(record["request"])),
                        _tensor_state(message.content["delta"]), record["status"],
                        json.loads(record["telemetry"]), serialized_message_bytes(message),
                    )
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    coordinator.reject(client_id, f"malformed_payload:{type(error).__name__}:{error}")
                    continue
                if coordinator.receive(reply, authenticated_client_id=client_id, received_at=received_at):
                    if client_id in previous_absent:
                        coordinator.events.emit("client_rejoined", client_id=client_id,
                                                attempt_id=coordinator.attempt_id, version=coordinator.version,
                                                synchronized_b_hash=reply.request.base_b_hash)
            result = coordinator.finish()
            evaluate_every = int(config.get("evaluate_every", 1))
            if (result["status"] == "committed" and evaluate_every > 0
                    and coordinator.version % evaluate_every == 0):
                evaluate_current()
            previous_absent = set(requests) - set(coordinator.accepted)
            print(json.dumps({k: v for k, v in result.items() if k != "clients"}), flush=True)
            pause = float(config.get("between_attempt_seconds", 0))
            if pause:
                time.sleep(pause)
        atomic_json(Path(config["run_dir"]) / "final_results.json", {
            "run_id": config["run_id"], "config_hash": coordinator.config_hash,
            "runtime": config.get("runtime", "unspecified"),
            "attempts": coordinator.attempt_id, "committed_versions": coordinator.version,
            "rounds": coordinator.results,
            "checkpoint": str(coordinator.store.path),
            "identity": envelope["identity"],
            "evaluations": evaluation_history,
            "total_wall_seconds": time.monotonic() - run_started,
        })
        coordinator.events.emit("run_completed", attempt_id=coordinator.attempt_id,
                                version=coordinator.version, checkpoint=str(coordinator.store.path))

    return app


client_app = make_client_app()
server_app = make_server_app()
