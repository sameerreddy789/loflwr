"""One supervised trainer subprocess per Ray actor, reused across logical clients.

This module has no Flower dependency. Deployment does not enable this cache:
five resident GPU models would violate the one-worker resource arrangement.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint import EventLog, read_progress
from .client_job import worker_profile
from .protocol import RoundRequest
from .state import state_from_dict


@dataclass
class Worker:
    process: subprocess.Popen
    profile: str
    log_path: Path
    stream: Any
    started_at: float
    jobs: int = 0


_worker: Worker | None = None
_lock = threading.Lock()


def stop_worker(*, kill: bool = False) -> None:
    """Caller owns the serial lease; also safe during actor interpreter shutdown."""
    global _worker
    worker, _worker = _worker, None
    if worker is None:
        return
    process = worker.process
    if process.poll() is None:
        if kill:
            process.kill()
        else:
            process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    if process.stdin:
        process.stdin.close()
    worker.stream.close()


atexit.register(stop_worker)


def _get_worker(config: dict, request: RoundRequest, log: EventLog) -> tuple[Worker, bool]:
    global _worker
    profile = worker_profile(config, request)
    if _worker is not None:
        if _worker.profile == profile and _worker.process.poll() is None:
            return _worker, True
        previous = _worker.process.pid
        stop_worker()
        log.emit("client_worker_replaced", client_id=request.client_id,
                 attempt_id=request.attempt_id, previous_pid=previous)
    log_dir = Path(config["run_dir"]) / "client_workers"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"actor_{os.getpid()}_{time.time_ns()}.log"
    stream = log_path.open("wb")
    started = time.time()
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "fedlora_edge.client_job", "--serve", "--parent-pid", str(os.getpid()),
             "--idle-timeout", str(config.get("worker_idle_timeout_seconds", 900))],
            stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                 "TOKENIZERS_PARALLELISM": "false",
                 "PYTHONPATH": str(Path(__file__).resolve().parent.parent) + os.pathsep
                 + os.environ.get("PYTHONPATH", "")},
        )
    except Exception:
        stream.close()
        raise
    _worker = Worker(process, profile, log_path, stream, started)
    return _worker, False


def run_persistent_job(request: RoundRequest, input_path: Path, output_path: Path,
                       progress_path: Path, config: dict, log: EventLog) -> dict:
    queued_at = time.monotonic()
    if not _lock.acquire(timeout=max(0, request.deadline_unix - time.time())):
        return {"status": "queue_deadline_expired", "delta": {}, "telemetry": {
            "optimizer_steps": 0, "persistent_queue_seconds": time.monotonic() - queued_at}}
    worker = None
    try:
        queued = time.monotonic() - queued_at
        started = time.monotonic()
        remaining = max(0, request.deadline_unix - time.time())
        if remaining <= 0:
            return {"status": "deadline_expired", "delta": {}, "telemetry": {"optimizer_steps": 0}}
        worker, reused = _get_worker(config, request, log)
        process = worker.process
        worker.jobs += 1
        fields = {"attempt_id": request.attempt_id, "version": request.version,
                  "client_id": request.client_id, "pid": process.pid,
                  "base_b_hash": request.base_b_hash, "job_dir": str(input_path.parent),
                  "worker_job_index": worker.jobs, "worker_log": str(worker.log_path)}
        log.emit("client_process_reused" if reused else "client_process_started", **fields)
        command = {"input": str(input_path), "output": str(output_path), "progress": str(progress_path)}
        assert process.stdin is not None
        process.stdin.write(json.dumps(command) + "\n")
        process.stdin.flush()
        interruption = next((entry for entry in config.get("process_interruptions", [])
                             if int(entry["attempt_id"]) == request.attempt_id
                             and str(entry["client_id"]) == request.client_id), None)
        progress: dict[str, Any] = {}
        status = "failed"
        while True:
            progress = read_progress(progress_path, progress)
            if (interruption is not None and progress.get("status") == "training"
                    and progress.get("optimizer_steps", 0) >= int(interruption["after_steps"])):
                stop_worker(kill=True)
                status = "process_interrupted"
                log.emit("client_process_interrupted", **fields, mechanism="OS process kill",
                         observed_progress=progress, returncode=process.returncode)
                break
            if time.monotonic() - started >= remaining:
                stop_worker()
                status = "deadline_expired"
                log.emit("client_process_deadline", **fields, observed_progress=progress,
                         returncode=process.returncode)
                break
            if output_path.exists():
                result = json.loads(output_path.read_text(encoding="utf-8"))
                result["delta"] = state_from_dict(result["delta"])
                result["telemetry"].update(
                    job_process_seconds=time.monotonic() - started, job_pid=process.pid,
                    worker_reused=reused, worker_job_index=worker.jobs,
                    worker_started_unix=worker.started_at, persistent_queue_seconds=queued,
                    process_log=str(worker.log_path),
                )
                if result["status"] != "completed":
                    stop_worker()
                return result
            if process.poll() is not None:
                stop_worker()
                break
            time.sleep(0.05)
        return {"status": status, "delta": {}, "telemetry": {
            **progress.get("telemetry", {}), "optimizer_steps": progress.get("optimizer_steps", 0),
            "job_process_seconds": time.monotonic() - started, "job_pid": process.pid,
            "job_returncode": process.returncode, "worker_reused": reused,
            "worker_job_index": worker.jobs, "persistent_queue_seconds": queued,
            "process_log": str(worker.log_path),
        }}
    except Exception:
        if worker is not None:
            stop_worker()
        raise
    finally:
        _lock.release()
