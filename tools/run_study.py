"""Freeze and execute a sequential FedLoRA study; incomplete runs stay visible.

This launcher imports only standard-library modules until an actual model worker
starts. All process output and checkpoints remain under the chosen study folder.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.metadata
import json
import os
import platform
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HETERO_BUDGETS = [2, 16, 8, 4, 2]
CONDITIONS = {
    "balanced-equal": ("balanced", [8] * 5, False),
    "balanced-heterogeneous": ("balanced", HETERO_BUDGETS, False),
    "imbalanced-heterogeneous": ("imbalanced", HETERO_BUDGETS, False),
    "imbalanced-dropout": ("imbalanced", HETERO_BUDGETS, True),
}
DEFAULT_TRAINING = {
    "learning_rate": 2e-4, "weight_decay": 0.0, "microbatch_size": 1,
    "gradient_accumulation": 4, "max_length": 128, "gradient_clip": 1.0,
    "proximal_mu": 0.0, "activation_checkpointing": False,
    "cpu_threads": 4, "eval_batch_size": 20,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    # Match the coordinator/checkpoint canonical JSON convention, including
    # ASCII escaping, so non-ASCII local paths keep the same contract hash.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def emit(run_dir: Path, run_id: str, event: str, **values: Any) -> dict[str, Any]:
    record = {"event": event, "run_id": run_id, "timestamp": utc_now(), **values}
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
    print(json.dumps({key: value for key, value in record.items() if key != "metrics"},
                     ensure_ascii=False), flush=True)
    return record


def study_specs(mode: str) -> list[dict[str, Any]]:
    """Return the declared matrix, in its fixed execution order."""
    def spec(key: str, kind: str, strategy: str, condition: str, seed: int = 42,
             omission: bool = False) -> dict[str, Any]:
        scenario, budgets, dropout = CONDITIONS[condition]
        return {"key": key, "kind": kind, "strategy": strategy, "condition": condition,
                "seed": seed, "scenario": scenario, "budgets": list(budgets),
                "dropout": dropout, "omission": omission}

    specs = [spec("base", "frozen_base", "frozen_base", "imbalanced-heterogeneous")]
    if mode == "pilot":
        specs.append(spec("centralized", "centralized", "centralized", "imbalanced-heterogeneous"))
        for key, strategy in (("sample", "sample"), ("equal", "equal"), ("main", "equal_clipped")):
            specs.append(spec(key, "federated", strategy, "imbalanced-heterogeneous"))
        specs.append(spec("dropout", "federated", "equal_clipped", "imbalanced-dropout"))
        specs.append(spec("omit4", "federated", "equal_clipped", "imbalanced-heterogeneous", omission=True))
        return specs
    if mode != "full":
        raise ValueError(f"Unknown study mode: {mode}")
    for seed in (42, 43, 44):
        for condition in CONDITIONS:
            if condition != "imbalanced-dropout":
                specs.append(spec(f"centralized-{condition}-s{seed}", "centralized", "centralized", condition, seed))
            for strategy in ("sample", "equal_clipped"):
                specs.append(spec(f"{strategy}-{condition}-s{seed}", "federated", strategy, condition, seed))
            if condition == "imbalanced-heterogeneous":
                specs.append(spec(f"equal-{condition}-s{seed}", "federated", "equal", condition, seed))
        specs.append(spec(f"omit4-s{seed}", "federated", "equal_clipped", "imbalanced-heterogeneous", seed, omission=True))
    return specs


def select_keys(specs: list[dict[str, Any]], selection: str | None) -> list[str]:
    keys = [spec["key"] for spec in specs]
    if not selection:
        return keys
    patterns = [pattern.strip() for pattern in selection.split(",") if pattern.strip()]
    selected = [key for key in keys if any(fnmatch.fnmatchcase(key, pattern) for pattern in patterns)]
    unmatched = [pattern for pattern in patterns if not any(fnmatch.fnmatchcase(key, pattern) for key in keys)]
    if unmatched:
        raise ValueError(f"Unknown run selector(s): {', '.join(unmatched)}. Use --list for run keys.")
    if not selected:
        raise ValueError("No runs selected.")
    return selected


def environment_record() -> dict[str, Any]:
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "packages": dict(sorted(packages.items()))}


def freeze_plan(args: argparse.Namespace) -> dict[str, Any]:
    from fedlora_edge.data import verify_manifests

    study_dir = args.study_dir.expanduser().resolve()
    if study_dir.exists() and any(study_dir.iterdir()):
        raise FileExistsError(f"Study directory is not empty: {study_dir}. Use --resume or a new directory.")
    if args.attempts <= 0:
        raise ValueError("Attempts must be positive; the frozen-base reference has its own zero-step horizon.")
    data_dir = args.data_dir.expanduser().resolve()
    asset_dir = args.asset_dir.expanduser().resolve()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    verified = verify_manifests(data_dir)
    initial_path = adapter_dir / "initial_b.json"
    if not initial_path.is_file():
        raise FileNotFoundError(f"Prepare model/common-A assets first; missing {initial_path}")
    initial = read_json(initial_path)
    identity = initial.get("identity")
    if not isinstance(identity, dict) or not identity.get("common_a_hash"):
        raise ValueError("Initial B is missing its common-A/model identity.")
    specs = study_specs(args.mode)
    selected = select_keys(specs, args.runs)
    study_id = args.study_id or study_dir.name
    calibration = args.attempts != 20
    training = {**DEFAULT_TRAINING, "cpu_threads": args.cpu_threads,
                "eval_batch_size": args.eval_batch_size}
    sources = {str(path.relative_to(ROOT)): file_hash(path)
               for path in sorted((ROOT / "fedlora_edge").glob("*.py"))}
    sources["tools/run_study.py"] = file_hash(Path(__file__))
    if (ROOT / "pyproject.toml").is_file():
        sources["pyproject.toml"] = file_hash(ROOT / "pyproject.toml")
    manifest_hashes = {}
    for path in sorted((data_dir / "manifests").rglob("*.json")):
        manifest_hashes[str(path)] = file_hash(path)
    plan = {
        "format_version": 1, "study_id": study_id, "created_utc": utc_now(),
        "mode": args.mode, "scope": "CALIBRATION" if calibration else ("SEED42_PILOT" if args.mode == "pilot" else "FULL_STUDY"),
        "declared_attempts": args.attempts,
        "selection_rule": "Final committed B after the declared attempt horizon; skipped attempts count.",
        "primary_metric": "macro_f1",
        "final_test_access": "Disabled: calibration uses validation only." if calibration else "Only after the declared horizon for each run; no checkpoint selection by test results.",
        "comparison_limit": "Compare actual completed work and recorded failures; nominal budgets do not prove matched processed work.",
        "training_config": training, "identity": identity,
        "initial_b_file": str(initial_path), "initial_b_file_sha256": file_hash(initial_path),
        "initial_b_hash": initial.get("b_hash"), "manifest_set_hash": verified["manifest_set_hash"],
        "manifest_files_sha256": manifest_hashes, "source_files_sha256": sources,
        "environment": environment_record(), "initial_selection": selected, "runs": [],
    }
    for spec in specs:
        key = spec["key"]
        run_dir = study_dir / "runs" / key
        horizon = 0 if spec["kind"] == "frozen_base" else args.attempts
        config = {
            "run_id": f"{study_id}-{key}", "run_dir": str(run_dir),
            "asset_dir": str(asset_dir), "adapter_dir": str(adapter_dir),
            "client_manifests": {str(client): [str(data_dir / "manifests" / spec["scenario"] / f"client_{client}.json")]
                                 for client in range(5)},
            "budgets": {str(client): budget for client, budget in enumerate(spec["budgets"])},
            "attempts": horizon, "quorum": 3, "strategy": spec["strategy"], "seed": spec["seed"],
            "deadline_seconds": args.deadline_seconds, "device": args.device,
            "training_config": training, "worker_cpus": args.worker_cpus,
            "runtime": "flower_ray_single_worker_one_laptop" if spec["kind"] == "federated" else "local_reference_one_laptop",
            "persistent_worker": spec["kind"] == "federated",
            "process_interruptions": [{"attempt_id": attempt, "client_id": "1", "after_steps": 2}
                                      for attempt in (4, 8, 12) if attempt <= horizon] if spec["dropout"] else [],
            "omitted_clients": ["4"] if spec["omission"] else [], "resume": False,
            "scenario": spec["scenario"], "condition": spec["condition"],
            "kind": spec["kind"], "study_scope": plan["scope"], "study_id": study_id,
            "identity": identity, "initial_b_hash": initial.get("b_hash"),
            "manifest_set_hash": verified["manifest_set_hash"],
            "validation_manifest": str(data_dir / "manifests" / "validation.json"),
            "test_manifest": str(data_dir / "manifests" / "test.json"),
            "evaluate_every": 1, "evaluation_device": args.device, "evaluate_initial": True,
            "evaluate_final_test": not calibration,
            "selection_rule": plan["selection_rule"],
        }
        config_path = study_dir / "configs" / f"{key}.json"
        atomic_json(config_path, config)
        plan["runs"].append({**spec, "run_id": config["run_id"], "config_path": str(config_path),
                             "config_sha256": canonical_hash(config), "run_dir": str(run_dir), "attempts": horizon})
    plan["plan_sha256"] = canonical_hash(plan)
    atomic_json(study_dir / "study.json", plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    if canonical_hash({key: value for key, value in plan.items() if key != "plan_sha256"}) != plan.get("plan_sha256"):
        raise ValueError("Study plan hash mismatch; do not mutate a frozen study.")
    for run in plan["runs"]:
        if canonical_hash(read_json(Path(run["config_path"]))) != run["config_sha256"]:
            raise ValueError(f"Frozen configuration changed: {run['key']}")
    for name, expected in plan["manifest_files_sha256"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f"Frozen manifest changed: {name}")
    if file_hash(Path(plan["initial_b_file"])) != plan["initial_b_file_sha256"]:
        raise ValueError("Initial B checkpoint changed after the study was frozen.")
    for name, expected in plan["source_files_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Source code changed after freezing: {name}. Create a new study identity.")


def stop_process_group(process: subprocess.Popen) -> None:
    """Stop only this launcher's child process group, including model workers."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        # The group may still contain a Ray/model descendant after its parent exits.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_child(command: list[str], log_path: Path, deadline: float | None) -> dict[str, Any]:
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               bufsize=1, start_new_session=os.name != "nt", creationflags=flags)
    lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    timed_out = False
    interrupted = False
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] command: {json.dumps(command)}\n")
        log.flush()
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline and not timed_out:
                    timed_out = True
                    stop_process_group(process)
                try:
                    line = lines.get(timeout=0.2)
                except queue.Empty:
                    if process.poll() is not None and not reader.is_alive():
                        break
                    continue
                if line is None:
                    if process.poll() is not None:
                        break
                    continue
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            interrupted = True
            stop_process_group(process)
        finally:
            if process.poll() is None:
                stop_process_group(process)
            process.wait()
            while not lines.empty():
                line = lines.get_nowait()
                if line:
                    print(line, end="", flush=True)
                    log.write(line)
            log.flush()
    reader.join(timeout=2)
    if process.stdout is not None:
        process.stdout.close()
    return {"returncode": process.returncode, "timed_out": timed_out, "interrupted": interrupted,
            "wall_seconds": time.monotonic() - started, "log": str(log_path), "command": command}


def _evaluation(config: dict[str, Any], trainer: Any, split: str, attempt: int, version: int) -> dict[str, Any]:
    from fedlora_edge.data import load_manifest

    run_dir = Path(config["run_dir"])
    source = Path(config[f"{split}_manifest"])
    records = load_manifest(source)
    started = time.perf_counter()
    emit(run_dir, config["run_id"], "evaluation_started", split=split, attempt_id=attempt,
         version=version, count=len(records), manifest_file_sha256=file_hash(source))
    metrics = trainer.evaluate(records)
    record = {"run_id": config["run_id"], "timestamp": utc_now(), "split": split,
              "attempt_id": attempt, "version": version,
              "manifest_hash": read_json(source)["manifest_hash"], "metrics": metrics,
              "evaluation_wall_seconds": time.perf_counter() - started}
    # Individual files let interrupted studies distinguish completed tests from starts.
    if split == "test":
        atomic_json(run_dir / "test_metrics.json", record)
    with (run_dir / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    emit(run_dir, config["run_id"], "evaluation", **{key: value for key, value in record.items()
                                                     if key not in {"run_id", "timestamp"}})
    return record


def validate_run_checkpoint(config: dict[str, Any], checkpoint: dict[str, Any], run_dir: Path) -> None:
    if checkpoint.get("run_id") != config["run_id"]:
        raise ValueError("Checkpoint run identity differs from the frozen run configuration.")
    if config["kind"] == "federated":
        produced = read_json(run_dir / "config.json")
        expected_runtime = {key: value for key, value in config.items() if key != "resume"}
        if produced.get("runtime") != expected_runtime:
            raise ValueError("Federated runtime differs from the frozen configuration.")
        expected_hash = canonical_hash(produced)
    else:
        expected_hash = canonical_hash(config)
    if checkpoint.get("config_hash") != expected_hash:
        raise ValueError("Checkpoint configuration hash differs from this run's contract.")


def model_worker(config_path: Path, *, phase: str, resume: bool) -> int:
    from fedlora_edge.data import load_manifest
    from fedlora_edge.state import hash_state
    from fedlora_edge.training import LocalTrainer, TrainingConfig, load_b_state, save_b_state

    config = read_json(config_path)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.json"
    kind = config["kind"]
    if phase == "test":
        if not config.get("evaluate_final_test", False) or config.get("study_scope") == "CALIBRATION":
            raise ValueError("Final-test access is disabled for calibration. Use validation evidence.")
        if (run_dir / "test_metrics.json").is_file():
            print("Final test already recorded; preserving the existing result.", flush=True)
            return 0
        if not checkpoint_path.is_file():
            raise ValueError("Cannot evaluate final test without a saved checkpoint.")
        checkpoint = read_json(checkpoint_path)
        validate_run_checkpoint(config, checkpoint, run_dir)
        attempt = int(checkpoint.get("attempt_id", -1))
        if attempt != int(config["attempts"]):
            raise ValueError(f"Final test is gated on horizon {config['attempts']}; checkpoint attempt is {attempt}.")
        if kind != "frozen_base":
            results = read_json(run_dir / "round_results.json")
            expected_attempts = list(range(1, int(config["attempts"]) + 1))
            if [int(item["attempt_id"]) for item in results] != expected_attempts:
                raise ValueError("Declared attempt horizon has not finished.")
        version = int(checkpoint.get("version", 0))
        trainer = LocalTrainer(config["asset_dir"], config["adapter_dir"],
                               device=config["evaluation_device"], config=TrainingConfig(**config["training_config"]))
        trainer.validate_identity(config["identity"])
        trainer.set_b_state(load_b_state(checkpoint_path, trainer.identity))
        _evaluation(config, trainer, "test", attempt, version)
        return 0
    if kind not in {"frozen_base", "centralized"}:
        raise ValueError("Reference worker cannot replace a Flower experiment.")
    atomic_json(run_dir / "config.json", config)
    trainer = LocalTrainer(config["asset_dir"], config["adapter_dir"],
                           device=config["device"], config=TrainingConfig(**config["training_config"]))
    trainer.validate_identity(config["identity"])
    attempt, version = 0, 0
    recovered_record = None
    if resume and checkpoint_path.is_file():
        checkpoint = read_json(checkpoint_path)
        validate_run_checkpoint(config, checkpoint, run_dir)
        attempt, version = int(checkpoint["attempt_id"]), int(checkpoint["version"])
        state = load_b_state(checkpoint_path, trainer.identity)
        recovered_record = checkpoint.get("metadata")
    else:
        if checkpoint_path.exists():
            raise FileExistsError("Reference checkpoint already exists; use --resume.")
        state = load_b_state(Path(config["adapter_dir"]) / "initial_b.json", trainer.identity)
        save_b_state(checkpoint_path, state, trainer.identity, run_id=config["run_id"],
                     attempt_id=0, version=0, config_hash=canonical_hash(config))
    trainer.set_b_state(state)
    emit(run_dir, config["run_id"], "run_resumed" if resume else "run_started",
         kind=kind, attempt_id=attempt, version=version, config_hash=canonical_hash(config))
    evaluated_versions = set()
    evaluation_path = run_dir / "evaluations.jsonl"
    if evaluation_path.is_file():
        for line in evaluation_path.read_text(encoding="utf-8").splitlines():
            try:
                observed = json.loads(line)
            except ValueError:
                continue
            if observed.get("split") == "validation" and observed.get("metrics"):
                evaluated_versions.add(observed.get("version"))
    if version not in evaluated_versions:
        _evaluation(config, trainer, "validation", attempt, version)
    results_path = run_dir / "round_results.json"
    results = read_json(results_path) if results_path.exists() else []
    if recovered_record and not any(item.get("attempt_id") == attempt for item in results):
        results.append(recovered_record)
        atomic_json(results_path, results)
        emit(run_dir, config["run_id"], "round_result", recovered_from_checkpoint=True,
             **{key: value for key, value in recovered_record.items() if key != "run_id"})
    clients = {int(client): [record for path in paths for record in load_manifest(path)]
               for client, paths in config["client_manifests"].items()}
    budgets = {int(client): steps for client, steps in config["budgets"].items()}
    for attempt in range(attempt + 1, int(config["attempts"]) + 1):
        started = time.perf_counter()
        emit(run_dir, config["run_id"], "attempt_started", attempt_id=attempt,
             version=version, selected=["centralized"], budgets={"centralized": sum(budgets.values())},
             base_b_hash=hash_state(state), client_budgets=config["budgets"])

        def progress(step: int, telemetry: dict[str, Any]) -> None:
            atomic_json(run_dir / "progress.json", {"attempt_id": attempt, "version": version,
                                                    "timestamp": utc_now(), "status": "training",
                                                    "optimizer_steps": step, "telemetry": dict(telemetry)})

        result = trainer.train_centralized(clients, state, budgets=budgets,
                                           run_seed=config["seed"], attempt_id=attempt,
                                           deadline_unix=time.time() + config["deadline_seconds"],
                                           progress_callback=progress)
        before = version
        completed = result["status"] == "completed"
        if completed:
            state, version = result["b_state"], version + 1
        else:
            trainer.set_b_state(state)
        record = {"run_id": config["run_id"], "attempt_id": attempt,
                  "version_before": before, "version_after": version,
                  "status": "committed" if completed else "skipped",
                  "accepted": ["centralized"] if completed else [], "absent": [], "rejected": [],
                  "collection_seconds": time.perf_counter() - started,
                  "checkpoint_hash": hash_state(state), "aggregation": None,
                  "telemetry": result["telemetry"], "kind": "centralized_reference_block"}
        save_b_state(checkpoint_path, state, trainer.identity, run_id=config["run_id"],
                     attempt_id=attempt, version=version, config_hash=canonical_hash(config), metadata=record)
        results.append(record)
        atomic_json(results_path, results)
        emit(run_dir, config["run_id"], "centralized_block_completed", attempt_id=attempt,
             version=version, status=result["status"], telemetry=result["telemetry"])
        emit(run_dir, config["run_id"], "round_result", **{key: value for key, value in record.items() if key != "run_id"})
        if completed:
            _evaluation(config, trainer, "validation", attempt, version)
    emit(run_dir, config["run_id"], "run_completed", attempt_id=int(config["attempts"]),
         version=version, kind=kind)
    atomic_json(run_dir / "final_results.json", {"run_id": config["run_id"], "kind": kind,
                "attempts": int(config["attempts"]), "committed_versions": version,
                "identity": trainer.identity, "config_hash": canonical_hash(config), "rounds": results})
    return 0


def write_study_summary(study_dir: Path, plan: dict[str, Any], state: dict[str, Any]) -> None:
    from fedlora_edge.results import build_report, load_run

    summary = {"study_id": plan["study_id"], "scope": plan["scope"],
               "plan_sha256": plan["plan_sha256"], "updated_utc": utc_now(),
               "declared_attempts": plan["declared_attempts"], "runs": [],
               "completion": "incomplete", "primary_metric": plan["primary_metric"]}
    for spec in plan["runs"]:
        status = state["runs"].get(spec["key"], {"status": "not_started"})
        entry = {"key": spec["key"], "kind": spec["kind"], "strategy": spec["strategy"],
                 "seed": spec["seed"], "condition": spec["condition"],
                 "config_sha256": spec["config_sha256"], "run_id": spec["run_id"], **status}
        run_dir = Path(spec["run_dir"])
        if run_dir.is_dir():
            evidence = load_run(run_dir)
            entry["evidence"] = {"attempts_recorded": len(evidence.rounds),
                                 "checkpoint": evidence.checkpoint,
                                 "accepted_reply_work": build_report(evidence)["accepted_reply_work"]}
            if (run_dir / "test_metrics.json").is_file():
                final = read_json(run_dir / "test_metrics.json")
                entry["test"] = {**final, "metrics": {key: value for key, value in final["metrics"].items()
                                                       if key != "predictions"}}
        summary["runs"].append(entry)
    if all(run["status"] == "completed" for run in summary["runs"]):
        summary["completion"] = "declared_matrix_completed"
    summary["completed_runs"] = sum(run["status"] == "completed" for run in summary["runs"])
    summary["declared_runs"] = len(summary["runs"])
    summary["limits"] = [
        "Completion records execution, not model improvement or superiority.",
        "All methods retain their measured results, including failures and unfavorable scores.",
        "Runs use one laptop; this is not multi-device deployment evidence.",
    ]
    atomic_json(study_dir / "summary.json", summary)


def run_study(args: argparse.Namespace) -> int:
    study_dir = args.study_dir.expanduser().resolve()
    if args.resume:
        plan = read_json(study_dir / "study.json")
    else:
        plan = freeze_plan(args)
    validate_plan(plan)
    selected = select_keys(plan["runs"], args.runs) if args.runs else plan["initial_selection"]
    print(f"{plan['scope']}: {plan['study_id']} · {len(selected)} selected / {len(plan['runs'])} declared runs", flush=True)
    print(f"Frozen plan: {study_dir / 'study.json'}", flush=True)
    if args.plan_only:
        return 0
    state_path = study_dir / "status.json"
    state = read_json(state_path) if state_path.exists() else {"study_id": plan["study_id"], "runs": {}}
    deadline = time.monotonic() + args.max_seconds if args.max_seconds is not None else None
    failure = False
    for spec in plan["runs"]:
        key = spec["key"]
        if key not in selected:
            state["runs"].setdefault(key, {"status": "not_selected"})
            continue
        previous = state["runs"].get(key, {})
        if previous.get("status") == "completed":
            print(f"[{key}] already complete; preserving recorded result.", flush=True)
            continue
        if deadline is not None and time.monotonic() >= deadline:
            state["runs"][key] = {**previous, "status": "not_started_in_budget"}
            failure = True
            continue
        run_dir = Path(spec["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        config = read_json(Path(spec["config_path"]))
        resume_run = (run_dir / "checkpoint.json").is_file()
        phase_results = []
        history = list(previous.get("execution_history", []))
        if previous and previous.get("status") not in {"not_started", "not_selected", "not_started_in_budget"}:
            history.append({field: value for field, value in previous.items() if field != "execution_history"})
        state["runs"][key] = {"status": "running", "started_utc": utc_now(),
                              "phases": phase_results, "execution_history": history}
        atomic_json(state_path, state)
        write_study_summary(study_dir, plan, state)
        try:
            training_done = (run_dir / "final_results.json").is_file()
            if not training_done:
                if spec["kind"] == "federated":
                    execution = {**config, "resume": resume_run}
                    execution_path = run_dir / "execution_config.json"
                    atomic_json(execution_path, execution)
                    command = [sys.executable, "-m", "fedlora_edge.federation", "simulate", "--config", str(execution_path)]
                else:
                    command = [sys.executable, str(Path(__file__).resolve()), "_worker", "--config", spec["config_path"], "--phase", "reference"]
                    if resume_run:
                        command.append("--resume")
                print(f"[{key}] starting training/reference phase", flush=True)
                phase_result = {"phase": "training", **run_child(command, run_dir / "stdout.log", deadline)}
                phase_results.append(phase_result)
                atomic_json(state_path, state)
                if phase_result["returncode"] != 0 or phase_result["timed_out"] or phase_result["interrupted"]:
                    raise RuntimeError("Training/reference phase did not finish; inspect stdout.log and saved progress.")
                if not (run_dir / "final_results.json").is_file():
                    raise RuntimeError("Training process exited without a completed final_results.json.")
            if config.get("evaluate_final_test"):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Budget ended before final-test evaluation.")
                print(f"[{key}] declared horizon finished; evaluating final committed checkpoint", flush=True)
                command = [sys.executable, str(Path(__file__).resolve()), "_worker", "--config", spec["config_path"], "--phase", "test"]
                phase_result = {"phase": "final_test", **run_child(command, run_dir / "stdout.log", deadline)}
                phase_results.append(phase_result)
                if phase_result["returncode"] != 0 or phase_result["timed_out"] or phase_result["interrupted"]:
                    raise RuntimeError("Final-test phase did not finish; no completed test is claimed.")
                if not (run_dir / "test_metrics.json").is_file():
                    raise RuntimeError("Final-test process exited without test_metrics.json.")
            else:
                print(f"[{key}] calibration horizon finished; final test remains untouched.", flush=True)
            state["runs"][key].update(status="completed", finished_utc=utc_now())
        except Exception as exc:
            failure = True
            status = "budget_exhausted" if deadline is not None and time.monotonic() >= deadline else "failed"
            state["runs"][key].update(status=status, error=f"{type(exc).__name__}: {exc}", finished_utc=utc_now())
            print(f"[{key}] {status}: {exc}", file=sys.stderr, flush=True)
        atomic_json(state_path, state)
        write_study_summary(study_dir, plan, state)
        if any(phase.get("interrupted") for phase in phase_results):
            failure = True
            break
    atomic_json(state_path, state)
    write_study_summary(study_dir, plan, state)
    print(f"Study status: {study_dir / 'summary.json'}", flush=True)
    return 1 if failure else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_worker":
        parser = argparse.ArgumentParser(description="Internal isolated model worker")
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--phase", choices=("reference", "test"), required=True)
        parser.add_argument("--resume", action="store_true")
        args = parser.parse_args(argv[1:])
        return model_worker(args.config, phase=args.phase, resume=args.resume)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, help="New output directory under runs, or existing directory with --resume")
    parser.add_argument("--study-id", help="Defaults to the study directory name")
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--runs", help="Comma-separated run keys or glob patterns; fixed plan order is retained")
    parser.add_argument("--list", action="store_true", help="List the declared run keys without loading assets")
    parser.add_argument("--attempts", type=int, default=20, help="Anything other than 20 labels the study CALIBRATION and disables final-test access")
    parser.add_argument("--asset-dir", type=Path, default=ROOT / "assets/model")
    parser.add_argument("--adapter-dir", type=Path, default=ROOT / "assets/adapter")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "assets/banking77")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deadline-seconds", type=float, default=300)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--worker-cpus", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, help="Overall execution budget; stops owned child process groups and preserves progress")
    parser.add_argument("--plan-only", action="store_true", help="Freeze configurations and identities without starting experiments")
    parser.add_argument("--resume", action="store_true", help="Use an unchanged existing plan; retry incomplete runs and skip completed runs")
    args = parser.parse_args(argv)
    if args.list:
        for spec in study_specs(args.mode):
            print(f"{spec['key']:<47} {spec['kind']:<13} {spec['strategy']:<14} seed={spec['seed']} budgets={spec['budgets']}")
        return 0
    if args.study_dir is None:
        parser.error("--study-dir is required unless --list is used")
    if min(args.deadline_seconds, args.cpu_threads, args.worker_cpus, args.eval_batch_size) <= 0:
        parser.error("Deadlines, thread counts, worker CPUs, and evaluation batch size must be positive")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    try:
        return run_study(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Study could not start: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
