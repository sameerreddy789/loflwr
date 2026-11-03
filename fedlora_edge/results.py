"""Read saved experiment evidence without importing the model or dashboard stack.

Missing measurements remain missing. JSONL files may be read while a writer is
appending; an incomplete last record is reported and omitted until the next read.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


KNOWN_FILES = (
    "config.json", "summary.json", "events.jsonl", "round_results.json",
    "evaluations.jsonl", "evaluation.json", "metrics.json", "results.json",
    "checkpoint.json", "final_results.json", "test_metrics.json",
)
QUALITY_KEYS = (
    "accuracy", "macro_f1", "task_loss", "validation_loss", "loss",
    "worst_domain_accuracy", "domain_gap", "per_domain_accuracy",
)
CLIENT_DOMAINS = {
    "0": "Card lifecycle", "1": "Card payments", "2": "Transfers",
    "3": "Cash withdrawals", "4": "Account verification",
}


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(value) else None
    return None


def first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def compact_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _ids(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [str(key) for key in value]
    if isinstance(value, (list, tuple)):
        return [str(item.get("client_id", item.get("id", "?")))
                if isinstance(item, Mapping) else str(item) for item in value]
    return []


@dataclass
class RunEvidence:
    path: Path
    config: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return str(first(self.config, "run_id", "id")
                   or self.summary.get("run_id") or self.path.name)

    @property
    def config_hash(self) -> str | None:
        return compact_hash(self.config) if self.config else None

    @property
    def latest_round(self) -> dict[str, Any]:
        return self.rounds[-1] if self.rounds else {}

    @property
    def latest_evaluation(self) -> dict[str, Any]:
        return self.evaluations[-1] if self.evaluations else {}

    @property
    def evidence_label(self) -> str:
        kind = str(first(self.config, "kind", "experiment_type", "mode") or "").lower()
        if "toy" in kind or "protocol" in kind:
            return "Protocol check"
        if self.evaluations:
            return "Model evaluation recorded"
        if any(event.get("telemetry") for event in self.events):
            return "Training telemetry only"
        return "No model evaluation recorded"


def _read_file(path: Path, run: RunEvidence) -> Any:
    try:
        raw = path.read_bytes()
        stat = path.stat()
    except OSError as exc:
        run.warnings.append(f"{path.name}: could not read ({exc}).")
        return None
    run.sources.append({
        "file": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    })
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        run.warnings.append(f"{path.name}: file is not valid UTF-8.")
        return None
    if path.suffix == ".jsonl":
        records = []
        lines = text.splitlines()
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("expected an object")
                records.append(value)
            except (ValueError, json.JSONDecodeError):
                tail = " Waiting for the writer to finish." if index == len(lines) else ""
                run.warnings.append(f"{path.name}:{index}: invalid record omitted.{tail}")
        return records
    try:
        return json.loads(text)
    except (ValueError, json.JSONDecodeError) as exc:
        run.warnings.append(f"{path.name}: invalid JSON; refresh after the write finishes ({exc}).")
        return None


def _evaluation(value: Any, source: str) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    metrics = value.get("metrics", value)
    if not isinstance(metrics, Mapping) or not any(key in metrics for key in QUALITY_KEYS):
        return None
    # Predictions can be large; the original source file remains available locally.
    return {
        "source": source,
        "split": first(value, "split", "dataset_split", "stage", "phase") or "unspecified",
        "attempt_id": first(value, "attempt_id", "attempt"),
        "version": first(value, "version", "version_after"),
        "timestamp": value.get("timestamp"),
        "metrics": {key: val for key, val in metrics.items()
                    if key not in {"predictions", "confusion_matrix", "records"}},
    }


def load_run(path: str | Path, event_limit: int | None = None) -> RunEvidence:
    run = RunEvidence(Path(path).expanduser().resolve())
    payloads = {}
    for filename in KNOWN_FILES:
        file_path = run.path / filename
        if file_path.is_file():
            payloads[filename] = _read_file(file_path, run)
    for name, target in (("config.json", "config"), ("summary.json", "summary")):
        value = payloads.get(name)
        if isinstance(value, dict):
            setattr(run, target, value)
    events = payloads.get("events.jsonl")
    if isinstance(events, list):
        run.events = events[:event_limit] if event_limit is not None else events
    checkpoint = payloads.get("checkpoint.json")
    if isinstance(checkpoint, dict) and event_limit is None:
        run.checkpoint = {key: value for key, value in checkpoint.items()
                          if key not in {"state", "b_state"}}
    # One final round per attempt, even if the writer also saves an aggregate file.
    round_map: dict[str, dict[str, Any]] = {}
    saved_rounds = payloads.get("round_results.json")
    if event_limit is None and isinstance(saved_rounds, list):
        for index, item in enumerate(saved_rounds):
            if isinstance(item, dict):
                round_map[str(item.get("attempt_id", f"saved-{index}"))] = item
    for index, event in enumerate(run.events):
        if event.get("event") in {"round_result", "attempt_result"}:
            round_map[str(event.get("attempt_id", f"event-{index}"))] = event
        if event.get("event") in {"evaluation", "validation", "test_evaluation"}:
            item = _evaluation(event, f"events.jsonl:{index + 1}")
            if item:
                run.evaluations.append(item)
    run.rounds = list(round_map.values())
    # During replay, a future summary must never leak into the visible history.
    if event_limit is None:
        for name in ("evaluations.jsonl", "evaluation.json", "metrics.json", "results.json", "test_metrics.json"):
            values = payloads.get(name)
            if values is None:
                continue
            for value in values if isinstance(values, list) else [values]:
                item = _evaluation(value, name)
                if item:
                    run.evaluations.append(item)
        for name in ("baseline", "frozen_base", "validation", "final", "test", "evaluation"):
            item = _evaluation(run.summary.get(name), f"summary.json:{name}")
            if item:
                if item["split"] == "unspecified":
                    item["split"] = name
                run.evaluations.append(item)
        item = _evaluation(run.summary, "summary.json")
        if item:
            run.evaluations.append(item)
    else:
        run.summary = {}
    unique_evaluations: dict[str, dict[str, Any]] = {}
    for evaluation in run.evaluations:
        signature = compact_hash({key: evaluation.get(key) for key in ("split", "attempt_id", "version", "metrics")})
        if signature in unique_evaluations:
            unique_evaluations[signature].setdefault("also_recorded_in", []).append(evaluation["source"])
        else:
            unique_evaluations[signature] = evaluation
    run.evaluations = list(unique_evaluations.values())
    return run


def discover_runs(root: str | Path) -> list[Path]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        return []
    paths = set()
    for filename in ("config.json", "events.jsonl", "summary.json", "round_results.json"):
        for path in root_path.rglob(filename):
            if path.is_file():
                if filename == "summary.json":
                    try:
                        summary = json.loads(path.read_text(encoding="utf-8-sig"))
                    except (OSError, ValueError):
                        summary = {}
                    # A study's matrix is a collection of runs, not one model run.
                    if isinstance(summary, dict) and summary.get("study_id") and isinstance(summary.get("runs"), list):
                        continue
                paths.add(path.parent)
    # Newest first makes a currently running experiment easy to find.
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


def attempt_rows(run: RunEvidence) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for event in run.events:
        if event.get("event") == "attempt_started":
            key = str(event.get("attempt_id", "?"))
            rows[key] = {
                "Attempt": event.get("attempt_id"), "Status": "Collecting",
                "Version before": event.get("version"), "Version after": None,
                "Accepted": None, "Selected": len(_ids(event.get("selected"))),
                "Collection seconds": None, "Checkpoint hash": None,
            }
    for result in run.rounds:
        key = str(result.get("attempt_id", "?"))
        rows[key] = {
            **rows.get(key, {}), "Attempt": result.get("attempt_id"),
            "Status": str(result.get("status", "recorded")).capitalize(),
            "Version before": result.get("version_before"),
            "Version after": first(result, "version_after", "version"),
            "Accepted": len(_ids(result.get("accepted"))),
            "Collection seconds": result.get("collection_seconds"),
            "Checkpoint hash": result.get("checkpoint_hash"),
        }
    return list(rows.values())


def _client_config(run: RunEvidence, client_id: str) -> dict[str, Any]:
    clients = run.config.get("clients", [])
    if isinstance(clients, Mapping):
        value = clients.get(client_id, clients.get(int(client_id), {}) if client_id.isdigit() else {})
        return dict(value) if isinstance(value, Mapping) else {"num_examples": value}
    for value in clients if isinstance(clients, list) else []:
        if isinstance(value, Mapping) and str(first(value, "client_id", "id")) == client_id:
            return dict(value)
    return {}


def client_rows(run: RunEvidence, attempt_id: Any = None) -> list[dict[str, Any]]:
    attempts = attempt_rows(run)
    if attempt_id is None:
        attempt_id = attempts[-1].get("Attempt") if attempts else None
    events = [event for event in run.events if str(event.get("attempt_id")) == str(attempt_id)]
    result = next((item for item in reversed(run.rounds)
                   if str(item.get("attempt_id")) == str(attempt_id)), {})
    start = next((event for event in events if event.get("event") == "attempt_started"), {})
    client_ids = set(_ids(run.config.get("clients"))) | set(_ids(start.get("selected")))
    for event in events:
        if "client_id" in event:
            client_ids.add(str(event["client_id"]))
    for key in ("accepted", "absent", "rejected"):
        client_ids.update(_ids(result.get(key)))
    aggregation = result.get("aggregation") or {}
    rows = []
    for client_id in sorted(client_ids, key=lambda val: (not val.isdigit(), int(val) if val.isdigit() else val)):
        own = [event for event in events if str(event.get("client_id")) == client_id]
        accepted = next((event for event in reversed(own)
                         if event.get("event") == "reply_accepted"), {})
        rejected = next((event for event in reversed(own)
                         if event.get("event") == "reply_rejected"), {})
        telemetry = accepted.get("telemetry") or (result.get("clients") or {}).get(client_id) or {}
        config = _client_config(run, client_id)
        status = "Selected" if client_id in _ids(start.get("selected")) else "Not selected"
        process_event = next((event for event in reversed(own)
                              if event.get("event") in {
                                  "client_process_started", "client_synchronized",
                                  "client_process_interrupted", "client_process_deadline",
                              }), {})
        if process_event:
            status = {
                "client_process_started": "Process started",
                "client_synchronized": "Synchronized",
                "client_process_interrupted": "Process interrupted",
                "client_process_deadline": "Process stopped at deadline",
            }[process_event["event"]]
            progress = process_event.get("observed_progress") or {}
            if not telemetry and isinstance(progress, Mapping):
                telemetry = dict(progress.get("telemetry") or {})
                if "optimizer_steps" in progress:
                    telemetry["optimizer_steps"] = progress["optimizer_steps"]
        if client_id in _ids(result.get("accepted")) or accepted:
            status = "Accepted"
        elif client_id in _ids(result.get("absent")):
            status = "Absent at deadline"
        elif rejected or client_id in _ids(result.get("rejected")):
            status = "Rejected"
        reason = rejected.get("reason")
        if reason is None:
            for item in result.get("rejected", []):
                if isinstance(item, Mapping) and str(item.get("client_id")) == client_id:
                    reason = item.get("reason")
        def influence(name: str) -> Any:
            values = aggregation.get(name) or {}
            return values.get(client_id) if isinstance(values, Mapping) else None
        budget = (start.get("budgets") or {}).get(client_id)
        if budget is None:
            runtime = run.config.get("runtime") or {}
            if isinstance(runtime, Mapping):
                budget = (runtime.get("budgets") or {}).get(client_id)
        rows.append({
            "Client": client_id, "Domain": config.get("domain", CLIENT_DOMAINS.get(client_id)),
            "Status": status, "Reason": reason,
            "Process event": process_event.get("event"), "Process ID": process_event.get("pid"),
            "Synchronized version": process_event.get("version"),
            "Rejoined": any(event.get("event") == "client_rejoined" for event in own),
            "Declared examples": first(config, "num_examples", "sample_count", "train_size", "examples"),
            "Step budget": budget if budget is not None else first(config, "step_budget", "max_steps", "optimizer_steps"),
            "Completed steps": first(telemetry, "optimizer_steps", "completed_optimizer_steps", "completed_steps", "steps"),
            "Processed examples": first(telemetry, "examples", "processed_examples", "examples_processed"),
            "Processed tokens": first(telemetry, "tokens", "processed_tokens", "tokens_processed"),
            "Training seconds": first(telemetry, "local_train_seconds", "elapsed_seconds", "training_seconds", "train_seconds"),
            "Queue seconds": telemetry.get("queue_wait_seconds"),
            "Peak RSS bytes": first(telemetry, "peak_rss_bytes", "peak_process_rss_bytes", "rss_bytes"),
            "Peak GPU allocated bytes": telemetry.get("peak_gpu_allocated_bytes"),
            "Peak GPU reserved bytes": telemetry.get("peak_gpu_reserved_bytes"),
            "Raw delta bytes": accepted.get("raw_delta_bytes"),
            "Serialized reply bytes": accepted.get("serialized_reply_bytes"),
            "Coefficient": influence("coefficients"), "Delta norm": influence("norms"),
            "Clip factor": influence("clip_factors"), "Clipped norm": influence("clipped_norms"),
        })
    return rows


def work_totals(run: RunEvidence) -> dict[str, Any]:
    replies: dict[tuple[str, str], dict[str, Any]] = {}
    for event in run.events:
        if event.get("event") in {"reply_accepted", "centralized_block_completed"}:
            client_id = event.get("client_id", "centralized")
            replies[(str(event.get("attempt_id")), str(client_id))] = event
    totals = {}
    for target, keys in {
        "optimizer_steps": ("optimizer_steps", "completed_optimizer_steps", "completed_steps", "steps"),
        "processed_examples": ("examples", "processed_examples", "examples_processed"),
        "processed_tokens": ("tokens", "processed_tokens", "tokens_processed"),
        "reported_training_seconds": ("local_train_seconds", "elapsed_seconds", "training_seconds", "train_seconds"),
    }.items():
        values = [number(first(event.get("telemetry") or {}, *keys)) for event in replies.values()]
        measured = [value for value in values if value is not None]
        totals[target] = sum(measured) if measured else None
        totals[f"{target}_coverage"] = {"measured_replies": len(measured), "accepted_replies": len(values)}
    for key in ("raw_delta_bytes", "serialized_reply_bytes"):
        values = [number(event.get(key)) for event in replies.values()]
        measured = [value for value in values if value is not None]
        totals[key] = sum(measured) if measured else None
        totals[f"{key}_coverage"] = {"measured_replies": len(measured), "accepted_replies": len(values)}
    received = [event for event in run.events if event.get("event") == "reply_received"]
    if received:
        values = [number(event.get("serialized_reply_bytes")) for event in received]
        measured = [value for value in values if value is not None]
        totals["serialized_reply_bytes"] = sum(measured) if measured else None
        totals["serialized_reply_bytes_coverage"] = {
            "measured_replies": len(measured), "received_replies": len(values),
        }
    totals["serialized_reply_bytes_scope"] = "All received replies" if received else "Accepted replies only (legacy event stream)"
    totals["scope"] = "Accepted federated replies or reported centralized blocks; partial centralized work is included. Missing telemetry is not zero."
    request_events = [event for event in run.events if event.get("event") == "requests_sent"]
    request_sizes = [number(event.get("serialized_request_bytes")) for event in request_events]
    measured_requests = [value for value in request_sizes if value is not None]
    totals["serialized_request_bytes"] = sum(measured_requests) if measured_requests else None
    totals["serialized_request_bytes_coverage"] = {
        "measured_dispatches": len(measured_requests), "recorded_dispatches": len(request_events),
    }
    return totals


def build_report(run: RunEvidence) -> dict[str, Any]:
    return {
        "report_format_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run.run_id, "run_directory": str(run.path),
        "evidence_label": run.evidence_label,
        "config_sha256": run.config_hash, "configuration": run.config,
        "summary": run.summary, "checkpoint": run.checkpoint,
        "attempts": attempt_rows(run), "accepted_reply_work": work_totals(run),
        "evaluations": run.evaluations,
        "round_details": run.rounds, "source_files": run.sources, "read_warnings": run.warnings,
        "interpretation": [
            "Recorded measurements only; missing fields remain null.",
            "A protocol check is not a language-model experiment.",
            "This report does not establish matched comparisons, superiority, or physical deployment.",
        ],
    }


def _cell(value: Any) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def report_markdown(report: Mapping[str, Any]) -> str:
    config = report.get("configuration") or {}
    lines = [
        f"# Run report: {_cell(report.get('run_id'))}", "",
        f"Evidence: **{_cell(report.get('evidence_label'))}**. Generated {_cell(report.get('generated_utc'))}.",
        "", "## Identity", "", "| Field | Recorded value |", "| --- | --- |",
        f"| Run ID | {_cell(report.get('run_id'))} |",
        f"| Configuration SHA-256 | {_cell(report.get('config_sha256'))} |",
        f"| Strategy | {_cell(config.get('strategy'))} |",
        f"| Seed | {_cell(config.get('seed'))} |",
        f"| Deadline seconds | {_cell(first(config, 'deadline_seconds', 'collection_timeout'))} |",
        f"| Quorum | {_cell(config.get('quorum'))} |",
        "", "## Task measurements", "",
    ]
    evaluations = report.get("evaluations") or []
    if not evaluations:
        lines.append("No model evaluation was recorded. No quality or learning claim is supported.")
    else:
        lines += ["| Source | Split | Version | Count | Accuracy | Macro F1 | Task loss | Worst domain | Domain gap |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for item in evaluations:
            metrics = item.get("metrics") or {}
            row = [item.get("source"), item.get("split"), item.get("version"), metrics.get("count"),
                   metrics.get("accuracy"), metrics.get("macro_f1"),
                   first(metrics, "task_loss", "validation_loss", "loss"),
                   metrics.get("worst_domain_accuracy"), metrics.get("domain_gap")]
            lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
        for item in evaluations:
            domains = item.get("metrics", {}).get("per_domain_accuracy")
            if isinstance(domains, Mapping):
                lines += ["", f"Per-domain accuracy ({item['source']}, {item['split']}):", "",
                          "| Domain | Accuracy |", "| --- | --- |"]
                lines += [f"| {_cell(domain)} | {_cell(value)} |" for domain, value in domains.items()]
    lines += ["", "## Attempts", "", "| Attempt | Status | Version before | Version after | Accepted | Collection seconds |",
              "| --- | --- | --- | --- | --- | --- |"]
    for attempt in report.get("attempts") or []:
        lines.append("| " + " | ".join(_cell(attempt.get(key)) for key in (
            "Attempt", "Status", "Version before", "Version after", "Accepted", "Collection seconds")) + " |")
    if not report.get("attempts"):
        lines += ["", "No federated attempt records were found."]
    lines += ["", "## Accepted-reply work and traffic", "", "```json",
              json.dumps(report.get("accepted_reply_work"), indent=2, ensure_ascii=False), "```",
              "", "## Full configuration", "", "```json",
              json.dumps(config, indent=2, ensure_ascii=False), "```",
              "", "## Source file identities", "", "| File | Bytes | SHA-256 |", "| --- | --- | --- |"]
    for item in report.get("source_files") or []:
        lines.append(f"| {_cell(item.get('file'))} | {_cell(item.get('bytes'))} | {_cell(item.get('sha256'))} |")
    lines += ["", "## Limits", ""]
    lines += [f"- {item}" for item in report.get("interpretation") or []]
    lines += [f"- Reader warning: {item}" for item in report.get("read_warnings") or []]
    return "\n".join(lines) + "\n"
