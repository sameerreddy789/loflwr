"""Export compact, reviewable measurements without weights or dataset queries.

Original evidence stays in runs/. Published paths are relative to the repository;
recorded configuration/source hashes still refer to the unchanged original files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedlora_edge.results import build_report, load_run  # noqa: E402

OMIT_KEYS = {
    "predictions", "records", "query", "queries", "text", "state", "b_state",
    "delta", "tensors", "a_schema", "b_schema", "node_map", "command",
}


def compact(value: Any) -> Any:
    """Remove known large/private payloads and normalize known workspace paths."""
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items() if key not in OMIT_KEYS}
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, str):
        for prefix in (str(ROOT).replace("\\", "/") + "/",
                       "/mnt/d/Hackathons/loflwr/", "D:/Hackathons/loflwr/"):
            normalized = value.replace("\\", "/")
            if normalized.lower().startswith(prefix.lower()):
                return normalized[len(prefix):]
    return value


def file_identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        name = str(path.resolve().relative_to(ROOT))
    except ValueError:
        name = path.name
    return {"file": name.replace("\\", "/"), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def export_run(path: Path) -> dict[str, Any]:
    run = load_run(path)
    if not run.sources:
        raise ValueError(f"No recognized evidence in {path}")
    report = build_report(run)
    # Use recorded checkpoint/config identity rather than hashing a transformed
    # public config. The reader's display hash uses a separate JSON convention.
    report["recorded_config_hash"] = run.checkpoint.get("config_hash")
    final_path = path / "final_results.json"
    if final_path.is_file():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        report["completion"] = {key: final[key] for key in (
            "run_id", "runtime", "kind", "attempts", "committed_versions",
            "config_hash", "total_wall_seconds",
        ) if key in final}
    report.pop("summary", None)  # Can contain redundant predictions/config copies.
    report["checkpoint"] = {key: run.checkpoint[key] for key in (
        "run_id", "attempt_id", "version", "b_hash", "config_hash", "identity",
    ) if key in run.checkpoint}
    report["recovery_events"] = [event for event in run.events if any(
        marker in str(event.get("event", ""))
        for marker in ("interrupt", "rejoin", "restarted", "recovered", "deadline")
    )]
    report["publication_note"] = (
        "Derived compact measurements. Tensor payloads, predictions and dataset queries "
        "are omitted; workspace paths normalized. Source/config hashes identify original "
        "local files, not this transformed report. Inspect limitations alongside results."
    )
    return compact(report)


def export_file(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(source, dict):
        raise ValueError("Single-file export expects a JSON object")
    return {"source_file": file_identity(path), "measurements": compact(source),
            "publication_note": "Compact copy; tensor payloads and dataset queries omitted."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.source.resolve() == args.output.resolve():
        parser.error("The export must not overwrite original evidence.")
    if args.output.exists():
        parser.error("Output already exists; use a new name so published evidence is explicit.")
    result = export_run(args.source) if args.source.is_dir() else export_file(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
                           + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
