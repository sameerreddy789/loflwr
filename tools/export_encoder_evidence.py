"""Export compact encoder evidence, excluding weights, examples and predictions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def export(run_dir: Path) -> dict:
    source = run_dir / "final_results.json"
    if not source.exists():
        source = run_dir / "summary.json"
    raw = source.read_bytes()
    result = json.loads(raw)
    keep = ("status", "kind", "runtime", "run_id", "config_hash", "identity", "plan",
            "selected", "final", "attempts", "committed_versions", "total_wall_seconds",
            "wall_seconds", "raw_update_bytes", "history", "evaluations", "limits",
            "checks", "passed", "frozen_before", "frozen_after", "initial_state_hash",
            "final_state_hash", "training", "training_batch_before", "training_batch_after",
            "total_seconds", "interpretation", "initial_parity", "method_count",
            "plan_file_sha256", "source_unchanged_during_run", "input_files_unchanged_during_run",
            "test_or_confirmation_scored")
    output = {k: result[k] for k in keep if k in result}
    output["run_directory"] = run_dir.as_posix()
    output["source_sha256"] = hashlib.sha256(raw).hexdigest()
    if isinstance(result.get("work"), list):
        work = result["work"]
        output["work_totals"] = {
            "client_jobs": len(work),
            "optimizer_steps": sum(row.get("optimizer_steps", 0) for row in work),
            "examples_processed": sum(row.get("examples_processed", 0) for row in work),
            "peak_worker_rss_bytes": max((row.get("peak_rss_bytes") or 0 for row in work), default=0),
            "local_train_seconds_sum": sum(row.get("local_train_seconds", 0) for row in work)}
    if "results" in result:
        output["methods"] = [{k: v for k, v in row.items() if k != "predictions"}
                             for row in result["results"]]
    if "rounds" in result:
        fields = ("attempt_id", "status", "accepted", "absent", "rejected", "version_before",
                  "version_after", "collection_seconds", "serialized_reply_bytes", "checkpoint_hash",
                  "quorum", "raw_b_bytes", "aggregation_seconds")
        output["rounds"] = []
        for row in result["rounds"]:
            small = {k: row[k] for k in fields if k in row}
            small["aggregation"] = row.get("aggregation")
            small["clients"] = {client: {k: data[k] for k in (
                "optimizer_steps", "examples_processed", "peak_rss_bytes", "raw_parameter_bytes",
                "state_hash_before", "state_hash_after", "encoder_identity_hash") if k in data}
                for client, data in row.get("clients", {}).items()}
            output["rounds"].append(small)
    output["supporting_files_sha256"] = {}
    for name in ("runtime.json", "runtime_preparation.json", "config.json", "events.jsonl", "plan.json"):
        path = run_dir / name
        if path.exists():
            output["supporting_files_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if name == "runtime.json":
                output["runtime_configuration"] = json.loads(path.read_bytes())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = export(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
