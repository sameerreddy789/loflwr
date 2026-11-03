"""Publish compact measured study evidence and figures without loading a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fedlora_edge.results import build_report, load_run, number  # noqa: E402
from tools.export_evidence import compact, export_run, file_identity  # noqa: E402

QUALITY = ("accuracy", "macro_f1", "worst_domain_accuracy", "domain_gap")
WORK = ("optimizer_steps", "processed_examples", "processed_tokens",
        "reported_training_seconds", "serialized_request_bytes", "serialized_reply_bytes")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#595959", "#56B4E9", "#8C6D00")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(compact(value), indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8")


def collect(study: Path, spec: dict, state: dict) -> tuple[dict, object]:
    key = spec["key"]
    if Path(key).name != key or key in {".", ".."}:
        raise ValueError("Run keys must be simple filenames")
    # Frozen studies have this layout; use local copies when reporting WSL runs on Windows.
    config_path, run_path = study / "configs" / f"{key}.json", study / "runs" / key
    config, run = read(config_path), load_run(run_path)
    report, final, test = build_report(run), read(run_path / "final_results.json"), read(run_path / "test_metrics.json")
    checkpoint, status = run.checkpoint, state.get("status", "not_started")
    issues = list(run.warnings)
    if not config or canonical(config) != spec["config_sha256"]:
        issues.append("Frozen configuration is missing or differs from the plan.")
    if status == "completed":
        runtime = {k: v for k, v in config.items() if k != "resume"}
        actual = run.config.get("runtime") if spec["kind"] == "federated" else run.config
        expected = runtime if spec["kind"] == "federated" else config
        horizon = spec["attempts"]
        if (actual != expected or checkpoint.get("run_id") != spec["run_id"]
                or checkpoint.get("identity") != config.get("identity")
                or checkpoint.get("config_hash") != canonical(run.config)
                or final.get("config_hash") != checkpoint.get("config_hash")
                or final.get("run_id") != spec["run_id"]
                or checkpoint.get("attempt_id") != horizon or final.get("attempts") != horizon
                or final.get("committed_versions") != checkpoint.get("version")
                or [r.get("attempt_id") for r in run.rounds] != list(range(1, horizon + 1))):
            issues.append("Completion/configuration/checkpoint/horizon evidence does not agree.")
        if config.get("evaluate_final_test") and (
                test.get("split") != "test" or test.get("run_id") != spec["run_id"]
                or test.get("attempt_id") != horizon or test.get("version") != checkpoint.get("version")
                or not checkpoint.get("b_hash") or test.get("metrics", {}).get("b_hash") != checkpoint.get("b_hash")
                or test.get("metrics", {}).get("contract_hash") != config.get("identity", {}).get("contract_hash")):
            issues.append("Final test is missing or does not identify the final checkpoint.")
        if issues:
            status = "incomplete_evidence"
    completed = status == "completed"
    metrics = test.get("metrics", {}) if completed and config.get("evaluate_final_test") else {}
    phases = [phase for attempt in [*state.get("execution_history", []), state]
              for phase in attempt.get("phases", [])]
    times = [number(phase.get("wall_seconds")) for phase in phases]
    work = report["accepted_reply_work"]
    row = {k: spec[k] for k in ("key", "run_id", "kind", "strategy", "condition", "seed", "attempts", "config_sha256")}
    row.update(status=status, recorded_status=state.get("status", "not_started"),
               attempts_recorded=len(run.rounds), version=checkpoint.get("version"),
               checkpoint_b_hash=checkpoint.get("b_hash"), recorded_config_hash=checkpoint.get("config_hash"),
               test_count=metrics.get("count"), **{k: number(metrics.get(k)) for k in QUALITY},
               test_identity={k: test.get(k) for k in ("run_id", "attempt_id", "version", "manifest_hash", "timestamp")} if metrics else {},
               per_domain_accuracy=metrics.get("per_domain_accuracy", {}),
               **{k: work.get(k) if completed else None for k in WORK},
               phase_wall_seconds=sum(times) if times and all(t is not None for t in times) else None,
               work_coverage=work if completed else {}, warnings=issues,
               configuration=config, configuration_file=file_identity(config_path) if config_path.is_file() else None)
    return row, run


def table(rows: list[dict], fields: list[str], path: Path | None = None) -> list[str]:
    if path is not None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    def cell(value: object) -> str:
        return "Not recorded" if value is None else (f"{value:.6g}" if isinstance(value, float)
                                                      else str(value).replace("|", "\\|").replace("\n", " "))
    return ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |",
            *["| " + " | ".join(cell(row.get(k)) for k in fields) + " |" for row in rows]]


def figures(rows: list[dict], runs: list, output: Path, label: str, study_id: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    for name in ("validation-task-loss", "final-test-macro-f1"):
        eligible = sum(row["status"] == "completed" and row["macro_f1"] is not None for row in rows)
        height = max(3.2, 0.52 * eligible + 1.7) if name == "final-test-macro-f1" else max(5.8, len(rows) * 0.19)
        fig, ax = plt.subplots(figsize=(10, height), layout="constrained")
        plotted = 0
        for i, (row, run) in enumerate(zip(rows, runs)):
            color = COLORS[i % len(COLORS)]
            legend = f"{row['key']} · seed {row['seed']} · {row['status']}"
            if name == "validation-task-loss":
                points = sorted({(e["attempt_id"], number(e["metrics"].get("task_loss")))
                                 for e in run.evaluations if e["split"] == "validation"
                                 and number(e["attempt_id"]) is not None
                                 and number(e["metrics"].get("task_loss")) is not None})
                if not points:
                    continue
                ax.plot(*zip(*points), label=legend, color=color, marker=("o", "s", "^", "D")[i % 4],
                        linestyle="-" if row["status"] == "completed" else "--", markersize=5)
            else:
                if row["status"] != "completed" or row["macro_f1"] is None:
                    continue
                ax.barh(legend, row["macro_f1"], color=color, height=0.6)
                ax.text(row["macro_f1"] + 0.012, plotted, f"{row['macro_f1']:.3f}", va="center")
            plotted += 1
        if name == "validation-task-loss":
            ax.set(xlabel="Attempt (skipped attempts retain their number)", ylabel="Validation task loss",
                   title=f"Validation measurements · {study_id}\n{label}")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlim(left=-0.25, right=max(1, max(row["attempts"] for row in rows)))
            if plotted:
                ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        else:
            ax.set(xlabel="Final-test macro F1", xlim=(0, 1.08), ylim=(max(plotted - 0.2, 0.8), -0.8),
                   title=f"Completed runs only · {study_id}\n{label}")
            ax.xaxis.set_major_formatter(PercentFormatter(1))
        if not plotted:
            ax.text(0.5, 0.5, "No eligible measured results", transform=ax.transAxes, ha="center")
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        fig.supxlabel("Individual runs; no confidence intervals. No matched-comparison or deployment claim.", fontsize=9)
        for suffix in ("png", "svg"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--note", default="", help="Context included in the report; e.g. why a study was superseded")
    args = parser.parse_args(argv)
    study, output = args.study_dir.resolve(), args.output_dir.resolve()
    plan, state = read(study / "study.json"), read(study / "status.json")
    if not plan.get("runs") or canonical({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan.get("plan_sha256"):
        parser.error("Missing or changed frozen study plan")
    if state and state.get("study_id") != plan["study_id"]:
        parser.error("Status does not belong to this study")
    if output.exists() or output == study or study.is_relative_to(output):
        parser.error("Use a new output directory that cannot overwrite study evidence")
    pairs = [collect(study, spec, state.get("runs", {}).get(spec["key"], {
        "status": "not_started" if spec["key"] in plan["initial_selection"] else "not_selected"})) for spec in plan["runs"]]
    rows, runs = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    completed = sum(row["status"] == "completed" for row in rows)
    seeds = sorted({row["seed"] for row in rows})
    label = f"{'PARTIAL STUDY' if completed != len(rows) else 'DECLARED MATRIX COMPLETE'} · {completed}/{len(rows)} runs · "
    label += f"single seed {seeds[0]}" if len(seeds) == 1 else "individual seeds " + ", ".join(map(str, seeds))
    label += f" · {plan['scope']}"
    changed = [name for name, expected in plan.get("source_files_sha256", {}).items()
               if not (ROOT / name).is_file() or file_identity(ROOT / name)["sha256"] != expected]
    output.mkdir(parents=True)
    (output / "run-evidence").mkdir()
    for row, run in pairs:
        if row["status"] == "completed":
            write_json(output / "run-evidence" / f"{row['key']}.json", export_run(run.path))
    write_json(output / "publication.json", {"format_version": 1, "label": label, "note": args.note,
               "generated_utc": datetime.now(timezone.utc).isoformat(), "study_file": file_identity(study / "study.json"),
               "status_file": file_identity(study / "status.json") if (study / "status.json").is_file() else None,
               "recorded_plan": plan, "current_source_mismatches": changed, "runs": rows})
    fields = ["key", "status", "seed", "attempts_recorded", "attempts", *QUALITY, *WORK, "phase_wall_seconds"]
    table(rows, fields, output / "summary.csv")
    summary = table(rows, fields[:9])
    resources = table(rows, ["key", "status", *WORK, "phase_wall_seconds"])
    domains = [{"key": row["key"], "status": row["status"], **row["per_domain_accuracy"]} for row in rows]
    domain_names = sorted({key for row in rows for key in row["per_domain_accuracy"]})
    domain_table = table(domains, ["key", "status", *domain_names], output / "per-domain-accuracy.csv")
    lines = [f"# {plan['study_id']}", "", f"**{label}**", "", args.note, "",
             f"Frozen plan SHA-256: `{plan['plan_sha256']}`. Full run/config/source/A/B/manifest identities: `publication.json`.", "",
             "Missing values are not zero. Final-test scores require a completed status and matching checkpoint/configuration/horizon evidence.",
             "Validation figures also retain incomplete runs, labelled by status. No seed means or confidence intervals are estimated.", "",
             "Phase wall seconds sum recorded child phases, including retry history; orchestration gaps are excluded. Work is accepted-reply/reported-centralized telemetry, including partial centralized blocks. Coverage is in publication.json.",
             "Traffic is serialized application request/reply bytes, not measured network-wire traffic; initial asset/A transfer is excluded. No matched-comparison, superiority, or physical-laptop claim is established.", "",
             f"Current source files differing from the frozen plan: {', '.join(changed) if changed else 'none'}. Historical hashes are retained; this does not validate a rerun.", "",
             "## Measured summary", "", *summary, "", "## Work, time and traffic", "", *resources,
             "", "## Final-test per-domain accuracy", "", *domain_table, "",
             "![Validation task loss](validation-task-loss.png)", "", "![Final-test macro F1](final-test-macro-f1.png)"]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figures(rows, runs, output, label, plan["study_id"])
    print(f"{label}\n{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
