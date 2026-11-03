"""Plot the twelve centralized top-1 validation references and their provenance.

This reads compact published evidence only. It does not train a model, score
queries, or turn candidate recall into classification accuracy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = (
    ("imbalanced", "Imbalanced training · 400 queries"),
    ("balanced", "Balanced training · 800 queries"),
    ("all_eligible_training", "All eligible training · 2,760 queries"),
)
METHODS = (
    ("word_tfidf_logreg", "Word TF-IDF + logistic", "#35618B"),
    ("word_char_tfidf_svm", "Word/character TF-IDF + SVM", "#A65A1B"),
    ("mpnet_logreg", "Frozen MPNet + logistic", "#2E7767"),
    ("mpnet_nearest_example", "Frozen MPNet nearest example", "#785698"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def read_measurements(path: Path) -> tuple[dict, dict, dict]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    measurements = evidence["measurements"]
    plan = measurements["plan"]
    if (measurements["status"] != "completed"
            or plan["split"] != "validation"
            or plan["kind"] != "centralized_accuracy_feasibility_references"):
        raise ValueError("Expected completed centralized validation references.")
    for scenario, count in (("imbalanced", 400), ("balanced", 800),
                            ("all_eligible_training", 2760)):
        if plan["scenarios"][scenario]["count"] != count:
            raise ValueError(f"Training count differs from figure label: {scenario}")
    rows = {}
    expected = {(scenario, method) for scenario, _ in SCENARIOS
                for method, _, _ in METHODS}
    for result in measurements["results"]:
        key = (result["scenario"], result["method"])
        metrics = result["metrics"]
        count, correct = metrics["count"], result["correct"]
        if key in rows or key not in expected:
            raise ValueError(f"Duplicate or unexpected reference: {key}")
        if (count != 200 or not 0 <= correct <= count
                or not math.isclose(metrics["accuracy"], correct / count,
                                    rel_tol=0, abs_tol=1e-12)
                or len(metrics["per_intent"]) != 20):
            raise ValueError(f"Invalid top-1 measurement: {key}")
        rows[key] = result
    if set(rows) != expected:
        raise ValueError("All twelve fixed reference combinations are required.")
    return evidence, plan, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "research/results/accuracy-feasibility-references.json")
    parser.add_argument("--output-prefix", type=Path,
                        default=ROOT / "research/results/accuracy-feasibility")
    args = parser.parse_args(argv)
    source, prefix = args.source.resolve(), args.output_prefix.resolve()
    outputs = {suffix: prefix.with_suffix(suffix) for suffix in (".png", ".svg")}
    provenance_path = prefix.parent / (prefix.name + "-figure-provenance.json")
    if source in {*outputs.values(), provenance_path}:
        parser.error("Figure outputs must not overwrite their evidence source.")
    evidence, plan, rows = read_measurements(source)
    source_hash = sha256(source)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.8), sharex=True)
    fig.subplots_adjust(left=0.265, right=0.965, top=0.84, bottom=0.19, hspace=0.67)
    best = max(row["metrics"]["accuracy"] for row in rows.values())
    target_summary = ", below 99%" if best < 0.99 else ""
    fig.text(0.035, 0.965, f"Stronger references reached {best:.1%}{target_summary}",
             fontsize=21, fontweight="bold", va="top", color="#172A3A")
    fig.text(0.035, 0.92, "Centralized references · top-1 intent accuracy · all 20 labels",
             fontsize=12, color="#43515C")
    fig.text(0.965, 0.882, "Dashed line: 99% target", ha="right", fontsize=10,
             color="#A33F30")
    values = []
    for axis, (scenario, title) in zip(axes, SCENARIOS):
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=11)
        for index, (method, label, color) in enumerate(METHODS):
            row = rows[(scenario, method)]
            percent = row["metrics"]["accuracy"] * 100
            axis.barh(index, percent, height=0.69, color=color, zorder=3)
            axis.text(percent - 1.4, index,
                      f"{percent:.1f}% ({row['correct']}/200)",
                      ha="right", va="center", fontsize=10.5,
                      fontweight="medium", color="white", zorder=4)
            values.append({"scenario": scenario, "method": method,
                           "correct": row["correct"], "count": 200,
                           "accuracy": row["metrics"]["accuracy"]})
        axis.set_yticks(range(len(METHODS)), [label for _, label, _ in METHODS])
        axis.tick_params(axis="y", length=0, pad=12)
        axis.tick_params(axis="x", labelbottom=True, length=0, pad=6, colors="#5D6870")
        axis.set_xlim(0, 102)
        axis.set_ylim(3.65, -0.6)
        axis.set_xticks([0, 20, 40, 60, 80, 100])
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        axis.spines["bottom"].set_color("#CBD2D7")
        axis.grid(axis="x", color="#DDE2E6", linewidth=0.8, zorder=0)
        axis.set_axisbelow(True)
        axis.axvline(99, color="#A33F30", linewidth=1.3,
                     linestyle=(0, (4, 3)), zorder=2)

    fig.text(0.035, 0.117,
             "Same 200 validation queries reused across 12 checks; not federated or final-test results.",
             fontsize=10.5, color="#263746")
    fig.text(0.035, 0.094,
             "Training regimes change both quantity and class coverage; this does not isolate balancing.",
             fontsize=10, color="#52606A")
    fig.text(0.035, 0.059, f"Run: {plan['run_id']} · evidence: {source.name}",
             fontsize=9, color="#52606A")
    fig.text(0.035, 0.035, f"Evidence SHA-256: {source_hash}",
             fontsize=8.5, color="#52606A", family="DejaVu Sans Mono")
    for suffix, path in outputs.items():
        fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    # Keep text artifact bytes consistent with the repository's LF policy.
    svg = outputs[".svg"]
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8", newline="\n")

    provenance = {
        "kind": "centralized_top1_validation_reference_figure",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"file": relative(source), "sha256": source_hash,
                   "original_evidence": evidence["source_file"]},
        "run_id": plan["run_id"], "config_hash": plan["config_hash"],
        "validation_records_hash": plan["validation_records_hash"],
        "split": "validation", "labels": 20, "target_accuracy": 0.99,
        "script": {"file": relative(Path(__file__)), "sha256": sha256(Path(__file__))},
        "environment": {"matplotlib": matplotlib.__version__},
        "measurements": values,
        "outputs": [{"file": relative(path), "sha256": sha256(path)}
                    for path in outputs.values()],
        "interpretation": (
            "Twelve top-1 centralized reference measurements on the same 200 validation queries. "
            "No top-k recall, federated accuracy or final-test scores are plotted. "
            "Data regimes change quantity and class coverage; they do not isolate balancing. "
            "The best score is selected from reused validation data, not independent confirmation."
        ),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False,
                                          allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    for path in [*outputs.values(), provenance_path]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
