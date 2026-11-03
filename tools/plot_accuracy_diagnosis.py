"""Render the committed accuracy evidence; never reads live runs or test data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research" / "results"
SOURCES = ["accuracy-attempt-7-audit.json", "tfidf-validation.json",
           "tfidf-balanced-validation.json", "accuracy-scoring-base.json",
           "accuracy-scoring-sample-v9.json"]


def read(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))["measurements"]


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.hashsalt": "fedlora-accuracy-20261103"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.1), gridspec_kw={"width_ratios": [1.25, 0.8, 1.25]})
    fig.subplots_adjust(top=0.73, bottom=0.30, left=0.05, right=0.985, wspace=0.35)
    fig.suptitle("Accuracy depends on data coverage and the scoring rule", fontsize=19,
                 fontweight="bold", x=0.05, y=0.965, ha="left")
    fig.text(0.05, 0.9, "Measured validation evidence | 200 queries, 20 intents | 3 November 2026", fontsize=11)
    fig.text(0.05, 0.85, "Separate comparisons below: do not read the three panels as one model leaderboard.", color="#515a66")
    evaluations = read(SOURCES[0])["runs"]["pilot-r2-20261103"]["evaluations"]
    ax = axes[0]
    palette = ["#156e83", "#bc4f22", "#506fb3", "#7a3c7d", "#55812c"]
    for domain, short, color in zip(
        ["Card lifecycle", "Card payments", "Transfers", "Cash withdrawals", "Account verification"],
        ["Lifecycle", "Payments", "Transfers", "Cash", "Verification"], palette,
    ):
        ax.plot([row["attempt_id"] for row in evaluations],
                [row["per_domain_accuracy"][domain] for row in evaluations],
                marker="o", markersize=3.5, linewidth=2, label=short, color=color)
    ax.set_title("A. Partial federated baseline", loc="left", pad=15)
    ax.set_xlabel("Attempt (sample weighting, seed 42)")
    ax.set_ylabel("Validation accuracy")
    ax.set_xticks(range(8))
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=8.5, frameon=False)
    axes[0].text(0, 1.03, "Imbalanced data; original scoring", transform=axes[0].transAxes, fontsize=9, color="#515a66")

    ax = axes[1]
    values = [read(name)["metrics"]["accuracy"] for name in SOURCES[1:3]]
    bars = ax.bar(["Imbalanced\n400 queries", "Balanced\n800 queries"], values,
                  width=0.58, color=["#7a8593", "#156e83"])
    ax.bar_label(bars, labels=[f"{100 * value:g}%" for value in values], padding=5, fontweight="bold")
    ax.set_title("B. Fixed TF-IDF reference", loc="left", pad=15)
    ax.text(0, 1.03, "Centralized; identical classifier settings", transform=ax.transAxes, fontsize=9, color="#515a66")
    ax.text(0.5, -0.31, "Both quantity and class coverage change.\nThis is not a federated LoRA result.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#515a66")

    ax = axes[2]
    rules = ["sum_including_eos", "mean_including_eos", "content_free_centered_sum"]
    for offset, source, label, color in [(-0.18, SOURCES[3], "Frozen base", "#a3abb6"),
                                         (0.18, SOURCES[4], "Trained, version 9", "#156e83")]:
        values = [read(source)["variants"][rule]["metrics"]["accuracy"] for rule in rules]
        bars = ax.bar([index + offset for index in range(3)], values, width=0.34, label=label, color=color)
        ax.bar_label(bars, labels=[f"{100 * value:g}%" for value in values], padding=4, fontsize=9)
    ax.set_xticks(range(3), ["Original\nsum", "Per-token\nmean", "Content-free\ncentered"])
    ax.set_title("C. Scoring diagnostic", loc="left", pad=15)
    ax.text(0, 1.03, "Same queries; no additional training", transform=ax.transAxes, fontsize=9, color="#515a66")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.27), frameon=False, fontsize=9)
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.17)
    fig.text(0.05, 0.045, "Partial checkpoints and exploratory validation only. No final-test or 100% accuracy claim. Sources: research/results/*.json", fontsize=9, color="#515a66")
    for suffix in ("png", "svg"):
        fig.savefig(RESULTS / f"accuracy-diagnosis.{suffix}", dpi=170, facecolor="white",
                    metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)
    svg = RESULTS / "accuracy-diagnosis.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    identity = {"sources_sha256": {name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() for name in SOURCES},
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "matplotlib_version": matplotlib.__version__,
                "scope": "Fixed partial validation snapshots; separate data regimes and methods."}
    (RESULTS / "accuracy-figure-provenance.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    print(RESULTS / "accuracy-diagnosis.png")


if __name__ == "__main__":
    main()
