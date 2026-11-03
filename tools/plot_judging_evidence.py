"""Plot published aggregate evidence for judging; never load query data."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = ROOT / "research/results/encoder-v2"
    filenames = ("imbalance-trials.json", "head-trials.json", "omission-head.json")
    sources = {name: json.loads((folder / name).read_text()) for name in filenames}
    imbalance = {row["plan"]["run_id"]: row for row in sources["imbalance-trials.json"]}
    full = next(row for row in sources["head-trials.json"] if row["plan"]["run_id"] == "balance-fast-long")
    omitted = sources["omission-head.json"]
    # Compare the same checkpoint attempt, not runs with different full horizons.
    matched_fields = ("seed", "identity", "head_config", "initial_state_hash", "budgets", "strategy")
    for key in matched_fields:
        if full["plan"][key] != omitted["plan"][key]:
            raise ValueError(f"Omission pair differs on {key}.")
    rows = [*imbalance.values(), full, omitted]
    if any(row["status"] != "completed" for row in rows):
        raise ValueError("Figure requires completed experiment evidence.")
    scores = [[imbalance[k]["selected"]["metrics"]["accuracy"] * 100 for k in ("sample", "equal", "clipped")],
              [next(e for e in row["history"] if e["attempt_id"] == 500)["metrics"]["accuracy"] * 100 for row in (full, omitted)]]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=160)
    fig.patch.set_facecolor("#f8fafb")
    fig.subplots_adjust(left=.07, right=.98, top=.74, bottom=.23, wspace=.28)
    fig.text(.07, .93, "Federation evidence on one laptop", fontsize=25, weight="bold", color="#172d42")
    fig.text(.07, .865, "Two separate validation experiments · 20 BANKING77 intents · one training seed (42)", fontsize=13, color="#42596b")
    titles = ["90% data imbalance", "Weak-client omission"]
    subtitles = ["400 training queries · equal local work", "2,760 training queries · one step per client"]
    labels = [["Sample\nweighting", "Equal\nweighting", "Equal +\nnorm clipping"],
              ["All five\nclients", "Client 4\nomitted"]]
    colors = [["#afbac5", "#22798a", "#174c65"], ["#22798a", "#b47b44"]]
    for ax, title, subtitle, values, names, palette in zip(axes, titles, subtitles, scores, labels, colors):
        ax.set_facecolor("#f8fafb")
        ax.set_title(title + "\n" + subtitle, loc="left", fontsize=13, pad=16)
        bars = ax.bar(names, values, color=palette, width=.56, zorder=3)
        ax.set_ylim(0, 106)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Best validation accuracy (%)" if ax is axes[0] else "Validation accuracy at attempt 500 (%)")
        ax.grid(axis="y", color="#dce4e9", zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#dce4e9")
        ax.tick_params(axis="both", length=0)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:g}%", ha="center", fontsize=17, weight="bold", color="#172d42")
    fig.text(.07, .11, "Clipping versus equal weighting: 0.5 percentage points (one query).\nThis does not establish a reliable clipping advantage.", fontsize=10, color="#42596b")
    fig.text(.57, .11, "At attempt 500: 18-point drop; client 4 solely owned four labels.\nThe full-client control continued to attempt 1,000.", fontsize=10, color="#42596b")
    fig.text(.07, .035, "Recorded experiments, not a live run. These are development scores; the deployed head's independent result is 94% (282/300).", fontsize=10, color="#42596b")
    target = folder / "judging-evidence.png"
    fig.savefig(target, facecolor=fig.get_facecolor())
    plt.close(fig)
    provenance = {"kind": "published_aggregate_figure", "figure": target.name,
        "source_files_sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in filenames},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figure_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "scores_percent": scores, "omission_matched_fields": list(matched_fields),
        "omission_comparison_attempt": 500,
        "omission_run_horizons": {row["plan"]["run_id"]: row["plan"]["attempts"] for row in (full, omitted)},
        "omission_evaluation_intervals": {row["plan"]["run_id"]: row["plan"]["evaluate_every"] for row in (full, omitted)},
        "interpretation": "Separate data regimes, reused validation, one seed. Omission compares identical initialization/optimizer/budgets at attempt500, not entire runs with different horizons/evaluation schedules. No physical-laptop or population guarantee."}
    (folder / "judging-evidence-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
