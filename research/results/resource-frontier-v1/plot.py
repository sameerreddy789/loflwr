"""Render only the published measured summary; no model or dataset access.

Reproduce: python research/results/resource-frontier-v1/plot.py
Requires Matplotlib. Outputs resource-frontier.png and .svg alongside this file.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
if summary["status"] != "completed" or summary["completed_cells"] != 9:
    raise ValueError("Plot requires all nine completed cells")

colors = {"1": "#176b67", "2": "#a76321", "4": "#6b5594"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), gridspec_kw={"width_ratios": [1, 1.12]})
for key in ("1", "2", "4"):
    group = summary["groups"][key]
    quality = group["trajectory"][-1]["metrics"]["accuracy"]
    payload = group["resources"]["estimated_total_bytes"]["mean"] / 1_000_000
    axes[0].errorbar(payload, quality["mean"], yerr=quality["sample_std"],
                     fmt="o", ms=9, color=colors[key], capsize=4, zorder=3)
    axes[0].annotate(f"{key} local step{'s' if key != '1' else ''}\n{quality['mean']:.1%}; {payload:.1f} MB",
                     (payload, quality["mean"]), xytext=(0, 44 if key == "2" else 13), textcoords="offset points",
                     ha="center", va="bottom", color=colors[key], fontsize=9)
    # Zero-step values are fully retained in summary and the report table.
    trajectory = group["trajectory"][1:]
    x = [r["steps_per_client"] for r in trajectory]
    mean = [r["metrics"]["accuracy"]["mean"] for r in trajectory]
    sd = [r["metrics"]["accuracy"]["sample_std"] for r in trajectory]
    axes[1].plot(x, mean, "o-", color=colors[key], label=f"{key} local step{'s' if key != '1' else ''}", lw=1.7, ms=5)
    axes[1].fill_between(x, [m-s for m, s in zip(mean, sd)], [m+s for m, s in zip(mean, sd)],
                         color=colors[key], alpha=.15)

axes[0].set(title="Final quality versus estimated payload", xlabel="Calculated cumulative upload + download (MB)",
            ylabel="Reused-validation accuracy", xlim=(35, 350), ylim=(.925, .969))
axes[0].set_yticks([.93, .94, .95, .96])
axes[1].set(title="Quality at matched completed work", xlabel="Completed optimizer steps per client",
            ylabel="Reused-validation accuracy", xlim=(90, 510), ylim=(.85, .965))
axes[1].set_xticks([100, 200, 300, 400, 500])
axes[1].legend(loc="lower right", frameon=False)
for axis in axes:
    axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axis.grid(axis="y", color="#e4e7e7", linewidth=.7)
    axis.set_axisbelow(True)
fig.suptitle("Fixed-work cached MPNet classifier head", x=.065, ha="left", fontsize=16, fontweight="bold")
fig.text(.065, .045, "Three seeds; means ± sample SD. Reused validation n=200. All cells: 500 steps/client, 1.38M examples.\n"
         "Payload is arithmetic FP32 head traffic, not measured network traffic. Zero-step accuracy is reported in the table.",
         ha="left", fontsize=9, color="#4b5555")
fig.subplots_adjust(left=.07, right=.985, bottom=.24, top=.83, wspace=.29)
fig.savefig(HERE / "resource-frontier.png", dpi=180, facecolor="white")
svg_path = HERE / "resource-frontier.svg"
fig.savefig(svg_path, facecolor="white")
# Matplotlib's multiline SVG paths can end lines with spaces. Keep every
# separator newline while removing only trailing horizontal whitespace.
svg_text = svg_path.read_text(encoding="utf-8")
svg_path.write_text("\n".join(line.rstrip(" \t") for line in svg_text.splitlines()) + "\n",
                    encoding="utf-8", newline="\n")
plt.close(fig)
