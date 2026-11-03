"""Plot published three-seed development and fixed unsupported-test evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix_path = ROOT / "research/results/encoder-matrix-v1/summary.json"
    scope_path = ROOT / "research/results/scope-v1/negative-test-result.json"
    matrix, scope = [json.loads(p.read_text()) for p in (matrix_path, scope_path)]
    if matrix["status"] != "completed" or matrix["completed"] != 33 or scope["count"] != 2279:
        raise ValueError("Only plot the completed declared evidence")
    groups = matrix["groups"]
    colors = ["#8c969e", "#287b8e", "#d77b37"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    ax = axes[0]
    values = [groups[f"imbalanced-heterogeneous-{name}"]["final"]["accuracy"]
              for name in ("sample", "equal", "equal_clipped")]
    bars = ax.bar(["Sample\nweighted", "Equal", "Equal +\nclipping"],
                  [100*v["mean"] for v in values], color=colors,
                  yerr=[100*v["sample_std"] for v in values], capsize=4)
    ax.bar_label(bars, labels=[f"{100*v['mean']:.1f}%" for v in values], padding=8)
    ax.set(title="90% data imbalance", ylabel="Validation accuracy (%)", ylim=(0, 100))
    ax.text(.5, -.22, "400 training queries; unequal budgets\nAttempt 500; mean ± sample SD, 3 seeds",
            ha="center", transform=ax.transAxes, fontsize=9)

    ax = axes[1]
    entries = [groups[f"expanded-heterogeneous-equal{suffix}"]["final"]["accuracy"]
               for suffix in ("", "-omit4")]
    x = np.arange(3)
    ax.bar(x-.18, 100*np.array(entries[0]["values"]), width=.36, color=colors[1], label="All clients")
    ax.bar(x+.18, 100*np.array(entries[1]["values"]), width=.36, color=colors[0], label="Omit client 4")
    ax.set(xticks=x, xticklabels=["42", "43", "44"], xlabel="Training seed", ylim=(0, 100),
           title="Weak-client contribution", ylabel="Validation accuracy (%)")
    ax.legend(frameon=False, loc="lower left")
    ax.text(.5, -.22, "2,760-query control; unequal budgets\nFour verification intents lose their only owner",
            ha="center", transform=ax.transAxes, fontsize=9)

    ax = axes[2]
    row = scope["comparisons"][-1]
    assert row["margin_threshold"] == 2.1202011108398438
    rates = [100*row[key] for key in ("original_margin_false_accept_rate", "with_scope_false_accept_rate")]
    bars = ax.bar(["Fixed margin", "+ Scope detector"], rates, color=[colors[0], colors[1]])
    ax.bar_label(bars, labels=[f"{rate:.2f}%" for rate in rates], padding=5)
    ax.set(title="Unsupported banking requests", ylabel="False acceptance (%) — lower is better", ylim=(0, 20))
    ax.text(.5, -.22, "2,279 new negative test queries; locked rules\nFresh supported-query coverage remains untested",
            ha="center", transform=ax.transAxes, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("FedLoRA-Edge: new evidence with separate evaluation boundaries", fontsize=16)
    output = ROOT / "research/results/remaining-work-v1"
    for suffix in ("png", "svg"):
        path = output/f"comparison.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n",
                            encoding="utf-8", newline="\n")
    plt.close(fig)
    provenance = {"sources": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (matrix_path, scope_path, Path(__file__))},
                  "matrix_plan_hash": matrix["plan_hash"], "scope_lock_hash": scope["lock_hash"],
                  "limits": "Development validation in panels 1–2; negative-only locked test in panel 3. Confirmed 20-intent accuracy remains 94%."}
    (output/"comparison-provenance.json").write_text(json.dumps(provenance, indent=2)+"\n",
                                                    encoding="utf-8", newline="\n")
    print(output/"comparison.png")


if __name__ == "__main__":
    main()
