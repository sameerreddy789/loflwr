"""Plot completed encoder validation histories; no model/data/test loading."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.runs) != len(args.labels):
        parser.error("Supply one label per run.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    provenance = []
    for run, label in zip(args.runs, args.labels):
        path = run / "summary.json"
        if not path.exists():
            path = run / "final_results.json"
        raw = path.read_bytes()
        result = json.loads(raw)
        history = result.get("history", result.get("evaluations", []))
        if not history or any(row["split"] != "validation" for row in history):
            raise ValueError("Completed validation histories are required.")
        for ax, metric in zip(axes, ("accuracy", "worst_domain_accuracy")):
            ax.plot([row["attempt_id"] for row in history],
                    [100*row["metrics"][metric] for row in history], label=label, linewidth=1.8)
        provenance.append({"run": str(run), "label": label, "sha256": hashlib.sha256(raw).hexdigest()})
    for ax, title in zip(axes, ("All 20 intents", "Lowest-scoring domain")):
        ax.set(title=title, xlabel="Round attempt", ylim=(0, 101))
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Frozen MPNet + federated head: measured learning curves", fontsize=13)
    fig.text(.01,.01,"200 reused validation queries · same 2,760 training examples · seed 42 · sequential federation",fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")


if __name__ == "__main__":
    main()
