"""Regenerate the human-readable report from measured v1 artifacts only."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
plan = json.loads((HERE / "plan.json").read_text(encoding="utf-8"))
if summary["status"] != "completed" or summary["completed_cells"] != 9 or summary["plan_hash"] != plan["plan_hash"]:
    raise ValueError("The complete matched v1 evidence is required")
groups = summary["groups"]
results = sorted(summary["results"], key=lambda r: (r["cell"]["seed"], r["cell"]["local_steps"]))


def mean_sd(value, scale=1, digits=2):
    return f"{value['mean'] * scale:.{digits}f} ± {value['sample_std'] * scale:.{digits}f}"


lines = [
    "# Fixed-work head accuracy/resource study — resource-frontier-v1",
    "",
    "All nine predeclared cells completed on 4 November 2026. At exactly 500 optimizer steps per client, "
    "one local step per round scored **95.5% (191/200)** and two or four local steps scored **93.5% (187/200)** "
    "on every seed. Four steps reduced calculated head payload by **75%** and observed mean process wall time "
    "by **66.8%** relative to one step. The observed cost is two percentage points of reused-validation accuracy.",
    "",
    "This is development evidence from cached frozen MPNet features and a trained linear classifier head: "
    "five sequential numerical clients on one host, **not LoRA, Flower, physical-device performance, or new test accuracy**. "
    "The 200-query validation set was already reused for development. The confirmed demo and all original, confirmation, "
    "CLINC and scope locks remain separate; this study does not score those sets or replace the deployed model.",
    "",
    "[Pre-run plan](plan.md) · [Frozen machine plan](plan.json) · [All results and statistics](summary.json) · "
    "[Figure source](plot.py) · [SVG](resource-frontier.svg)",
    "",
    "![Measured quality and calculated payload](resource-frontier.png)",
    "",
    "## Final results",
    "",
    "Mean ± sample SD across seeds 42/43/44. MB is decimal; MiB is binary. Payload is calculated, not measured network traffic.",
    "",
    "| Steps/client/round | Rounds | Accuracy (%) | Macro F1 | Worst domain (%) | Cross-entropy | Estimated payload (MB) | Process wall (s) | Peak process RSS (MiB) |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
]
for key in ("1", "2", "4"):
    group = groups[key]
    metrics, resources = group["trajectory"][-1]["metrics"], group["resources"]
    lines.append(f"| {key} | {group['rounds']} | {mean_sd(metrics['accuracy'], 100, 1)} | "
                 f"{mean_sd(metrics['macro_f1'], digits=5)} | {mean_sd(metrics['worst_domain_accuracy'], 100, 1)} | "
                 f"{mean_sd(metrics['cross_entropy'], digits=5)} | {resources['estimated_total_bytes']['mean']/1e6:.1f} | "
                 f"{mean_sd(resources['process_wall_seconds'])} | {mean_sd(resources['peak_process_rss_bytes'], 1/2**20)} |")
lines += ["", "All nine final observations (no checkpoint selection):", "",
          "| Cell | Accuracy (%) | Macro F1 | Cross-entropy | Process wall (s) | Peak RSS (MiB) |",
          "| --- | --- | --- | --- | --- | --- |"]
for row in results:
    metrics = row["history"][-1]["metrics"]
    lines.append(f"| {row['cell']['id']} | {100*metrics['accuracy']:.1f} | {metrics['macro_f1']:.6f} | "
                 f"{metrics['cross_entropy']:.6f} | {row['process_wall_seconds']:.3f} | {row['peak_process_rss_bytes']/2**20:.3f} |")
lines += ["", "## Every scheduled accuracy observation", "",
          "Columns are completed optimizer steps **per client**; entries are reused-validation accuracy (%). "
          "The machine summary retains all 54 complete metric records, per-intent/domain results, state hashes, "
          "cumulative timings, actual work and estimated payload at each checkpoint.", "",
          "| Cell | 0 | 100 | 200 | 300 | 400 | 500 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
for row in results:
    lines.append("| " + row["cell"]["id"] + " | " + " | ".join(f"{100*p['metrics']['accuracy']:.1f}" for p in row["history"]) + " |")
lines += ["", "Final per-domain accuracy is identical across all three seeds for each schedule "
          "(sample SD 0.0 percentage points). The one-step advantage is in Transfers; matching overall scores do not imply identical predictions.", "",
          "| Local steps | Lifecycle (%) | Payments (%) | Transfers (%) | Cash (%) | Verification (%) | Domain gap (pp) |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
for key in ("1", "2", "4"):
    final = groups[key]["trajectory"][-1]
    lines.append("| " + key + " | " + " | ".join(f"{100*v['mean']:.1f}" for v in final["per_domain_accuracy"].values())
                 + f" | {100*final['metrics']['domain_gap']['mean']:.1f} |")
lines += ["", "## Timing and memory", "",
          "All cells performed **2,500 optimizer steps and 1,380,000 processed examples**. Per-client totals were "
          "256,500 / 307,500 / 304,000 / 308,000 / 204,000 examples. The raw affine head is **61,520 bytes** "
          "(15,380 FP32 parameters). The fixed full-batch SGD settings, initial tensors and dataset hashes are in the frozen plan.", "",
          "Mean ± sample SD in seconds:", "",
          "| Local steps | Cell wall | Summed local trainer calls | Optimizer loop (subset) | Aggregation | Evaluation (six passes) |",
          "| --- | --- | --- | --- | --- | --- |"]
for key in ("1", "2", "4"):
    resource = groups[key]["resources"]
    lines.append("| " + key + " | " + " | ".join(mean_sd(resource[name], digits=3) for name in
                 ("cell_wall_seconds", "local_train_seconds", "optimization_seconds", "aggregation_seconds", "evaluation_seconds")) + " |")
lines += ["", "The supervisor started **fresh serial Linux subprocesses** in predeclared rotated order, with one CPU thread "
          "and OMP/MKL/OpenBLAS/NumExpr thread limits of one. Root inference/tests and video encoding were paused throughout the timed interval. "
          f"Total child-process wall time was {sum(r['process_wall_seconds'] for r in results):.3f} seconds. "
          "The process clock includes startup, imports and exit. Cell wall includes source/data verification, loading, training, aggregation, "
          "evaluation and output writing before final result serialization. Trainer-call timing includes its validation, state conversion/hash, "
          "optimizer setup and monitoring overhead; optimizer-loop timing is a subset and must not be added again. "
          "Unattributed cell overhead includes state replacement, delta conversion, orchestration, hashing and I/O. "
          "Equal optimizer steps therefore do not imply equal total arithmetic work.", "",
          "Peak RSS is the **complete child-process lifetime high-water mark**, obtained after exit using Linux `os.wait4` rusage "
          "and converted from KiB to bytes. It includes Python/Torch imports, all five resident feature caches, validation, training and serialization. "
          "It excludes the supervisor/other processes, encoder inference/preparation and physical-device clients. "
          "The child also saves an explicitly named earlier RSS observation; the published peak comes from the supervisor. "
          "One process sample per seed is not a repeated hardware timing experiment. Later cells ran faster, so host/runtime drift may affect time comparisons despite serial scheduling.", "",
          "Runtime: AMD Ryzen 5 6600H, 12 logical CPUs visible to WSL2; Linux 6.6.87.2, Python 3.11.15, "
          "Torch 2.8.0+cu126 **used on CPU**, NumPy 2.4.6, psutil 7.2.2. No GPU timing or memory measurement is claimed.", "",
          "## Tradeoff and limits", "",
          "The estimate is `2 × 5 × rounds × 61,520`: one head download and delta upload per client per round. "
          "The 307.6 / 153.8 / 76.9 MB totals include first-round head distribution but exclude encoder/data transfer, metadata, "
          "transport overhead, retries and final deployment. **No serialized or wire bytes were measured.**", "",
          "For the predeclared **accuracy-versus-payload** frontier, local-step 1 and 4 are non-dominated: one has higher accuracy; "
          "four has lower estimated payload. Four dominates two on those two dimensions because both score 93.5% and four uses half the payload. "
          "When process time and RSS are also included, **all three are formally non-dominated**: local-step 4's mean RSS is about "
          "0.030 MiB above local-step 2. That tiny difference is not evidence of a practical memory advantage. "
          "Two steps also have lower cross-entropy than four, so four does not dominate every quality metric.", "",
          "Paired final accuracy differences versus one step are **−2.0 pp for every seed**, for both two and four steps "
          "(mean −2.0 pp, sample SD 0.0). Their worst-domain accuracy is 2.5 pp lower and domain gap 2.5 pp wider. "
          "Zero observed accuracy SD means these three seeds had equal counts on these 200 queries; it is not zero uncertainty about generalization. "
          "This result supports retaining one step for accuracy-focused development and considering four steps when estimated communication "
          "cost is the constraint. It does not justify replacing the confirmed demo, changing thresholds, or claiming a new untouched evaluation.", "",
          "## Identity and reproduction", "",
          f"Plan hash: `{plan['plan_hash']}`. Baseline: `{plan['baseline_commit']}`. "
          f"The public plan was written locally at `{plan['created_utc']}`, before any measured cell. "
          "This is a development plan, not a pre-pushed independent-test lock. The runner verified source/data/runtime/initialization hashes before each cell; "
          "no frozen source or initial tensor was changed during execution.", "",
          "Follow [plan.md](plan.md) for exact commands and separate reproduction directories. "
          "`tools/run_resource_frontier.py report` verifies the frozen plan and process/result identities and regenerates `summary.json`. "
          "`python research/results/resource-frontier-v1/render_report.py` regenerates this report. "
          "`python research/results/resource-frontier-v1/plot.py` renders the PNG/SVG using Matplotlib; plotting reads only the summary. "
          "The exported figure was rendered with Matplotlib 3.10.8 in an isolated ignored plotting package directory, after timing; the pinned benchmark environment was not modified.", "",
          "The eight focused boundary/configuration tests pass in the WSL benchmark environment with zero skips. "
          "They cover matched work and full batches, prohibited configuration changes, payload arithmetic, actual-work rejection, "
          "source/public-plan/initialization tampering, incomplete/mixed evidence, strict Pareto dominance and sample standard deviation. "
          "The repository-wide required suite is recorded separately by the integration run.", ""]
(HERE / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
