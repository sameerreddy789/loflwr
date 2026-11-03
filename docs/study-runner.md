# Run a frozen study

`tools/run_study.py` freezes the selected study matrix before executing real
training. Prepare the cached model/common A and verified BANKING77 manifests
first. The launcher does not download assets or install dependencies.

List run keys without loading model dependencies:

```powershell
python tools/run_study.py --list
python tools/run_study.py --mode full --list
```

The pilot contains `base`, `centralized`, `sample`, `equal`, `main`, `dropout`,
and `omit4`. It fixes seed 42, imbalanced training manifests, budgets
`[2,16,8,4,2]`, and twenty attempts/centralized blocks. The dropout run kills
client 1 after two completed steps on attempts 4, 8, and 12. The omission run
leaves client 4 out of requests while preserving the other client IDs and
quorum three.

The full matrix contains thirty federated runs, nine centralized references,
and one frozen-base reference. It includes the four declared conditions for
sample and median-clipped equal aggregation at seeds 42/43/44, plain equal
controls for imbalanced no-dropout runs, and the three omission runs.

## Calibrate using validation only

```powershell
python tools/run_study.py --study-dir runs/calibration-001 --mode pilot --attempts 1 --runs base,centralized,main --device cpu --deadline-seconds 600 --max-seconds 3600
```

Any horizon other than twenty labels the entire plan **CALIBRATION** and
disables final-test access in both the launcher and its isolated model worker.
Calibration still records complete validation evaluations. It does not count
as completion of the twenty-attempt study. There is no calibration bypass flag
for final-test access.

## Freeze and execute the pilot

```powershell
python tools/run_study.py --study-dir runs/pilot-001 --mode pilot --device cpu --deadline-seconds 600 --plan-only
python tools/run_study.py --study-dir runs/pilot-001 --resume --max-seconds 14400
```

Use a measured collection deadline that includes the single worker's queue.
The example deadline is a declared configuration, not a claim about measured
runtime. Every run uses the same declared deadline. `--max-seconds` bounds the
launcher invocation, including evaluation and model loading; it stops the
owned child process group when time expires and keeps the last durable
checkpoint, logs, and status. Unfinished runs stay visibly unfinished.

For a subset, use `--runs sample,equal,main`; the frozen plan's execution order
is retained. Glob selectors such as `--runs "omit4-*"` work with the full
matrix. The study summary retains unselected runs and never calls a partial
selection a completed matrix.

```powershell
python tools/run_study.py --study-dir runs/full-001 --mode full --device cpu --plan-only
python tools/run_study.py --study-dir runs/full-001 --resume --runs "*-s42,base" --max-seconds 14400
```

The runner starts Flower through
`python -m fedlora_edge.federation simulate --config ...`. The model workers
import the training stack only when they run. All experiments execute
sequentially. The centralized reference uses the trainer's declared per-client
round-robin minibatch replay and resets its optimizer at each block.

## Results and interrupted work

The study directory contains:

- `study.json`: the complete plan, its hash, source-code hashes, model/common-A
  identity, initial B hash, manifest hashes, and installed environment identity.
- `configs/`: immutable per-run runtime configurations written before execution.
- `status.json`: process status and phase history, including failed retries.
- `summary.json`: the declared matrix with completed, failed, budget-exhausted,
  and unselected runs plus actual final-test measurements when available.
- `runs/<key>/`: stdout, events, checkpoints, work, validation, and any
  `test_metrics.json` from the final committed checkpoint.

`--resume` skips completed runs, continues an unfinished declared horizon, and
preserves earlier execution failures. It rejects changed source/configuration,
manifest, or initial-B files. Use a new study identity for changed experiments.
Run directories are compatible with the [dashboard](dashboard.md).

For pilot/full studies with twenty declared attempts, the final-test worker
requires a saved checkpoint at the complete horizon and completed round
records. Skipped attempts count; they do not create new model versions. All
configuration choices are frozen before test access. A successful execution
does not establish learning improvement, superiority, matched actual work, or
multi-device deployment: inspect those measured outcomes separately.
