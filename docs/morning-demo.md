# Morning demo on the prepared laptop

This guide is for the first hands-on check of the one-laptop prototype. It uses the existing local model, adapter, data, and Python environments. Five clients in the run are logical participants sharing one GPU worker. The three-physical-laptop deployment remains a later rehearsal.

## 1. Open the saved evidence

Open [http://127.0.0.1:8510](http://127.0.0.1:8510). If the dashboard is not running, start it from PowerShell:

```powershell
Set-Location D:\Hackathons\loflwr
.\tools\start_demo.ps1
```

Keep that terminal open; use Ctrl+C there when finished. The launcher finds the Windows CPU environment at `C:\Users\samee\.venvs\loflwr-cpu\Scripts\python.exe`, sets offline model flags, and serves only on localhost. Use `-Port 8511` if another application already owns port 8510. Do not start a second dashboard merely because the existing page needs **Refresh files**.

In **Recorded run**, select `flower-gpu-smoke-20261102` for the completed initial check, or inspect a later identified run from the [results register](../research/results/README.md). For the initial check:

1. **Overview:** confirm run ID, one attempt, one commit, and saved model version 1.
2. **Attempts & clients:** confirm all five clients were accepted, each completed two optimizer steps, and each coefficient is 0.2. Inspect the norm/clipping details and recorded traffic.
3. **Quality:** confirm the split is validation and the count is 200. There is no matched frozen-base comparison in this one-round run; do not describe its accuracy as an improvement.
4. **Evidence:** inspect the saved configuration and hashes. A timestamp describes a recorded event; it does not prove that training is still running.

The console does not start training or invent results. Study outputs can appear in nested run directories; use **Refresh files** after a process finishes writing. Read an overnight study's `status.json` and `summary.json` before describing its matrix as completed. Unselected, failed, or budget-exhausted runs remain incomplete.

## 2. Try actual inference

Stay in **Latest saved state**, open **Ask a query**, and choose the run's saved checkpoint. The default local paths are `assets/model` and `assets/adapter`. Enter a new banking request, for example:

> I made a transfer yesterday and it still says pending.

Press **Score query**. The first request loads the model on CPU; subsequent requests can reuse it. Check that the result shows a permitted intent and the authored response, and inspect the candidate scores. Try another request about a different domain without treating a few examples as an accuracy test.

The router always chooses one of twenty labels, including for unrelated text. Scores are summed completion log probabilities, including EOS, rather than calibrated confidence. Responses are demonstration content. The app has no bank account access and performs no banking actions.

Select **Frozen base + zero B** only when intentionally comparing inference for the initial adapter state. A single query comparison is illustrative, not the frozen study. If loading fails, the panel reports the error; inspect the asset paths and checkpoint identity before retrying. Run inspection is still available without model loading.

## 3. Show recorded event replay

Choose **Replay recorded events** in the sidebar and move the event-count slider. This displays a prefix of the selected real event stream. It is explicitly a replay; it does not simulate new training. Query inference is disabled in this mode because the latest checkpoint may belong to a later event. Return to **Latest saved state** to score queries again.

For a dropout demonstration, select a run whose saved evidence actually includes an interruption and later synchronization. Show the accepted/rejected replies and whether the attempt committed or skipped. Do not relabel the initial all-client smoke as dropout recovery. See the [status page](../research/implementation-status.md) for the current evidence boundary.

## 4. Check the prepared environments

Run these from a separate PowerShell terminal in `D:\Hackathons\loflwr`:

```powershell
& 'C:\Users\samee\.venvs\loflwr-cpu\Scripts\python.exe' -m fedlora_edge.data verify
wsl -d Ubuntu --cd /mnt/d/Hackathons/loflwr /home/sameer/.venvs/loflwr/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The prepared training environment is Ubuntu/WSL2 with Python 3.11 and a CUDA-enabled Torch build on an NVIDIA GeForce RTX 3050 6 GB Laptop GPU. The Windows environment is for CPU training/inference and the dashboard; it does not contain Flower. Full pins and clean setup instructions are in [requirements](../requirements/README.md).

Do not repeat the download or adapter preparation steps on the prepared machine just to launch the demo. They are for clean setup; the existing adapter files bind the checkpoint to a specific common A and initial B.

## 5. Optionally run a fresh short GPU check

First allow any existing study/training process to finish or stop it in its own terminal. Concurrent GPU jobs change timing and can exhaust memory. Keep the laptop powered and prevent sleep during a run. This command launches actual training with five logical clients and a full validation evaluation:

```powershell
$demoRun = 'morning-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
wsl -d Ubuntu --cd /mnt/d/Hackathons/loflwr /home/sameer/.venvs/loflwr/bin/python -m fedlora_edge.federation simulate --run-id $demoRun --device cuda --layout five --scenario balanced --strategy equal_clipped --budgets 2 --attempts 1 --deadline-seconds 600 --validation-manifest assets/banking77/manifests/validation.json
```

The unique run name preserves existing results. The runtime disables model downloads and records its configuration and outputs under `runs/<run-id>/`. The collection deadline includes the single worker's queue; validation happens outside that collection interval. The command may use the optimized worker implementation, so its timings are new measurements rather than a repeat of the earlier implementation's timings.

Refresh the dashboard and select that run. Confirm the actual outcome: accepted clients, completed work, committed/skipped attempt, and validation count. A skipped attempt retains the previous model and is evidence to inspect, not a successful learning round. An exception should leave logs and any durable checkpoint; keep them for diagnosis.

For a study, follow the [frozen-study guide](study-runner.md) instead of extending this ad hoc smoke. Short calibration plans cannot access the final test. `--resume` requires the same frozen plan and source/assets; after code or configuration changes, use a new study directory. Do not edit a saved plan to make a failed run appear complete.

## Before calling the demo offline

Stop the dashboard, disconnect internet access, and start it again with the launcher. Open a saved run and score a query from a fresh process. This verifies the local cache path rather than relying on an already-loaded model. Record the outcome. A one-laptop offline restart still leaves LAN transport, known-node authentication, physical dropout/rejoin, and budgets on two additional laptops to be tested.

Keep run artifacts in `runs/`. Publish selected reports with run/configuration/model/manifest identity and limitations; keep weights, raw queries, and private keys out of commits. The [dashboard guide](dashboard.md) describes report export and the measured file contract.
