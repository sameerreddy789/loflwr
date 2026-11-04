# Check the laptop before presenting

Run the readiness check from the prepared repository before each rehearsal:

```powershell
Set-Location D:\Hackathons\loflwr
.\tools\check_demo.ps1
```

The command prints five checks and saves a timestamped JSON report under ignored
`runs/`. It uses the prepared Windows CPU environment by default. A clean clone
first needs the [tested dependencies](../requirements/README.md), pinned MPNet
assets in `assets/reference-mpnet/`, and the confirmed checkpoint in
`runs/encoder-flower-balanced-v2/selected/`. Those large local assets are excluded
from Git. The preflight does not install or download them.

## Act on the result

- **`ready: open_existing`**: open <http://127.0.0.1:8511/>. The check identified
  this repository's running encoder dashboard, matching run/model/policy paths
  and its healthy local endpoint. Select **Live banking query** inside the app.
- **`ready: launch`**: run `.\tools\start_encoder_demo.ps1`, then open that URL.
  Keep the launch terminal open while presenting.
- **`ready_with_warnings`**: read the dependency differences in the JSON report.
  Inference passed, but installed versions differ from the tested pins. Review
  that difference before relying on this environment for a demo.
- **`blocked`**: fix the named failure, then rerun with a new report filename.
  The script leaves other processes, model files and evaluation rules unchanged.

Open the [judging guide](judging-presentation.md) after the checks pass. Warm one
authored query, confirm the live/recorded labels, and rehearse the backup locally.
An existing listener can change after the check; readiness is point-in-time.

## Resolve common failures

| Failure | Action |
| --- | --- |
| Python or dependencies | Select the prepared environment, or finish the documented installation before disconnecting the internet. A missing dependency fails the check; pin drift is an explicit warning. |
| Published evidence | Restore or review the affected committed evidence. The original confirmation lock, deck and original backup recording must match their recorded identities. Do not replace hashes merely to make the check pass. |
| Port occupied | Inspect the listener or choose another port. An arbitrary healthy web server is not accepted as this dashboard. The tool does not terminate listeners. |
| Cached model or checkpoint | Restore the exact pinned assets and version-500 confirmed head from the prepared laptop's files. The report includes the inference error. Do not substitute an unconfirmed checkpoint. |
| Inference timeout | Close unrelated heavy workloads and try again. The default probe limit is 120 seconds; the Python CLI accepts `--timeout` from 10 to 300 seconds. |
| Report exists | Choose a new filename. Existing reports are never overwritten. Reports must stay under `runs/`. |

To use another prepared environment or port:

```powershell
.\tools\check_demo.ps1 -Python C:\path\to\environment\Scripts\python.exe -Port 8512
.\tools\start_encoder_demo.ps1 -Python C:\path\to\environment\Scripts\python.exe -Port 8512
```

If the first command says `open_existing`, open its URL instead of starting a
second server. For a named report:

```powershell
.\tools\check_demo.ps1 -Output runs/demo-preflight-rehearsal-02.json
```

The underlying `python tools/demo_preflight.py` exits **0** for ready or ready
with warnings, and **2** for blocked readiness or invalid arguments/report output.
The PowerShell wrapper raises an error on any nonzero result.

## What the check establishes

It verifies Python 3.11+, critical dependency versions, the published evidence
used by the demo, the local port, and actual cached inference in a separate
bounded process. The probe verifies model assets, run/configuration/head
identity, version 500 and the fixed clarification policy. It uses the authored
functional input “My card payment is still pending.” It never reads held-out
queries, retrains a model or changes the selected policy.

The reviewed 4 November 2026 run `demo-preflight-20261104-v2` passed all five
checks and returned `pending_card_payment`. Model loading took 28.05 seconds and
the query 0.286 seconds in that probe. The [exported report](../research/results/demo-extension-v1/preflight.json)
preserves those observations. Probe times and sampled process RSS are
functional diagnostics, not an isolated performance comparison or new accuracy
measurement. The independently confirmed accuracy remains **94% (282/300)**.

Cached loading is enforced with offline library settings. OS-level internet
disconnection, live TLS/known-node operation, physical-laptop recovery, and the
presenter's rehearsal remain separate checks. Follow the
[physical-laptop guide](encoder-laptop-rehearsal.md) for the networked run.
