# Local encoder demo

From the repository root on the prepared laptop:

```powershell
.\tools\start_encoder_demo.ps1
```

Open <http://127.0.0.1:8511>. Keep the terminal open; Ctrl+C stops it. Use
`-Python` for another prepared environment, `-Port` for another loopback port,
or `-RunDirectory` for a completed encoder experiment:

```powershell
.\tools\start_encoder_demo.ps1 -RunDirectory runs/encoder-balanced-v2/balance-fast-long
```

The default is `encoder-flower-fullbatch5-v1` (92.5% validation). The example
selects the newer sequential 95.5% validation run; its transport must not be
called Flower. The sidebar also accepts a run path. The selected version and
run appear above the query form.

## Ask and inspect

Enter an English banking question and choose **Classify query**. The app loads
pinned MPNet locally, applies masked mean pooling and L2 normalization, and uses
the saved classifier head. It shows an intent, a team-authored response, and
three raw logits. These scores are uncalibrated. Every input is assigned one of
twenty intents; unsupported-request detection and account actions are absent.

**Training evidence** shows saved validation metrics, their trajectory, domain
results and, for Flower runs, participation, steps and aggregation weights.
**Supported intents** lists the task scope. Training evidence is recorded;
query inference is live. The reused validation set guided selection and is not
an untouched-test estimate.

Loading verifies pinned assets, the checkpoint tensor hash and the 20-intent
model contract. Evidence must identify the same selected state and version.
Changed checkpoint signatures reload the inference resource. Missing or
incompatible files show a recovery message. This console supports head-only
checkpoints; it does not load the separate MPNet B-plus-head trials.

The server binds to loopback. Model loading uses `local_files_only=True`; no
package or model download is attempted. Queries are not written to disk by
this app. It never loads training, validation or confirmation queries. The
first query includes model initialization; subsequent queries reuse the model.

## Verification

Four dashboard tests passed. A separate real CPU check using the 95.5% selected
head recomputed all 200 validation queries and exactly reproduced cached logits
and predictions. This was an inference parity check on an already scored split.
A fresh pending-payment question returned the matching intent in 0.38 seconds
after loading. Desktop/mobile browser layouts and query submission were checked.

```powershell
python -m unittest discover -s tests -p test_encoder_dashboard.py
```

UI fixtures are not experiment results. The full WSL suite after this change
passed 149 tests, zero skips. Model optimization and Flower transport remain
separate evidence gates. See the [earlier encoder results](../research/results/encoder-v1/README.md).
