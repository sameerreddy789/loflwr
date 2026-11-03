# Local encoder demo

From the repository root on the prepared laptop:

```powershell
.\tools\start_encoder_demo.ps1 -PolicyFile research/results/encoder-v2/final-policy.json
```

Open <http://127.0.0.1:8511>. Keep the terminal open; Ctrl+C stops it. The default
run is `encoder-flower-balanced-v2`, selected version 500, trained through actual
five-client Flower. It scored **94% on independent confirmation (282/300)** and
**95.5% on reused validation (191/200)**. The final model uses frozen MPNet plus
a federated classifier head. Head-only training is distinct from LoRA.

Use `-Python` for another prepared environment, `-Port` for another loopback port,
or `-RunDirectory` for a completed encoder experiment. Omit `-PolicyFile` to always
return the highest-ranked supported intent. No installation or model download is
needed on this prepared laptop; a clean clone still needs the documented assets
and environments.

## Ask and inspect

Enter an English banking question and choose **Classify query**. The app loads
pinned MPNet locally, applies masked mean pooling and L2 normalization, and uses
the saved classifier head. It displays the predicted intent, a team-authored
response, and three raw logits. Logits are uncalibrated scores.

With the policy enabled, a top-two logit margin below `2.1202011108398438` asks
for more detail and withholds the support response. This rule was fixed before
confirmation: it answered **223/300 queries**, got **220 correct (98.65%)**, and
declined 77. Its **74.33% coverage** must accompany the answered-query accuracy.
The full-set result remains 94%; no frozen rule demonstrated 99-100%. Clarification
does not detect every unsupported request, and user responses to clarification
have not been evaluated.

**Training evidence** shows published confirmation aggregates, saved validation
metrics, their trajectory, domain results, and recorded Flower participation,
steps, and weights. **Supported intents** lists the task scope. Training evidence
is recorded; query inference is live. The app never loads training, validation,
or confirmation queries. It has no bank account access and performs no banking
actions.

Loading verifies pinned assets, the checkpoint tensor hash, and the 20-intent
model contract. Evidence must identify the same selected state and version.
Changed checkpoint signatures reload the inference resource. Missing or
incompatible files show a recovery message. This console supports head-only
checkpoints; it does not load the separate MPNet B-plus-head trials.

## Inspect dropout recovery

Change the run path in the sidebar to `runs/encoder-flower-trained-recovery-v2`
and clear the optional policy path. Its warm-start checkpoint and all three
attempts retained 95.5% validation accuracy. In **Training evidence**, inspect
attempt 2: client 1 disappeared during optimization, four replies were accepted,
and each survivor received weight 0.25. Attempt 3 accepted all five clients after
fresh synchronization. These are recorded events from actual local processes,
not five physical laptops or a live network-disconnection demonstration.

Restore `runs/encoder-flower-balanced-v2` and the published policy for the query
demo. Only that selected configuration has the reported independent confirmation.

## Offline operation and verification

The server binds to loopback. Model loading uses `local_files_only=True`; no
package or model download is attempted. Queries are not written to disk by this
app. The first query includes model initialization; subsequent queries reuse it.

A real CPU parity check recomputed all 200 already-scored validation queries and
exactly reproduced cached logits and predictions. Desktop/mobile layouts and
live query submission were checked. Dashboard fixtures test behavior separately:

```powershell
python -m unittest discover -s tests -p test_encoder_dashboard.py
```

The complete experiment evidence and identities are in the
[final results](../research/results/encoder-v2/README.md). Physical multi-laptop
deployment remains a separate rehearsal.
