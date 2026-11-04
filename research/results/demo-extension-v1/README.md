# Demo readiness and fixed-work study delivery

Delivered **4 November 2026**. This extension adds an automated local preflight,
a 1:51 narrated backup, and a completed nine-run development study. The confirmed
query model remains frozen MPNet plus its federated linear head, version 500:
**94% independent confirmation accuracy (282/300)**. Head-only training is
distinct from the preserved SmolLM2 LoRA benchmark.

## What to use

| Deliverable | Entry point | Verified boundary |
| --- | --- | --- |
| Demo preflight | [Guide](../../../docs/demo-preflight.md), [PowerShell command](../../../tools/check_demo.ps1) | All five checks passed: Python/dependencies, evidence, matching local dashboard configuration and actual cached inference |
| Narrated backup | [MP4](../../../presentation/narrated-demo.mp4), [script and reproduction guide](../../../presentation/narrated-demo.md) | 111.25 seconds, offline synthetic narration, burned captions, explicit recorded labels, actual browser clips and published evidence |
| Fixed-work study | [Report](../resource-frontier-v1/README.md), [frozen plan](../resource-frontier-v1/plan.json), [all results](../resource-frontier-v1/summary.json) | Nine fresh serial CPU processes, three seeds, equal total optimizer work; reused validation only |
| Software verification | [Verification record](verification.json) | 246 passed, zero failed/skipped, 29.359 seconds in the prepared WSL environment |

The latest [preflight export](preflight.json) is a point-in-time check, not a new
evaluation. It returned `pending_card_payment` for the authored input “My card
payment is still pending.” Its load/query times and sampled RSS are diagnostics,
not the controlled resource measurements below. Both the original and reviewed
preflight executions passed; the published export is the reviewed second run.

## The resource result and its implication

Every cell completed 500 optimizer steps per client, with five clients and the
same cached training features. Only local steps between equal-weight averages
changed. All three seeds produced these final validation accuracies:

| Steps per average | Rounds | Validation accuracy | Estimated raw upload + download, all clients | Mean observed process wall time |
| --- | --- | --- | --- | --- |
| 1 | 500 | 95.5% | 307.6 MB | 85.50 s |
| 2 | 250 | 93.5% | 153.8 MB | 48.48 s |
| 4 | 125 | 93.5% | 76.9 MB | 28.37 s |

Four steps saved **75% of the calculated raw update payload**, with a **2-point
validation accuracy loss** in each seed. On accuracy and raw payload, the
two-step schedule is dominated by four steps in these observations. Memory and
wall-time observations have separate limitations; the full report publishes all
seed values, sample standard deviations and trajectories.

No transport was exercised in this new study. MB is decimal; the byte formula
omits encoder/data transfer, message metadata, TLS framing and retransmissions.
Process time/RSS include imports and all cached feature splits but exclude
feature preparation and other processes. This cannot establish physical-laptop
speed or memory costs. It does not justify changing the already-confirmed model.

## Verification and preserved evidence

The required command was `python -m unittest discover -s tests` in the prepared
Ubuntu WSL2 environment. It passed **246 tests with no failures or skips** in
29.359 seconds. The extra checks cover preflight failure/identity cases and
fixed-work plan/work/payload boundaries. Fixture logs are software evidence,
not banking accuracy measurements.

All **24 original confirmation**, **36 external CLINC**, and **40 scope** locked
file hashes matched. These sets overlap; they are not 100 unique files. The
selected checkpoint, thresholds, dataset identities and scoring sources remain
unchanged. The original six-slide deck and silent browser recording also retain
their earlier hashes and [delivery record](../demo-delivery-v1/verification.json).

The video was fully decoded and captioned frames from every scene were visually
reviewed. Its [manifest](../../../presentation/narrated-demo-manifest.json)
records codecs, source intervals, hashes and offline synthetic-voice settings.
Play it with sound during the human rehearsal; file/audio checks do not establish
that the presenter has rehearsed.

## Remaining gates

Follow the [updated future plan](../../../docs/future-plans.md): rehearse with
the actual presenter, obtain two additional laptops for a physical network run,
and collect independent fresh supported queries before another model or combined
scope-policy claim. The scope detector remains experimental. **99–100% accuracy
is still unachieved.**

Model weights, raw queries, private node keys, logs and full run directories stay
out of Git. The public exports carry run/configuration identities and hashes;
reproduction on a fresh checkout needs the documented cached assets and tested
environments.
