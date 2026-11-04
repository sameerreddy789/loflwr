# Browser rehearsal and future evaluation tooling

Completed **4 November 2026**, following commit `282751e`. This delivery adds a
separate fresh-evaluation runner and records a browser walkthrough on a fresh
local server. It adds **no new model-quality result**: the selected model remains
the unchanged frozen-MPNet/federated-head model at **94% independent confirmation
accuracy (282/300)**.

## Current readiness

| Gate | Status | Evidence or remaining requirement |
| --- | --- | --- |
| Software checks | Passed | 260 tests, zero failures/skips, 26.109 seconds in the prepared WSL environment |
| Browser walkthrough | Passed | Ten observed functional checks on a fresh port-8512 server, followed by backup-video playback |
| Fresh-evaluation software | Implemented and fixture-tested | Audited record binding, publication checks, exclusive query claims, fixed FP32 policy and separate metric denominators |
| Fresh human collection | Blocked | Zero submissions; at least 20 eligible queries per intent, 400 total, are needed under the declared collection protocol |
| Physical-laptop experiment | Pending | User reports only one laptop available; two additional machines are needed |
| Human presentation | Pending | Automated clicking/playback does not establish that the presenter has rehearsed |
| Combined scope-policy evaluation | Deferred | The new runner evaluates only the confirmed head and fixed clarification rule; the scope detector remains experimental |

See the [verification record](verification.json),
[browser observations](browser-rehearsal.json) and
[actual empty-collection check](empty-collection-check.json).

## What the browser walkthrough checked

The temporary server launched with `tools/start_encoder_demo.ps1 -Port 8512`,
using cached assets and offline model-library settings. `agent-browser` operated
the page, observed the rendered text and captured screenshots. The existing demo
on port 8511 was preserved. The temporary browser and identified server were
closed after the walkthrough.

1. The authored pending-payment query returned `Pending card payment`, with the
   expected version-500 model and the published 94%/98.65%-at-74.33% labels.
2. Whitespace-only input displayed the blank-query warning and cleared the prior answer.
3. The authored input `Help.` requested more detail without selecting a support response.
4. Five-client recorded attempt 4 displayed 2/5 contributors, quorum 3, version 3 retained.
5. Five-client recorded attempt 5 displayed 5/5 contributors, fresh synchronization and version 4.
6. Three-client recorded attempt 5 displayed 3/3 contributors, quorum 2 and version 4.
7. Three-client recorded attempt 4 displayed 1/3 contributors and version 3 retained.
8. Comparisons displayed the 33 development runs and marked the scope detector experimental.
9. The supported-intent view listed all twenty expected intents across five domains.
10. The existing 111.25-second, 1920×1080 backup played in the browser beyond
    100 seconds with no media error. This is a playback check, not a human listening review.

The recovery screens replay earlier real same-host Flower runs; no training or
new process-dropout experiment was started here. The two authored inputs and
blank-input check are functional probes, not an accuracy dataset. Browser setup
had initial off-screen/stale-target automation misses, resolved with fresh
snapshots and visible controls. No application-code change was needed.
OS-level internet isolation was not retested in this walkthrough.

## What the new evaluator can do

Follow the [runner guide](../../../docs/fresh-evaluation-runner.md). The code
recomputes the collection gate, binds the frozen rows to the original audited
adjudications, and locks source/model/data identities and numeric semantics.
Scoring requires a fresh check that the declared GitHub repository contains the
publication commit and matching plan/source bytes. It sends no query data to GitHub.

Before model construction, a shared ledger claims every retained normalized
query. Changing an output directory or collection metadata cannot silently reuse
the same questions. A started or failed inference attempt remains spent; report
regeneration uses saved predictions without another model call. Primary human
accuracy, answered-query accuracy/coverage, external/generated records,
unsupported/ambiguous outcomes, exclusions and writer concentration are separate.

Fourteen focused tests exercised synthetic collection, Git and score fixtures.
Review also caught and fixed a threshold-boundary mismatch: FP32 logits must use
FP32 subtraction to preserve the frozen policy's answer/decline decision.
The full repository suite passed 260 tests with no skips. The actual empty pack
returned exit 2 before creating any lock, query-exposure claim or model inference.

A qualifying real collection, live publication-to-scoring run and new human
evaluation are still untested. Do not label fixture scores as model accuracy.
Decide and implement any combined scope comparison before exposing a future
collection; a later policy selected on its results needs another fresh test.

## Next real action

Use the [judging script](../../../docs/judging-presentation.md) for the presenter's
rehearsal and the [collection kit](../../../docs/fresh-query-collection.md) for
independent data. Use the [laptop guide](../../../docs/encoder-laptop-rehearsal.md)
when two more machines are available. The code cannot supply those human and
hardware observations. No 99–100% accuracy claim is supported by this delivery.
