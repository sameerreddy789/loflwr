# Paired scope evaluator readiness

Delivery date: **4 November 2026**. This delivers software and functional checks
for a future paired evaluation. **No fresh human evaluation occurred.** The
confirmed frozen-MPNet/federated-head accuracy remains **94% (282/300)** and the
live demo keeps its existing fixed clarification policy.

The [new evaluator guide](../../../docs/fresh-scope-evaluation.md) defines one
locked comparison of margin alone versus the same margin plus the existing
scope detector. It requires 400 reviewed independent-human supported queries,
at least twenty per intent, and 200 reviewed independent-human unsupported
queries. The latter is an exploratory quota, not a power calculation or a
representative traffic mix. Raw queries remain private; the public plan binds
both policies, implementation, environment, records and provenance before
inference. The baseline and paired runners share one exposure ledger.

The result separates correct answers lost, wrong answers removed, unsupported
false accepts and ambiguity, including supported-domain coverage costs. Top-1
predictions are unchanged by the scope gate. No pooled overall-accuracy claim
is manufactured by counting rejected unsupported requests as correct intents.

## Observed functional checks

- [Offline parity](parity.json): eighteen authored, unlabeled probes, including
  a second batch and a deliberately long input. All eight checks passed:
  bitwise-equal head logits and scope scores, identical decisions, matching
  truncation, one shared encoder pass, and unchanged model identities.
- The paired path performed two encoder forwards for eighteen probes at batch
  size sixteen. The smoke tool also runs separate reference passes solely to
  compare results. Those probes are not independent accuracy observations.
- Token-length inspection produced the tokenizer's long-input warning on the
  908-token probe; actual model input was truncated to the fixed 384-token limit.
  The run completed and all three paths counted one truncation.
- [Empty collection refusal](empty-collection-check.json): the actual zero-query
  pack exited with code 2. No private evaluation, public plan or new exposure
  claim was created. Missing people, reviews and overlap inventory remain visible.
- [Verification](verification.json) records the full software suite, frozen
  evaluation locks, source hashes and the local log identity.

The numerical contract retains FP32 head scores and margin subtraction, FP64
scope coefficients/score/threshold, and lowest-label-index head ties. The fixed
predictor requests two Torch threads, but its head constructor sets the effective
count to one. The new runner explicitly checks that actual behavior. The paired
batch size is sixteen; the historical negative-only scope test used eight and
has not been recomputed.

## Remaining completion gates

The [current abstract addendum](../../abstract-alignment.md#current-alignment-addendum--4-november-2026)
documents the original proposal and the measured MPNet/head variant, preserving
the earlier LoRA review and results. It contains a portal clarification draft;
no organizer message or submission was made.

The presenter must still rehearse aloud. Two additional physical laptops and
fresh independent contributors/reviewers are still unavailable. Three-machine
LAN evidence, the combined system's fresh supported coverage, and organizer
acceptance of the disclosed architecture change therefore remain pending.

Reproduce the functional smoke with a new output path using
[check_fresh_scope_parity.py](../../../tools/check_fresh_scope_parity.py). It
uses local cached models and authored probes, not collection or spent-test data.
Do not deploy the experimental scope gate based on these software checks alone.
