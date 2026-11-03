# Future plans after the working prototype

Updated **3 November 2026**. These are prioritized next steps for the solo
builder, not completed experiments or commitments to particular scores.
The [current results](../research/results/encoder-v2/README.md) establish 94%
independent confirmation accuracy for the frozen-MPNet/federated-head variant,
95.5% reused-validation accuracy, and working five-client Flower training.
Only one physical laptop has been tested. Preserve the separate original LoRA
benchmark and its unfavorable results.

Keep the published checkpoint, model assets, policy, confirmation plan/result,
and lock identities unchanged. Give every new experiment a fresh run ID and
frozen configuration. The existing evidence remains the baseline for comparison.

## 1. Rehearse the prepared demo

Use the [morning guide](morning-demo.md) and [encoder guide](encoder-demo.md).
Show the fixed checkpoint, independent accuracy, recorded client participation,
and a new live banking query. When clarification is enabled, show answered
accuracy together with coverage. Prepare a recording as a clearly labeled
fallback, without presenting it as live training.

**Done when:** the prepared laptop starts the demo from a fresh process with
internet disconnected, inference works, the recorded dropout/rejoin is easy to
inspect, and the launch/recovery checklist is followed by the user. Automated
checks already passed; this milestone is the hands-on rehearsal.

## 2. Move encoder feature preparation onto each client

The current encoder experiment generated features centrally from a public
reference cache, then partitioned them. Add a local preparation path so each
client embeds only its own queries using the same pinned tokenizer, encoder,
pooling, normalization, and length limit. Keep the server's public evaluation
data separate from client training data. Cache features locally and share only
head updates during federation. Continue treating this as head training, not LoRA.

**Done when:** on a public parity fixture, local and reference preparation agree
within a declared numerical tolerance and produce the same evaluated predictions;
client processes can operate without other clients' query files; the training
server operates without client query/feature files; and payload inspection
confirms that training messages contain updates and metadata only. Publish
feature-preparation time, peak process memory, and identity hashes.

## 3. Rehearse three physical laptops

This depends on two additional laptops becoming available and on client-local
feature preparation. Adapt the existing Flower application to the three-client
layout: host owns lifecycle/payments, laptop 1 owns transfers/cash withdrawals,
and laptop 2 owns verification. Keep that experiment separate from the
five-client domain benchmark.

Use the [deployment guide](deployment.md), a dedicated LAN/hotspot, TLS, and
authenticated known nodes. CPU clients need at least 8 GB RAM; measure complete
process-tree memory rather than just tensor size. Cache tested dependencies and
model assets before disabling internet. Start from the planned 60-second reply
deadline and calibrate actual step budgets before the run; record any change
before collecting evidence.

**Done when:** three physical machines train; two valid survivors commit within
the declared deadline; fewer than two valid replies retain the global version;
and a dropped/rejoining device synchronizes current state before training.
Record an actual network-loss test, a fresh offline start, work completed,
participation, aggregation coefficients, measured traffic, and wall time.

## 4. Complete the experiment matrix across seeds

Freeze a new study plan using seeds **42, 43, and 44**, with matched data,
initialization, optimization, work budgets, and dropout schedules. Compare
sample weighting, equal weighting, and equal weighting plus clipping under
balanced/imbalanced data and equal/unequal budgets. Pair each weak-client
omission run with an otherwise identical full-client run, retaining independent
client random streams.

Use the [encoder configurations](../configurations/encoder/README.md) as starting
points. Freeze the encoder study's own horizon and budgets: do not silently
substitute the original SmolLM2 twenty-attempt settings for the encoder's
500-attempt regime. Keep the expanded 2,760-example data and original
400-example imbalance comparisons distinct.

**Done when:** every planned run is completed or explicitly marked failed;
accuracy, macro F1, domain gap, worst-domain accuracy, participation, work,
memory, traffic, and time include per-seed values and mean/standard deviation.
Report paired omission differences and adverse results. Three seeds provide a
variability check, not a universal convergence or fairness guarantee.

## 5. Improve task quality without reusing the final test for selection

Acquire new training/development examples for difficult intent boundaries and
independently collected or authored evaluation queries representative of the
demo's twenty intents. Deduplicate and adjudicate labels before experiments;
freeze row identities and manifests. Existing validation analysis can motivate
development work. Do not change confirmation labels, exclusions, or thresholds
to increase the existing score.

Use the [fresh-evaluation collection guide](fresh-evaluation.md) for independent
query authorship, blinded annotation, provenance, and a suggested 400-query pilot.

Evaluate training-data coverage, intent-boundary modeling, and optional
clarification/reranking on development data with matched baselines. Candidate
recall is not final top-1 accuracy; user clarification requires an end-to-end
evaluation of its own. Shared-A MPNet adapter trials so far did not outperform
the head-only model, so additional adapter tuning needs a specific hypothesis.

**Done when:** a new model and all policies are chosen on development data,
then locked before a new independent final evaluation. Report all-query
accuracy, macro F1, domain scores, and selective accuracy/coverage, including
regressions. The current 300-query confirmation set is already scored and cannot
become untouched again. Reaching 99–100% is not guaranteed.

## 6. Measure efficiency before adding complexity

On each target device, measure cold startup, local feature preparation, training
time, warm/cold inference latency, peak process-tree RAM/VRAM, raw update bytes,
serialized messages, and network bytes. Separate preparation from per-round
cost. Include repeated-query p50/p95 latency with sample count and load conditions.

Use those measurements to decide whether another encoder, smaller precision,
adapters, or scheduling changes justify their cost. Secure aggregation,
differential privacy, asynchronous federation, and heterogeneous ranks remain
separate extensions with their own threat models and accuracy/efficiency tests.

**Done when:** a reproducible report states environment pins, hardware, sample
counts, run identities, and the tradeoffs of any selected change. Data locality,
TLS, or a small payload alone does not establish formal privacy or robustness
against malicious clients.
