# Recovering task quality without losing the federated-learning contribution

Decision note: **3 November 2026**. This is a proposed, separately identified
architecture experiment, not a change to the active SmolLM2/B-only contract.

## Recommendation

The current 34.4% test accuracy is insufficient for a convincing banking-intent
application. Keep the working federated protocol, local-data ownership,
deadline/quorum logic, update validation, recovery, telemetry and dashboard.
Prioritize the representation and classification objective before spending more
time tuning aggregation. Do not promise 99–100% unseen-query accuracy.

The strongest evidence for the next experiment is the frozen MPNet **linear
classifier's 94% validation accuracy** with 2,760 eligible original-training
queries. Its parameter-update interface is closer to federation than the
**95.5% nearest-example reference**, which depends on a centralized example
index. Both are references on reused validation, not final-test or federated
measurements. See the [feasibility review](accuracy-99-feasibility.md).

Start with a federated frozen-encoder/classifier-head experiment to establish
task quality cheaply. Then compare a shared-A encoder-LoRA-plus-head variant
if the first gate succeeds. This makes adapter learning a measured part of the
proposal rather than assuming it improves an already useful representation.
Head-only training is model specialization, but it is not LoRA.

The final-checkpoint scoring check is now complete on the same 200 validation
queries. Content-free centering improves sample/equal/clipped models from
22.5%/34.5%/32.5% to **52%/63%/58.5%**. All original-score accuracies and B hashes
match their recorded version-20 validation evaluations. This validates the
diagnosis of scoring bias, while showing that the correction alone does not
deliver high-quality classification. See the [completed diagnostics](results/README.md).

## What the completed pilot establishes

The [seven-configuration seed-42 pilot](results/pilot-r2-20261103/README.md)
completed its declared horizon. It is not the full multi-seed program.

| Configuration | Final-test accuracy | Account-verification accuracy |
| --- | ---: | ---: |
| Sample weighting | 24.6% | 10% |
| Equal weighting | 35.4% | 36% |
| Equal plus clipping | 34.4% | 32% |
| Equal plus clipping with dropout | 33.4% | 31% |
| Equal plus clipping, client 4 omitted | 34.8% | 32% |

Clipping did not improve quality over plain equal weighting in this seed.
Including weak client 4 did not improve its domain accuracy versus omission,
and overall accuracy was 0.4 points lower. This is a preliminary null/unfavorable
utility result, not evidence that weak clients can never help. The two runs
also differ in membership, coefficients and clipping threshold; omission does
not isolate hardware speed as a cause. Report these outcomes openly.

Dropout handling succeeded mechanically: all twenty attempts committed and
the interrupted client rejoined after synchronization. That is useful system
evidence, but it does not solve the task-quality deficit.

## Experiment order and decision gates

The numbers below are **internal development goals**, not organizer requirements,
published results, or forecasts. The supplied problem statement has no stated
99% accuracy threshold, and judge acceptance cannot be inferred from a score.

| Order | Work | Decision gate |
| --- | --- | --- |
| 1 — complete | Repeat the same three fixed scoring diagnostics on final SmolLM checkpoints. Preserve the original results and scoring contract. | Best corrected final-checkpoint accuracy is 63% validation. Time-box this route; no high-accuracy result is established. |
| 2 | Create a new manifest from the 2,760 eligible training examples, partitioned by the existing five domains. Retain frozen validation/test exclusions and original source IDs. | Complete disjointness/count/hash audit; no held-out query or label enters training. |
| 3 | Fit a fresh twenty-way head on locally computed frozen MPNet features; compare matched centralized and federated training. | First reach at least 90% validation with acceptable per-domain scores; aim toward the measured 94% centralized reference. No guarantee is implied. |
| 4 | Test shared-A encoder LoRA plus the head using twenty-class cross-entropy. Compare centrally first, then under federation with matched initial state and work. | Aim for 95% validation after the first 90% gate, within actual CPU/RAM/traffic limits. These are targets; a negative adapter result remains visible. |
| 5 | Compare sample weighting, equal weights and clipping under the same task-quality setup, then dropout and weak-client omission. | Measure the gap to the matched centralized model; an internal target is within 2–3 percentage points, with domain scores and uncertainty reported. |
| 6 | Investigate training-only hard-negative pairs and a candidate reranker only if remaining errors justify it. | Measured end-to-end top-1 improvement and acceptable latency; top-k recall is insufficient. |
| 7 | Freeze the selected method, repeat seeds and evaluate independent confirmation data. | A 99% claim requires the actual result, intact labels and disclosed sample uncertainty; unfavorable results stay published. |

For the working-demo checkpoint, prioritize steps 1–5 over an open-ended search
for a perfect validation score. Set a short implementation budget for the first
federated-head gate before committing to the encoder-adapter variant. If that
gate fails, debug central/local loss, feature handling, class coverage and local
work rather than masking the deficit with a centralized inference component.

## The new variant's training and privacy contract

- Use the already pinned `sentence-transformers/all-mpnet-base-v2` revision
  `e8c3b32edf5434bc2275fc9bab85f82640a19130` as the candidate encoder.
- Train a twenty-class cross-entropy classifier instead of generating intent
  names. All twenty logits must participate in the loss, including labels absent
  from a particular client's local domain.
- Start centralized and federated experiments from the same **fresh** head and
  encoder state. Do not seed federation with a centrally fitted 94% head and
  present the result as learning from decentralized data.
- In the frozen-encoder variant, compute and cache each client's own training
  features locally. Send only classifier parameter updates. Never collect the
  nearest-example index or individual training embeddings at the server.
- A proposed 768-feature, twenty-class affine head has 15,380 parameters,
  approximately 60.1 KiB in FP32. This is arithmetic, not a measured transport
  payload, latency or memory result.
- In the encoder-LoRA variant, centrally initialize and distribute common A,
  train B plus the head, and create a new model/schema/scaling identity. This
  explicitly changes the current B-only message contract. Reuse the generic
  response-validation and quorum principles, not the old tensor identity.
- Encoder-adapter training requires fresh forward passes. Cached features from
  the frozen encoder cannot be reused after its effective weights change.
- Measure CPU training, whole-process peak RAM, inference latency and actual
  payloads. Use real step budgets. Recalibrate deadlines for this model.
- Local data and local feature caches do not provide differential privacy.
  Parameter updates can still leak information; preserve the existing privacy
  limitations and do not claim secure aggregation unless implemented.

## Data coverage and a fair high-accuracy demonstration

The original 90% imbalance leaves only two or three examples per minority
intent. Keep it as an explicitly difficult stress condition. It should not be
the only demonstration of useful learning. The expanded-data condition keeps
the same twenty intents and held-out separation while providing substantially
more local phrasing coverage. Compare methods within each condition; changing
the amount and balance of data together does not isolate either cause.

Use original-training data for hard-negative mining, label-definition design,
augmentation or pair training. If synthetic examples are later introduced,
record their provenance and measure their effect separately. Do not generate
training examples from inspected validation/test mistakes or silently change
held-out labels to improve the score.

The already-inspected 500-query test cannot become an untouched confirmation
set again. Keep its historical comparisons. Lock a previously unscored holdout
or collect independently reviewed queries before the final selected-method
confirmation. This does not replace the original benchmark or erase its results.

## How 99–100% should be treated

The current reference gets the right intent among its first two suggestions on
198/200 validation queries. A restricted reranker would have to choose correctly
on **all 198** to turn that into 99% accuracy. No such reranker has been measured.
Several remaining errors involve missing context or suspected annotation
inconsistency; selected-error inspection does not establish a numerical ceiling.

Clarification can make the product more useful on ambiguous queries. Evaluate
it separately as correct answers among accepted queries **and** coverage,
deferral rate and end-to-end outcomes. High selective precision must not be
renamed overall accuracy. A scripted demo or memorized set of known questions
cannot establish generalization.

For judging, demonstrate useful held-out classification, an actual update from
a weak client, visible influence, and dropout/rejoin behavior with measured
resource costs. A stronger application and honest comparative evidence are
the next deliverables; a near-perfect number must be earned by the experiment.
