# Accuracy audit: partial seed-42 pilot

Audit captured on 3 November 2026 at 03:05:51 UTC. The quantitative adapter
analysis below stops at **attempt 7 / committed version 7** of
`pilot-r2-20261103-sample`, on the 200-query validation split. The live run was
already advancing to version 8 during capture. This is a fixed partial snapshot,
not a completed twenty-attempt result, final-test result, or method comparison.
No model execution or final-test access was performed for this audit.

The active pilot is **imbalanced-heterogeneous**, not balanced. Its frozen plan
uses 400 training queries: 360 for card payments and ten for each other domain.
Each minority client has only 3/3/2/2 examples across its four intents. The
balanced 800-query scenario appears below only as a separate TF-IDF reference.
The [experiment contract](../skills/banking77-experiments/SKILL.md) and
[evaluation plan](evaluation.md) require this distinction.

## What the partial run actually shows

Validation accuracy improves from 31.5% to 37.5% (63 to 75 correct out of 200),
macro F1 from 0.2447 to 0.3431, and completion task loss falls 54.5%, from 5.1089
to 2.3252. Learning is occurring. The accuracy change is uneven across domains:
payments gains 12 correct queries and transfers gains five, while lifecycle loses
five and verification loses one. Cash withdrawals gains one. The domain gap
widens from 42.5 to 52.5 percentage points.

| Attempt / version | Accuracy | Macro F1 | Validation task loss | Worst-domain accuracy |
| --- | ---: | ---: | ---: | ---: |
| 0 / 0 | 31.5% | 0.244722 | 5.108943 | 5.0% |
| 1 / 1 | 33.0% | 0.256159 | 4.387379 | 5.0% |
| 2 / 2 | 34.0% | 0.270172 | 3.821317 | 7.5% |
| 3 / 3 | 36.5% | 0.297009 | 3.307993 | 10.0% |
| 4 / 4 | 37.0% | 0.300719 | 2.885684 | 10.0% |
| 5 / 5 | 38.0% | 0.323155 | 2.636347 | 7.5% |
| 6 / 6 | 37.0% | 0.329901 | 2.452742 | 7.5% |
| 7 / 7 | 37.5% | 0.343091 | 2.325195 | 7.5% |

The fixed protocol selects the final committed checkpoint after attempt 20.
The observed peak at attempt 5 is not a replacement selection rule.

| Attempt | Card lifecycle | Card payments | Transfers | Cash withdrawals | Account verification |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 37.5% | 22.5% | 47.5% | 5.0% | 45.0% |
| 1 | 40.0% | 30.0% | 50.0% | 5.0% | 40.0% |
| 2 | 42.5% | 32.5% | 52.5% | 7.5% | 35.0% |
| 3 | 45.0% | 40.0% | 52.5% | 10.0% | 35.0% |
| 4 | 45.0% | 42.5% | 52.5% | 10.0% | 35.0% |
| 5 | 40.0% | 47.5% | 60.0% | 7.5% | 35.0% |
| 6 | 30.0% | 50.0% | 60.0% | 7.5% | 37.5% |
| 7 | 25.0% | 52.5% | 60.0% | 7.5% | 42.5% |

There are 40 validation queries per domain, so one changed answer is 2.5
percentage points within a domain. These repeated measurements of one seed and
one held-out set do not establish uncertainty or generalization across runs.

## Accepted work and data exposure

All five clients were accepted in each of the seven finished rounds: 35 accepted
replies, 224 optimizer steps, and 896 example presentations. No skipped round or
rejected client appears inside this boundary. Work from an in-progress attempt
is excluded. Microbatch size is one and accumulation is four, so a completed
optimizer step processes four examples.

| Client / domain | Unique manifest examples | Steps per attempt | Examples per attempt | Accepted steps through 7 | Example presentations through 7 | Approximate passes through 7 | Nominal passes after 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 / Lifecycle | 10 | 2 | 8 | 14 | 56 | 5.6 | 16.0 |
| 1 / Payments | 360 | 16 | 64 | 112 | 448 | 1.244 | 3.556 |
| 2 / Transfers | 10 | 8 | 32 | 56 | 224 | 22.4 | 64.0 |
| 3 / Cash | 10 | 4 | 16 | 28 | 112 | 11.2 | 32.0 |
| 4 / Verification | 10 | 2 | 8 | 14 | 56 | 5.6 | 16.0 |

“Passes” means presentations divided by manifest size, not independently
measured epoch completion. Each attempt starts a freshly shuffled deterministic
client stream. Reconstructing that stream from recorded client seeds and
processed counts yields all ten unique indices for every minority client and
273 of 360 payments indices by attempt 7. This is a deterministic reconstruction
of the specified sampler, not independent proof of what a remote client ran.
Nominal twenty-attempt values assume every client completes every budget.

More minority steps chiefly repeat the same ten queries; they cannot provide
the semantic coverage of more unique training examples. Transfers has already
seen 22.4 manifest-equivalent passes while the majority manifest has 1.244.
The differing repetitions and tiny minority intent counts make overfitting a
concrete risk, not a demonstrated explanation for every domain error.

Summed reported local training time is 337.82 seconds, collection time 442.62
seconds, and serialized requests/replies total 26,332,196 / 26,395,720 bytes
(50.29 MiB combined) within the seven-round boundary. These are application
payload bytes, not measured network-wire traffic. They exclude initial assets/A;
collection time excludes validation and is not full study wall time. Accepted
telemetry reports zero truncated training examples throughout this boundary.

## Weighting and clipping: measured facts and risks

The active sample baseline uses coefficients `[0.025, 0.9, 0.025, 0.025, 0.025]`.
Each validation domain contributes 20% of queries, while payments owns 90% of
the training manifest and receives 90% of the aggregation coefficient. Its
larger local budget also produces the largest observed delta norm.

| Client | Round-1 delta norm | Sample coefficient | Coefficient × delta norm |
| --- | ---: | ---: | ---: |
| 0 | 0.145711 | 0.025 | 0.003643 |
| 1 | 0.914509 | 0.900 | 0.823058 |
| 2 | 0.510735 | 0.025 | 0.012768 |
| 3 | 0.268017 | 0.025 | 0.006700 |
| 4 | 0.149991 | 0.025 | 0.003750 |

Payments represents 96.48–96.84% of the sum of coefficient-weighted delta norms
over these seven rounds. That quantity describes contribution-norm bounds;
directions may align or cancel, and it is not a causal percentage attribution
of predictions, quality, or the aggregate vector. The observed improvement in
payments and deterioration in lifecycle are consistent with majority influence,
but this partial run alone does not establish causality.

**Median delta clipping is not active in this measured run.** Its recorded clip
factors are all one. The main `equal_clipped` run had not started at the initial
audit inspection, so its performance cannot explain the sample baseline's
scores. Per-optimizer-step gradient clipping at norm one is a separate operation
used by all methods. Recorded nonzero gradient norms and falling validation loss
provide no evidence that learning has stopped.

For a mathematical illustration only, applying median clipping to the recorded
round-1 delta norms gives threshold 0.268017. Payments would retain about 29.3%
of its norm and transfers about 52.5%; the two smallest updates are unchanged.
Plain equal coefficients would reduce payments' share of the sum of weighted
norm bounds to 46.0%, and equal plus clipping to 24.4%. These are arithmetic
counterfactuals on the sample run's deltas, not measured alternative trajectories
or quality results. Subsequent local updates depend on the changed global B.

Clipping can suppress harmful majority dominance, but it can also suppress useful
work from the larger, more diverse client. Equal weighting can amplify updates
fit to only two or three examples per intent. A completed plain-equal control is
needed to separate coefficient effects from clipping effects; neither a benefit
nor a failure of clipping can currently be claimed.

## The superseded pilot is separate infrastructure evidence

`pilot-20261103-sample` committed only attempt 1 before infrastructure failure
and interruption. That round accepted clients 0, 2, and 3; clients 1 and 4 were
rejected with progress-file `FileNotFoundError` failures. The sample coefficients
renormalized to one third each because all three survivors owned ten examples.
Only 14 optimizer steps / 56 example presentations entered that committed
update, versus 32 steps / 128 presentations in active pilot round 1.

Its validation accuracy changed 31.5% → 31.0%, macro F1 0.244722 → 0.236338,
and loss 5.108943 → 4.861978. This does not measure the intended five-client
sample-weighted baseline. The attempt-2 checkpoint counter is not evidence of
a second committed update; the committed version remained one. Do not pool the
old pilot with the revised identity or use its score as a valid matched result.

## Separate fixed TF-IDF validation reference

The existing `tools/classifier_reference.py` results use the same fixed
word unigram/bigram TF-IDF plus logistic-regression settings: sublinear TF,
`C=1`, `max_iter=1000`, inverse-class-frequency loss weights, and random seed 42.
Both records identify scikit-learn 1.9.1 and the same 200-query validation hash.

| Training scenario | Training queries | Validation accuracy | Macro F1 | Worst-domain accuracy |
| --- | ---: | ---: | ---: | ---: |
| Imbalanced | 400 | 41.5% | 0.360030 | 20.0% |
| Balanced | 800 | 87.0% | 0.868288 | 77.5% |

| Scenario | Lifecycle | Payments | Transfers | Cash | Verification |
| --- | ---: | ---: | ---: | ---: | ---: |
| Imbalanced | 25.0% | 97.5% | 20.0% | 20.0% | 45.0% |
| Balanced | 87.5% | 95.0% | 82.5% | 92.5% | 77.5% |

Sources are [imbalanced reference](results/tfidf-validation.json) and
[balanced reference](results/tfidf-balanced-validation.json). Original local files
are `runs/tfidf-imbalanced-validation.json` and
`runs/tfidf-balanced-validation-20261103.json`. The balanced
run changes both total training size (400 → 800) and per-domain coverage; majority
payments falls from 360 to 160 examples while each minority rises from ten to 160.
This supports data regime as a serious accuracy factor, and shows that fixed
class weights cannot create missing examples. It does not isolate balance from
data quantity, establish an FFA/LoRA effect, or supply a federated-method result.

## Next checks justified by the evidence

1. Finish and retain the frozen twenty-attempt sample and main runs, including
   unfavorable scores and actual participation/work. Preserve the final-checkpoint
   rule; do not select attempt 5 because its intermediate accuracy was highest.
2. Complete the declared plain-equal control before attributing an effect to
   clipping. Compare domain outcomes, actual accepted work, norms, and failures
   within the same scenario and seed. The selected subset is not the full matrix.
3. If the practical objective is higher task accuracy, evaluate balanced data in
   a separately frozen, validation-only calibration or declared full-matrix run.
   Do not mutate this active plan or represent a 400-versus-800 comparison as an
   isolated weighting experiment.
4. Treat more steps on ten-example clients as an overfitting risk. Prioritize
   coverage and saved validation confusions before increasing those budgets.
5. A falling completion loss need not produce the same-sized improvement in
   twenty-way argmax accuracy. Diagnose the fixed scoring rule and validation
   error distribution separately before changing learning rate, rank, or prompt.
   Any changed configuration needs its own identity and validation-only selection.

## Reproducible identities

| Field | Recorded value |
| --- | --- |
| Active study plan SHA-256 | `82a05841701de0e477555e67031c65734a8d791c0434eb1f9b4bc13062137b7a` |
| Active sample frozen configuration SHA-256 | `464ea6cd3364e1f5e8c67c6683779730a47d2889603fb0a81e09b8a3942c7fd5` |
| Active sample coordinator configuration SHA-256 | `01cf4faea5f733adb116a6bc7eb900b77e0b9484dcdbfbfd18d398a083861c8f` |
| Attempt-7 evaluated B hash | `8e2883facc47589e4f5dc2dc1e520d5ba8405ab225a25abb58925640ac9fb855` |
| Superseded plan SHA-256 | `ccf4796944df552b8342189146de3fbbe02e55bc5647e53b1aae55160b8c112d` |
| Model / revision | `HuggingFaceTB/SmolLM2-135M-Instruct` / `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Common A hash | `9b34c3898b65789470bf19c062e613239585d2092e4be89e2956f8cce1f6f409` |
| Initial B hash | `1f8c5306739b608bf948b536d550af04b62b348c54712e163627dc125f2298e2` |
| Manifest set hash | `ae4831d655b0f0a0a56039d39b97357e4d25ae35be2236c63fca15899b1465fc` |
| TF-IDF imbalanced configuration hash | `926cfc3cc1707b5ea8e0fcfa9740f779260b1c2147292da5c410d654cda3bc10` |
| TF-IDF balanced configuration hash | `67bd579c2a5303402a2543edd54e68e6d6c88bfe064e02a4251eb93fc865ed27` |
| Both TF-IDF validation records hash | `93bee22f2479d7dca75eca0b65a8923d4de36305b0c4babcb565842957759a9e` |

The active configuration fixes seed 42, 20 attempts, a 300-second collection
deadline, optimizer reset per attempt, AdamW learning rate `2e-4`, weight decay
zero, gradient norm limit one, microbatch one, accumulation four, and maximum
length 128. Shared frozen A and B-only rank-8 updates target `q_proj`/`v_proj`,
alpha 16, dropout 0.05, and 184,320 B parameters. Scoring sums completion log
probabilities including EOS. The frozen plan records source and environment
identities; configuration hashes above distinguish the frozen launcher input
from the coordinator wrapper rather than treating them as interchangeable.

The [published audit snapshot](results/accuracy-attempt-7-audit.json) contains
compact validation/work/norm facts and hashes of the source bytes read. Its
original local source is `runs/accuracy-audit/facts-through-attempt-7.json`.
Those files were read while the study was running; restricting calculations to
finished attempts 1–7 avoids interpreting future counters or pending replies as
applied work. Original checkpoints, manifests, queries, and private keys are not
copied into this report.
