# Local encoder bucketing: measured development results

Measured **4 November 2026** on one AMD Ryzen 5 6600H laptop under Ubuntu WSL2,
using CPU FP32 and one Torch thread. This implements the hypothesis in the
[efficiency review](../../efficiency-paper-review.md): tokenize once, batch
similar exact token lengths, and restore source row order before caching.
The confirmed model, inference code, head weights and prior evaluation sets
remain unchanged.

## Encoding-time comparison

The [locally frozen plan](plan.json) declared both complete cohorts and all six
paired cells before execution. Each cell used a fresh Python process, the same
pinned model and batch size eight, warmups, and alternating backend order.
The [complete summary](summary.json) retains every repetition and parity result.

| Cohort | Queries | Original median | Bucketed median | Median time reduction | Padded-token reduction |
| --- | --- | --- | --- | --- | --- |
| Reused twenty-intent validation | 200 | 58.72 s | 35.49 s | **39.56%** | **48.33%** |
| Weak account-verification client 4, training | 408 | 57.20 s | 50.14 s | **12.34%** | **31.31%** |

These are ratios of median encoding times, not median paired speedups or a
guarantee for another machine. Encoding timing includes tokenization, sorting,
padding, pooling, order restoration and forward-shape instrumentation. It
excludes model loading/verification, head scoring and the surrounding audit.
Model-loading time is recorded separately in every cell.

| Cohort / repetition | First measured backend | Original | Bucketed |
| --- | --- | --- | --- |
| Validation / 0 | Original | 58.72 s | 35.49 s |
| Validation / 1 | Bucketed | 38.97 s | **42.04 s** |
| Validation / 2 | Original | 64.08 s | 26.60 s |
| Client 4 / 0 | Bucketed | 57.20 s | **100.12 s** |
| Client 4 / 1 | Original | 39.68 s | 39.22 s |
| Client 4 / 2 | Bucketed | 64.68 s | 50.14 s |

**Two repetitions were slower with bucketing.** Both cohort medians improved,
but the timing variation is large. The laptop remained shared with desktop and
background activity. One host snapshot during the study recorded 73.8% total
CPU use and 88.7% memory use while plugged in ([snapshot](host-observation.json)); it is not a continuous monitor or
an energy measurement. Three repetitions are a limited engineering comparison,
not evidence of a universal speedup or statistical significance. Run order is
balanced over all six cells, but each individual three-repeat cohort has a 2:1
first-backend split. Keep the new recipe **opt-in** and profile the target device.

## Work and numerical checks

The actual forward tensor shapes matched the declared arithmetic in every run:

| Cohort | True token positions, including special tokens | Original padded positions | Bucketed padded positions | Attention-cell reduction |
| --- | --- | --- | --- | --- |
| Validation | 3,360 | 7,168 | 3,704 | 64.00% |
| Client 4 | 5,099 | 8,048 | 5,528 | 45.54% |

Attention cells mean the sum of `examples_in_batch × padded_length²` over batches.
They are a work proxy, not measured FLOPs or energy. No query was dropped or
shortened relative to the fixed 384-token rule; these two cohorts had zero
truncations in either path.

All six comparisons passed the predeclared `1e-5` absolute feature tolerance,
with **zero changed top-1 predictions and zero changed strict-margin decisions**.
The largest feature difference was `1.2386590242385864e-7`; the largest logit
difference was `3.337860107421875e-6`. Rebatching is therefore not claimed to be
bitwise identical. The 608 unique queries were repeated for timing; they do not
become 1,824 independent accuracy observations.

Maximum sampled process RSS across the cells was 1,194,168,320 bytes. This
includes the encoder/runtime, allocator state and retained outputs; it does not
include the browser or whole laptop. Pair order affects memory allocation, so
no memory-saving claim follows from these measurements.

## Scope, integration and reproduction

The new [preparation CLI](../../../tools/prepare_encoder_bucketed.py) has distinct
plan, registration, recipe and dataset identities. It uses the existing
ownership validators and restores original row/label alignment. Original caches
and the confirmed predictor remain available unchanged. Follow the
[usage guide](../../../docs/bucketed-feature-preparation.md) for new local runs.

The [full preparation/one-round check](integration.json) completed all 2,960
training/validation records in six serial fresh processes, but **failed the
strict compatibility gate**. All row IDs and labels matched; all top-1 predictions
were unchanged. One of the 200 reused-validation queries changed its fixed
answer/clarify decision. All five training clients had zero such changes.

| Integration check | Observed result |
| --- | --- |
| Complete role counts | 513 / 615 / 608 / 616 / 408 training rows; 200 validation rows |
| Source IDs, labels and order | Exact for all 2,960 rows |
| Maximum absolute feature / logit differences | `1.257285475730896e-7` / `4.0531158447265625e-6` |
| Fixed-head top-1 changes | **0 / 2,960** |
| Fixed strict-margin decision changes | **1 / 2,960**, in reused validation; gate required zero |
| Authored tokenizer boundary probes | All four passed exact IDs/masks; includes 384-token and over-limit inputs |
| Real head optimization | One SGD step per client in each arm; all heads changed; equal 0.2 aggregation coefficients |
| Largest client-delta / aggregate-state differences | `2.384185791015625e-7` / `4.6566128730773926e-8`, within `1e-5` |
| Post-update validation top-1 changes | **0 / 200**, including comparison on common original features |

The six timing pairs compare freshly encoded original/candidate features. This
separate integration compares the new recipe against the **previously stored
reference caches**, which is the stronger migration check. Small FP32 differences
can cross a sharp clarification threshold even when every intent prediction is
identical. The failed result is retained; no threshold, tolerance or source label
was changed to turn it into a pass.

The [cached-feature diagnostic](boundary-diagnostic.json) found that the old
margin was exactly the threshold, `2.1202011108398438`; the candidate margin
was `2.1202001571655273`. That four-FP32-ULP decrease changed **answered to
abstained** under the unchanged `>=` rule, while keeping both leading classes
in the same order. It used two head passes on cached validation features and
no encoder inference or ground-truth labels. Historical and current cache
generation also differ in runtime/build provenance, so this does not isolate
one unique low-level cause of the rounding difference.

Use this recipe only for separately identified development experiments. It is
**not a compatible replacement for the confirmed model's calibration caches or
inference path**. Keep the original preprocessing and fixed decision rule for
the confirmed demo. A future adoption needs its own calibration and evaluation.
The actual one-round optimization check passed, but it establishes neither
convergence nor Flower transport or physical-laptop behavior.

Integration lock:
`dc2cad90667693548b7a2b1d1c4a8a2a3489a16728e563e950565ee2cf3317b4`.
The auditor explicitly collects these public benchmark features on one host;
it is not a new data-isolation test or independent confirmation run.

The encoding study used Python 3.11.15, Torch 2.8.0+cu126 in CPU mode,
Transformers 4.57.1, tokenizers 0.22.2 and NumPy 2.4.6. Full model/data/source and
runtime identities are in the plan. Its lock is
`f8f118d90199c5c1cf07cc0eda3e4324b25f81e23cc1165c58e9b492f6c893b7`.
This plan was frozen locally before a development measurement; it is not a
pre-published independent-test lock.

The change does not reduce the head's 61,520 raw update bytes, change local
training budgets, add LoRA to the head variant, establish faster single-query
inference, or raise confirmed accuracy. The existing independent result remains
**94% (282/300)**. The [model card](../../../MODEL_CARD.md) summarizes its intended
use, per-domain results and limitations.

The [delivery verification](verification.json) records **300 passing software
tests with zero failures and zero skips**, in 67.635 seconds under the prepared
WSL environment. The suite includes 27 new preparation, benchmark and integration
fixtures. These are included in the 300, not additional tests; they do not erase
the real-model strict compatibility failure above.
