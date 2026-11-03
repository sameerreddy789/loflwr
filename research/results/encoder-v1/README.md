# Encoder accuracy recovery, 3 November 2026

Actual five-client Flower training reached **92.5% (185/200) validation accuracy**,
macro F1 **0.92504**, and worst-domain accuracy **85%**. This is a separate
frozen-MPNet/classifier-head variant; it does not train LoRA. The original
SmolLM2 B-only results remain in the [pilot report](../pilot-r2-20261103/README.md).
No model in this report has demonstrated 99% or 100% held-out accuracy.

## Data and architecture

The pinned `sentence-transformers/all-mpnet-base-v2` encoder uses revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`. Masked mean pooling and L2
normalization produce 768-dimensional features. Clients train an affine 20-class
head: 15,380 parameters, **61,520 raw FP32 bytes per direction**.

The same twenty intents use 2,760 eligible original-training queries, allocated
by domain as `[513,615,608,616,408]`. This expands the original 400-example pilot;
the accuracy difference is not an isolated algorithm comparison. The manifest
hash is `adfccc13fa4898e88f7b2fba07765694d2ef9dfa01f13768e9e3ad02389d9927`.
Validation reuses the original 200-query development set. A further 300 official
test queries remain sealed until final model/configuration selection is frozen.

Features were prepared from a public, centralized reference cache and then
partitioned. During Flower training each client loads only its own partition and
sends head parameters. This proves neither distributed feature preparation nor
physical data isolation.

## Completed measurements

| Run | Runtime and configuration | Best validation accuracy |
| --- | --- | --- |
| `encoder-flower-fullbatch5-v1` | Actual Flower, equal weights, full local batch, one SGD step/client, LR 5, weight decay 0.0004, 300 attempts | **92.5%** |
| `encoder-pilot-v1/fed-sgd1` | Sequential federation, LR 1, 4 steps, 100 attempts | 87.5% |
| `encoder-pilot-v1/fed-sgd3` | Sequential federation, LR 3, 4 steps, 150 attempts | 92.5% |
| `encoder-pilot-v1/fed-adam01` | Sequential federation, AdamW LR 0.01, 4 steps, 100 attempts | 92.0% |
| `encoder-adapter-pilot-v1/federated-lr1e4` | Live MPNet encoder training, shared frozen A, train B+head, sequential equal averaging, 30 attempts | 92.5%, selected **before adaptation** |
| `encoder-adapter-pilot-v1/centralized-lr1e4` | Pooled reference with same adapter configuration | 92.5% |
| `encoder-reference-ablation-v1` | Centralized class-balanced logistic regression, C 10 or 100 | **95.5%** |

All scores use the reused validation set for development. The adapter trials
showed no accuracy improvement over their federated-head warm start. The
eight-method reference ablation also tried logistic C 0.1/1, RBF SVC C 1/10/100,
and an MLP with training-only early stopping; full outcomes accompany this report.

The Flower accuracy run committed all 300 attempts with all five contributors.
It performed 1,500 optimizer steps and 828,000 feature presentations. Server wall
time was 641.60 seconds, excluding Ray startup and feature preparation. Peak
worker RSS was 603,234,304 bytes, excluding the encoder preparation process.
Collection median was 1.683 seconds under a 60-second deadline. These are
one-host measurements, not a network of weak laptops.

Domain accuracy was lifecycle 97.5%, payments 97.5%, transfers 85%, cash 95%,
and verification 87.5%. The selected head hash is
`7ba42b061e4af0c8c63bb07577246ee3f15ee55811fe6aa2d99157cef13bb681`;
configuration hash is `25f19d50f8b6ea5b36e9247b1c793f43c93110ab4c46c9c08dabc0b2dd70318c`.

## Recovery and adapter checks

`encoder-flower-recovery-v1` performed three real Flower attempts. All five
clients contributed in attempt 1; client 1's optimizer process was killed after
one observed step in attempt 2, and the four survivors committed; all five
rejoined after fresh synchronization in attempt 3. Collection times were
7.79/6.21/1.69 seconds. This is actual local process interruption, not a physical
laptop disconnect.

MPNet shared-A adapter smoke tests passed on CPU and GPU: base/A unchanged,
B/head changed, nonzero B gradients, and all requested steps completed. The
adapter transmits 162,836 parameters (651,344 raw FP32 bytes). Two-step smoke
loss reductions are optimization checks, not held-out accuracy results.

## Reproduction and evidence

Use the prepared Python environments from [requirements](../../../requirements/README.md).
The new tools are `prepare_encoder_data.py`, `run_encoder_study.py`,
`run_encoder_flower.py`, `run_encoder_adapter_study.py`, and
`encoder_reference_ablation.py`. They refuse run-directory overwrite and do not
evaluate the sealed confirmation set. Saved configurations identify sources and
input hashes. Windows/WSL head initialization is numerically close but not
bit-identical; use a saved common initial state for strict cross-platform pairs.

The complete WSL suite passed **143 tests, zero skips** before the next loss
change. Raw checkpoints and datasets stay ignored; the following exports retain
metrics, configurations and source-evidence hashes without query text or weights:

- [Flower accuracy](flower-accuracy.json)
- [Flower recovery](flower-recovery.json)
- [Frozen-feature optimizer trials](head-trials.json)
- [Adapter trials](adapter-trials.json)
- [CPU/GPU adapter smoke checks](adapter-smokes.json)
- [Eight centralized reference methods](reference-ablation.json)
