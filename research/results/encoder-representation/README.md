# Encoder representation diagnostics

**3 November 2026.** Both three-epoch adaptation variants reached **96.0%
validation accuracy (192/200)**, versus the published head's 95.5% (191/200).
This is one extra correct development query in one seed. It is not a new
independent score, a federated result, or evidence of 99–100% accuracy.
The selected Flower demo remains at its measured **94% confirmation** result.

A subsequent fixed ten-epoch comparison also peaked at **96%**. Its final
two-layer score fell to 95.5%; adapting all twelve transformer layers ended at
94.5%. These four tested configurations did not achieve 99–100%. The modest
development gain does not justify claiming a stronger confirmed system.

## Three-epoch comparison

The [complete export](three-epoch-results.json) preserves the plan, all evaluated
epochs, initial-head identity, unchanged frozen-layer hashes, changed trainable
layer hashes, memory observations, and original source hashes.

| Variant | Epoch 0 | Epoch 1 | Epoch 2 | Epoch 3 | Trainable parameters | Peak allocated VRAM |
| --- | --- | --- | --- | --- | ---: | ---: |
| Last two MPNet layers + head | 95.5% | 95.0% | 95.5% | 96.0% | 14,191,124 | 746,747,392 bytes |
| Last four MPNet layers + head | 95.5% | 95.5% | 95.5% | 96.0% | 28,366,868 | 1,016,492,544 bytes |

Both selected epoch 3, with macro F1 0.96005 and worst-domain accuracy 90%.
The runs completed 261 optimizer steps each in 43.01/54.60 seconds after model
preparation. These are observed local training/evaluation times, not a controlled
throughput comparison. Process RAM and whole-network traffic were not measured.

This diagnostic trains on the pooled 2,760 public training records and evaluates
only the reused 200-query validation split. It freezes model/optimizer choices
before scoring: seed 42, last two versus four layers, three epochs, FP32, AdamW
persisting across batches/epochs, encoder LR 1e-5, head LR 1e-3, decay 0.01,
microbatch 8, effective batch 32, and gradient clipping at 1. Class weights use
training counts only. Frozen lower layers remain in evaluation mode; the adapted
layers use training-mode dropout. Initial live logits agree with the cached
published head within 1e-3 and have identical predictions.

The 1,800-second per-variant guard is a cooperative optimization budget checked
between update groups, not a hard process deadline. All three epochs completed.
No old test or confirmation query was opened. Original assets, checkpoint,
thresholds, and evaluation sources are unchanged.

## Ten-epoch follow-up

After the short study completed, a new plan fixed ten epochs and two versus
twelve trainable transformer layers before evaluating that follow-up. This is a
sequential development decision, not a claim that both studies were registered
together. The embedding layer remains frozen even in the twelve-layer variant.

| Variant | Selected epoch | Best validation | Final validation | Trainable parameters | Peak allocated VRAM | Training/evaluation time |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Last two layers + head | 3 | 96.0% | 95.5% | 14,191,124 | 748,595,712 bytes | 127.17 s |
| All twelve transformer layers + head | 5 | 96.0% | 94.5% | 85,069,844 | 2,089,741,312 bytes | 317.61 s |

Each completed **870 optimizer steps**, with the same training data, seed,
microbatch/effective batch, learning rates, precision, and optimizer as the short
comparison. The [complete ten-epoch export](ten-epoch-results.json) includes
every evaluated epoch and both favorable and unfavorable outcomes. Its source
adds explicit equality checks between validation feature IDs/labels and the
validation text records, and exposes epoch/layer counts as command arguments.

The larger variant's trainable state would occupy **340,279,376 FP32 bytes** if
transmitted. This arithmetic estimate is over 5,500 times the deployed head's
61,520 bytes; the diagnostic did not transmit it. It is not a practical drop-in
replacement for the current edge federation. More training or more trainable
layers did not reliably improve development accuracy in this comparison.

## Interpretation and reproduction

The new representation can change one previously incorrect development decision,
but it costs far more trainable parameters than the 15,380-parameter head. If all
those parameters were transmitted in FP32, they would occupy 56,764,496 or
113,467,472 bytes per direction; these are arithmetic sizes, not measured traffic.
No Flower transport or physical-device experiment was performed for these models.

Use this result to decide whether further representation learning is worth
testing. It does not justify replacing the confirmed demo model or presenting
96% as an independently tested score. Future selection requires a new independent
evaluation set, not repeated use of the now-scored 300-query confirmation.

```sh
python tools/run_encoder_representation_study.py --output runs/representation-fresh
python tools/run_encoder_representation_study.py --output runs/representation-long-fresh --epochs 10 --trainable-layers 2 12
```

Outputs must be fresh directories inside ignored `runs/`. The original script
and dependencies are identified by hashes in the export. The source repository
retains the experiment's code history: the short run used commit `022fc9c`'s
script. [Verification](verification.json) records the earlier 159-test result
and the latest **160 passing full-suite tests, zero skips**, including checks
for data origin/overlap, cached-label alignment and gradient freezing. All 24
files in the original confirmation lock still match. Those checks are separate
from model-quality evidence.
