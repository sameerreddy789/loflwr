# Encoder representation diagnostics

**3 November 2026.** Both three-epoch adaptation variants reached **96.0%
validation accuracy (192/200)**, versus the published head's 95.5% (191/200).
This is one extra correct development query in one seed. It is not a new
independent score, a federated result, or evidence of 99–100% accuracy.
The selected Flower demo remains at its measured **94% confirmation** result.

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
```

Outputs must be fresh directories inside ignored `runs/`. The original script
and dependencies are identified by hashes in the export. The source repository
retains the experiment's code history. [Verification](verification.json) records
**159 passing full-suite tests, zero skips**, including two new data-boundary and
gradient-freezing tests. Those tests are separate from model-quality evidence.
