# Reproduce the encoder-head experiments

These configurations use the prepared `assets/encoder-bank20-v1` dataset and
pinned MPNet feature identity. Run from the repository root in the tested
environment. Every output path must be fresh; rename it before repeating.

Sequential mathematical federation:

```sh
python tools/run_encoder_study.py --config configurations/encoder/head-equal.json
python tools/run_encoder_study.py --config configurations/encoder/head-clipped.json
```

Actual Flower ClientApp/ServerApp transport (tested in Ubuntu WSL2):

```sh
python tools/run_encoder_flower.py --config configurations/encoder/flower-head-equal.json
python tools/run_encoder_flower.py --config configurations/encoder/flower-head-clipped.json
```

The Flower launcher derives and records the common encoder/head contract from
the frozen manifest and head configuration when no explicit identity is supplied.
Clients validate it before work. No confirmation data is opened by these tools.

The selected 95.5% development head first appeared at attempt 500 in a longer
1,000-attempt sequential study. These 500-attempt configurations reproduce that
chosen development horizon. This distinction is recorded in the
[full results](../../research/results/encoder-v2/README.md).

After all model/configuration choices are final, the confirmation utility has
two separate steps. The first locks the completed checkpoint, source files,
manifest metadata and reason for selection without loading confirmation queries:

```sh
python tools/confirm_encoder.py freeze --run-dir runs/CHOSEN-COMPLETED-RUN --output runs/confirmation --reason "Final validation-only selection"
python tools/confirm_encoder.py evaluate --output runs/confirmation
```

Do not use the resulting confirmation score for further model selection. The
tool rejects modified locked files and returns a saved result on an unchanged
repeat invocation. Weights and queries remain in ignored local directories.
