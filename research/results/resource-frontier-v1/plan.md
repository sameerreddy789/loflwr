# Fixed-work classifier-head resource study

This is a new development study on the reused 200-query BANKING77 validation
split. It does not read or score the 300-query confirmation split, the original
test, CLINC, or scope-v1. It does not change the confirmed demo model or any
threshold. The original benchmark and later locked evaluations remain separate.

The dataset is `assets/encoder-bank20-v1`: cached features from frozen
`sentence-transformers/all-mpnet-base-v2`, with 2,760 training examples divided
between five domain clients (513, 615, 608, 616, 408). Only the FP32 affine
768-to-20 head is trained. No transformer layers or LoRA factors are updated.
This measures sequential numerical federation on one host, not Flower or five
physical devices.

The full 3-by-3 design is declared before execution:

| Local optimizer steps/client/round | Rounds | Steps/client | Total steps | Total full-batch examples |
| --- | --- | --- | --- | --- |
| 1 | 500 | 500 | 2,500 | 1,380,000 |
| 2 | 250 | 500 | 2,500 | 1,380,000 |
| 4 | 125 | 500 | 2,500 | 1,380,000 |

Run every row at seeds 42, 43 and 44, loading the same saved seed-specific
initial tensors across schedules. All five clients participate in every round.
Aggregation is plain equal averaging (0.2 each); there is no median clipping,
dropout, omission, sample weighting, or selected-checkpoint rule. Use the exact
`HEAD` configuration from `tools/run_encoder_matrix.py`: SGD, learning rate 20,
batch size 1,024, weight decay 0.000036231884057971014, gradient clipping 10,
one CPU thread and fixed local inverse-frequency class weighting. Optimizer
state resets per existing trainer call. SGD has no momentum. Batch size exceeds
each client's dataset, so every step consumes every local example; client RNG
streams remain independently derived but do not affect full-batch row choice.

Evaluate at 0, 100, 200, 300, 400 and 500 completed steps per client. Report all
54 observations, all nine final results, seed means and sample standard
deviations, all per-domain results, and paired differences from local-step 1.
Do not choose a checkpoint, retune settings, or hide unfavorable results.

Before the first measured cell, `prepare` writes identical machine-readable
`plan.json` files here and in the ignored run folder, with creation time,
baseline commit, exact cell order/configuration, source/data SHA-256 hashes,
saved initialization file/state hashes, runtime and encoder/manifest identities.
This is a local development plan written before the study, not a pushed lock
for a fresh independent evaluation. The runner checks it before each cell.

Measured cells run one at a time in fresh WSL Linux Python processes. Schedule
order rotates by seed: 1/2/4, 2/4/1, 4/1/2. Threads are fixed to one. Other model
inference, test-suite work and video encoding are paused during this interval.
Report total subprocess wall time, in-cell wall time, summed local training and
optimizer-loop time, aggregation time, evaluation time and Linux process peak
RSS. Cold imports, optimizer setup and all five resident feature caches belong
to the process measurement. Encoder inference, feature preparation, physical
client memory and concurrent host/device contention are not measured. This is
one observed time/memory sample per seed, not a hardware benchmark with repeated
timing trials.

The raw head has 15,380 FP32 parameters: 61,520 bytes. The communication estimate
is `2 * 5 * rounds * 61,520` bytes (full head down, delta up each round), or
307,600,000 / 153,800,000 / 76,900,000 bytes across the three schedules. This is
arithmetic, not actual serialized or wire traffic. It excludes encoder/data
transfer, transport metadata, retry overhead and final deployment. These exact
exclusions and the separate measured resource fields are frozen in `plan.json`.

Compare accuracy against estimated payload using strict Pareto dominance:
another schedule must have at least equal accuracy and no greater payload,
with at least one strict improvement. Also report the observed frontier when
process wall time and peak RSS are added as costs; small timing/memory differences
do not establish a reproducible hardware advantage. Report a quality/resource
tradeoff when neither row dominates the other. Three seeds and reused validation
do not establish fresh test accuracy, significance, or general superiority.

Reproduce from repository root in the pinned WSL environment:

```sh
/home/sameer/.venvs/loflwr/bin/python -m unittest discover -s tests -p test_resource_frontier.py -v
/home/sameer/.venvs/loflwr/bin/python tools/run_resource_frontier.py prepare
/home/sameer/.venvs/loflwr/bin/python tools/run_resource_frontier.py run
/home/sameer/.venvs/loflwr/bin/python tools/run_resource_frontier.py report
```

Existing plans and timed cells cannot be overwritten. For an explicitly separate
reproduction, use both `--output runs/resource-frontier-reproduction-N` and
`--public runs/resource-frontier-reproduction-N-public`; preserve the published
v1 files. The new plan has a distinct timestamp/hash. Completed cells may be
resumed without remeasurement; incomplete cells require investigation.
