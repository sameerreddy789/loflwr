# Prepare client features with less padding

**Experimental:** the [full compatibility check](../research/results/encoder-bucketing-v1/README.md)
preserved all 2,960 top-1 predictions and passed one-round head optimization,
but one reused-validation answer/clarify decision changed relative to the stored
reference caches. The zero-decision-change gate failed. Use new development
identities; do not replace the confirmed calibration caches or inference path.

This optional recipe targets the cost of **batched, local frozen-MPNet feature
preparation**. It does not change the confirmed query predictor, shorten its
384-token limit, quantize the model, or change federation budgets. Read the
[engineering guidance](../research/ml-engineering-guidelines.md) and
[paper review](../research/efficiency-paper-review.md) for the rationale and
measurement boundaries.

The implementation tokenizes each query once without special tokens, lets the
tokenizer add/truncate its boundary tokens, and sorts by exact truncated token
length. Batches contain similar-length records, so fewer padding positions pass
through the encoder. It restores every vector to its original record position
before attaching labels and source IDs. The preparation contract fixes CPU
FP32, one Torch thread, batch size eight, masked mean pooling and L2 normalization.

Rebatching can change floating-point rounding. The new recipe therefore has
distinct plan, registration and dataset hashes. A parity result on development
data is not a new independent accuracy measurement. Keep the original caches
and confirmed checkpoint intact.

## Prepare a single participant

Use the [tested CPU/WSL environment](../requirements/README.md), already cached
model files and the participant's own approved record bundle. This example
prepares client 4, the account-verification participant, in new ignored paths:

```powershell
python tools/prepare_encoder_bucketed.py plan --source-dataset assets/encoder-bank20-v1 --layout five --output runs/bucketed-plan.json
python tools/prepare_encoder_bucketed.py prepare --plan runs/bucketed-plan.json --participant 4 --records assets/encoder-bank20-v1/clients/client_4.json --asset-dir assets/reference-mpnet --output runs/bucketed-client-4
```

For a physical client, transfer the shared plan, its own approved records, code,
dependencies and model assets using the [laptop workflow](encoder-laptop-rehearsal.md).
It does not need other clients' records to run `prepare`. Ownership, record
hashes and labels are validated before loading the encoder or writing output.
The source hash is part of the plan: an unrelated record bundle is rejected.

The output contains the private local feature archive, source-record envelope,
`registration.json` and `preparation.json`. Registration contains metadata and
hashes; it does not include queries or feature arrays. The preparation report
records actual elapsed time and sampled process RSS, plus padded-token and
attention-cell work proxies. Keep all these generated files under ignored
`runs/` or `assets/`. The command refuses existing output paths.

## Assemble metadata for a new run

Repeat preparation independently for clients 0–4 and the server's separate
validation role. The server role receives only its approved validation bundle.
The three-role layout is also represented explicitly in the plan; do not mix
layouts or old registrations with this recipe.

```powershell
python tools/prepare_encoder_bucketed.py assemble --plan runs/bucketed-plan.json --registrations runs/bucketed-client-0/registration.json runs/bucketed-client-1/registration.json runs/bucketed-client-2/registration.json runs/bucketed-client-3/registration.json runs/bucketed-client-4/registration.json runs/bucketed-server/registration.json --output runs/bucketed-manifest.json
```

Assembly reads registration metadata only. It does **not** copy or upload client
features. Place the common manifest and each role's own archive at the relative
paths expected by the existing local loaders, as described in the laptop guide.
Use a new dataset/run identity when configuring training. The new manifest
binds the new recipe while retaining the pinned encoder and original record
identities; it must never be substituted into an old locked run.

## Reproduce the controlled comparison

The [benchmark tool](../tools/benchmark_encoder_bucketing.py) fixes two complete
cohorts in advance: 200 reused validation queries and 408 client-4 training
queries. It runs three paired repetitions in fresh processes with alternating
backend order. It keeps individual timings, model-loading time, process RSS,
feature differences, head predictions and fixed-margin decisions.

```sh
python tools/benchmark_encoder_bucketing.py prepare --output runs/bucketing-repro --public research/results/bucketing-repro
python tools/benchmark_encoder_bucketing.py run --output runs/bucketing-repro
```

Use fresh paths and keep source/data/runtime settings unchanged between these
commands. Do not run other benchmarks concurrently. The script freezes its
development plan locally before measuring; this is not a new independent-test
protocol. Keep slower results and failed parity checks. Interrupted runs retain
their logs and start markers; inspect failures instead of overwriting them.

Encoding timing includes tokenization, sorting, padding, pooling and restoration,
with the model already resident. Cold verification/loading is reported separately.
The benchmark also counts actual forward tensor shapes and checks those counts
against the arithmetic work counters. Neither count is measured energy or FLOPs.
Sampled RSS includes the runtime and allocator; pair order can affect it.

For a full compatibility check, the
[integration runner](../tools/check_bucketed_integration.py) creates a fresh plan,
prepares all five clients and server validation in serial processes, and checks
all 2,960 original training/validation records. It also exercises actual local
head optimization and equal aggregation for one round with both feature sets:

```sh
python tools/check_bucketed_integration.py --output runs/bucketing-integration-repro --public research/results/bucketing-integration-repro.json
```

Run this after timing, not concurrently. It deliberately collects the public
benchmark outputs into a same-host auditor dataset. That is not the physical
deployment or a new private-data isolation test. It reports unchanged source IDs,
labels, predictions and fixed-margin decisions, or retains a failed compatibility
result. One round checks integration; it does not establish convergence. Actual
model/record/state files remain under the ignored output directory.

This optimization leaves the **61,520 raw FP32 bytes per head update** unchanged.
It does not claim faster single-query inference or fewer training rounds. The
confirmed **94% (282/300)** result remains the existing independent evidence.
