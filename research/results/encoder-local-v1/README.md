# Client-local encoder preparation

Both layouts completed on 3 November 2026 using separate Linux mount, PID and
network namespaces on the same laptop. Each role loaded the pinned frozen
MPNet encoder and only its own approved query bundles. This closes the local
preparation gap in the original head experiment, whose public-data features
were generated centrally before partitioning.

| Layout / run | Client query counts | Separate server validation | Wall time |
| --- | --- | --- | --- |
| `encoder-local-five-v1` | 513 / 615 / 608 / 616 / 408 | 200 | 340.23 s |
| `encoder-local-three-v1` | 1,128 / 1,224 / 408 | 200 | 411.85 s |

The three-client layout combines original domains 0+1, 2+3 and 4. It is a
deployment layout, not the five-client comparison benchmark. Both layouts
cover the same 2,760 training examples and 200 development examples.

All 2,960 recomputed feature vectors in each layout matched the reference within
the predeclared absolute tolerance of `1e-5`. The largest absolute difference
was `1.043081283569336e-07`; the unchanged confirmed head produced **zero changed
predictions in either layout**. This is numerical parity on training and
validation data, not another independent accuracy evaluation. Neither old
confirmation queries nor the CLINC evaluation was scored.

The full original repository was hidden inside each participant namespace.
Only its code, model, own input and own output were mounted at task paths.
Checks verified that reference data paths were inaccessible and an external
network connection failed. Each role returned a hash-checked metadata
registration; the shared manifest was assembled from these registrations.
The server received its separate public validation set, not client features.

## CPU cost

Preparation uses FP32, one CPU thread, batch 8, masked mean pooling followed by
L2 normalization, and a 384-token limit. No example was truncated. Whole-process
sampled RSS includes the encoder and runtime. Sample interval: 20 ms.

| Layout | Role | Queries | Encode/load/verification time | Peak RSS |
| --- | --- | --- | --- | --- |
| Five | Client 0 | 513 | 32.94 s | 1,101.46 MiB |
| Five | Client 1 | 615 | 61.66 s | 1,134.23 MiB |
| Five | Client 2 | 608 | 67.70 s | 1,130.04 MiB |
| Five | Client 3 | 616 | 59.30 s | 1,116.86 MiB |
| Five | Client 4 | 408 | 28.11 s | 1,131.46 MiB |
| Five | Server | 200 | 38.75 s | 1,140.37 MiB |
| Three | Client 0 | 1,128 | 109.24 s | 1,137.24 MiB |
| Three | Client 1 | 1,224 | 180.87 s | 1,133.79 MiB |
| Three | Client 2 | 408 | 32.68 s | 1,131.99 MiB |
| Three | Server | 200 | 32.72 s | 1,139.82 MiB |

These were sequential role preparations with other experiments running on the
laptop. Times are observations, not controlled comparisons between layouts or
physical devices. Stored features remain local; their storage is additional
to the encoder assets and is not the 61,520-byte training update payload.

The separate Windows CPU [demo timing profile](demo-profile.json),
`encoder-demo-profile-v1`, measured 30 batch-one queries over five authored
timing probes: **152 ms median, 229 ms p95**, and 735.03 MiB peak process RSS.
Pinned asset verification and model loading took 15.76 s; the first query took
452 ms. It used FP32 and two CPU threads. This is a fresh Python process with
possible warm operating-system file caches and concurrent experiment activity.
It excludes browser memory and import time from the loading timer. There are no
expected labels in these probes, so this profile supplies no accuracy evidence.

```powershell
python tools/profile_encoder_demo.py --output runs/demo-profile-new.json
```

## Reproduce and audit

In the prepared Ubuntu WSL environment, from the repository root:

```sh
python tools/check_encoder_local_preparation.py run --layout five --output runs/local-five-new
python tools/check_encoder_local_preparation.py run --layout three --output runs/local-three-new
```

See [five-role export](five.json) and [three-role export](three.json) for the
run, source, model, plan and manifest hashes, per-role isolation checks, feature
parity and measured memory. Raw query bundles and features stay in ignored
`runs/`; the exports contain metadata only.

The parent process was a public-benchmark provisioner and auditor that could
access the source data before isolation and compare results afterward. These
checks establish the exercised process boundary; they do not prove production
data ownership, independent physical devices, secure aggregation, differential
privacy or resistance to malicious participants. The physical test remains
pending in the [laptop rehearsal guide](../../../docs/encoder-laptop-rehearsal.md).
