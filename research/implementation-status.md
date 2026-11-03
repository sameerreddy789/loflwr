# Implementation status

Last updated: **3 November 2026**. This is the prototype handoff status. Detailed published measurements belong in the [results register](results/README.md); ignored run files remain on the prepared laptop.

## Implemented and exercised

| Component | Current evidence |
| --- | --- |
| Deterministic BANKING77 preparation | Pinned source, duplicate/conflict checks, disjoint query sets, exact allocations, and manifest verification. [Committed identity record](reproducibility/README.md) contains hashes/counts, not queries. |
| Model/common-A preparation | Cached pinned SmolLM2 model/tokenizer, saved common-A tensors, initial B, and adapter/model/scoring identity checks. Realized B count is 184,320. |
| B-only local training | Real Windows CPU and WSL2 CUDA optimization smokes passed: B changed; frozen base and A hashes did not. Both used FP32. Completion/padding/EOS and scoring checks passed. |
| Scorer optimization | The gathered completion-logit implementation reproduced all 200 validation predictions and true-completion scores from the initial Flower checkpoint. Eight same-logit probes also matched exactly. This is scorer parity, not an additional training comparison. |
| Persistent model worker | Real isolated-process calibration passed cold/warm and interleaved-client bitwise delta equivalence, process termination with no partial delta, equivalent replacement-worker output, and bounded deadline cleanup. |
| Round state and aggregation | Equal, equal-plus-median-clipping, and manifest-size weighting; identity/schema/finite validation, deadline/quorum logic, stale/duplicate rejection, atomic checkpoints, and resume contracts have unit coverage. |
| Flower ClientApp/ServerApp | A real five-client, one-worker GPU run committed one round with all five clients accepted. Each completed two optimizer steps on balanced manifests. Full 200-query validation was recorded. |
| Study launcher | Pilot/full plans, immutable configuration/asset/source identities, sequential execution, bounded process lifetime, resume, and explicit incomplete states are implemented and unit tested. A shorter horizon is calibration and cannot access final test. |
| Dashboard and reports | Saved events, work/traffic, coefficients/clipping, quality, replay, evidence downloads, and checkpoint-backed CPU queries are implemented. The Windows launcher uses offline model settings and binds localhost. Reader and Streamlit fixture checks exist. |

The full WSL2 environment passed **101 tests with no skips** on 3 November 2026. This includes interrupted-attempt recovery, stable dashboard run selection, and transient progress-file replacement handling. Optional integration tests need their corresponding packages. Unit tests and runtime probes are distinct from language-model experiments.

## Traceable local checks

| Local artifact | Run identity and configuration | What it establishes |
| --- | --- | --- |
| `runs/smoke-cpu.json` | `smoke-20261102T183138Z`; Windows CPU, FP32, seed 42, two steps on the first four records of the imbalanced client-0 manifest | Actual optimization, frozen base/A, changing B, and completion-mask/scoring invariants |
| `runs/smoke-gpu.json` | `smoke-20261102T183617Z`; WSL2, RTX 3050 6 GB, CUDA FP32, same smoke setup | The same invariants on the prepared GPU environment |
| `runs/flower-gpu-smoke-20261102/` | `flower-gpu-smoke-20261102`; seed 42, balanced five-client manifests, budgets `[2,2,2,2,2]`, `equal_clipped`, quorum 3, 600-second collection deadline, one attempt | Real Flower requests/replies, ten total completed optimizer steps, version 1 commit, equal 0.2 coefficients, and complete validation evaluation |
| `runs/scorer-gather-gpu-check-20261102/result.json` | `scorer-gather-gpu-check-20261102`; CUDA FP32, batch size 20, initial Flower checkpoint, all 200 validation queries | Exact prediction/true-completion-score parity after the scorer memory change, with separate recorded timing and allocator measurements |
| `runs/scorer-gather-gpu-check-20261102/training-parity.json` | Original Flower client 0 compared with the repeated warm-worker job; same incoming B, client random stream, records, and training configuration | Bitwise-equal B delta and equal work, task loss, and recorded gradient norms for this observed job |
| `runs/persistent-worker-calibration-20261102/results.json` | `persistent-worker-calibration-20261102`; isolated CUDA subprocesses, balanced clients 0/1, seed 42, two-step equivalence probes | Cold/warm/interleaved/replacement equivalence; an eight-step request interrupted after one completed step returned no delta; an expired request returned no delta and cleaned up within the probe bound |

The local smoke reports include full configuration and model/adapter/manifest identities. The CPU/GPU smoke manifest hash is `a8a117397a9dff1b39b285d423de8bfeab8d877af2a09d9601456fcdd4fbae30`. The five-client run's `config.json`, `runtime.json`, `events.jsonl`, `round_results.json`, and `evaluations.jsonl` preserve its contracts and measured results. Raw payload sizes and serialized Flower-message sizes are separate measurements; they exclude network framing and retransmissions.

These are execution checks. A small smoke loss change or a single post-round accuracy is insufficient evidence of improvement over the frozen base, convergence, fairness, or useful weak-client participation.

## Remaining acceptance gates

| Gate | Evidence still required |
| --- | --- |
| Frozen comparative study | Completed declared runs with frozen-base/centralized references, sample/equal/clipped comparisons, exact actual work, and final-test access only after the full horizon. The full multi-seed program remains separate from a seed-42 pilot. |
| Useful weak-client participation | Matched omission runs and paired seed differences, including negligible or unfavorable effects. Nonzero updates alone do not prove utility. |
| Actual interruption and recovery | Real Ray rehearsal passed process kill, below-quorum B retention, and all-five-client fresh rejoining. [TLS rehearsal](results/tls-node-recovery.json) also passed full SuperNode process-tree termination, two-survivor commit, and fresh rejoin. These are same-host tests. The 60-second Ray deadline missed cold/queued work, so the frozen comparative pilot declares 300 seconds. |
| Fresh offline startup | Fresh CPU inference and fresh three-node local TLS deployment passed inside a loopback-only network namespace. The deployment committed with two accepted updates; one client missed its deadline. See [offline deployment evidence](results/offline-deployment.json). Physical-laptop deployment remains untested. |
| Three physical laptops | Measured CPU budgets and whole-process memory, LAN/TLS/known-node setup, real device loss/rejoin, and a fresh start with internet disconnected. Logical clients on one laptop do not satisfy this gate. |
| Morning hands-on demo | Inspect the real run, score fresh queries, exercise replay, and perform the unplugged-internet restart checklist in [the morning guide](../docs/morning-demo.md). |

The same-host deployment launcher and transport probes are implemented. Their use of TLS or multiple processes does not prove the three-laptop target. The live layout is three clients with quorum two; the benchmark layout is five with quorum three. The [manual deployment guide](../docs/deployment.md) records the required LAN/TLS/configuration preparation and current limitations; that recipe has not been rehearsed across physical laptops.

See [the tested environments](../requirements/README.md), [study instructions](../docs/study-runner.md), [evaluation contract](evaluation.md), and [roadmap](roadmap.md) for reproducibility and acceptance criteria. Keep this status aligned with the results register as further evidence completes.
