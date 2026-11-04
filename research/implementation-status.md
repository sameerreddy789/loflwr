# Implementation status

Last updated: **4 November 2026**. This is the prototype handoff status. Detailed published measurements belong in the [results register](results/README.md); ignored run files remain on the prepared laptop.

## Delivered accuracy variant

The selected frozen-MPNet/federated-head model scored **94.0% on independent
confirmation (282/300)**, following **95.5% validation (191/200)**. Model, assets,
scoring sources and all optional margin rules were frozen in commit `7413e6b`
before confirmation. The strictest rule achieved **98.65% on 223 answered
queries**, or **74.33% coverage**; no frozen rule reached 99%. Further tuning
requires a new independently held-out evaluation set. See the
[complete report and locked results](results/encoder-v2/README.md).

| Component | Current evidence |
| --- | --- |
| Actual Flower head training | `encoder-flower-balanced-v2`, selected version 500; all 500 attempts committed with five clients. The head has 15,380 parameters / 61,520 raw FP32 bytes per direction. |
| Unequal budgets and clipping | Matched 500-attempt runs with budgets `[1,4,2,2,1]` reached 95.5% validation with equal averaging and with median clipping. |
| Weak-client contribution | The new matched three-seed study at attempt 500 shows 95.5% with client 4 versus 77.5% without it in every seed; verification falls from 90% to 0%. It owns the only training examples of four intents. |
| Original 90% data imbalance | The new 400-query, three-seed head study finishes at 62.5% sample-weighted, 76.5% equal, and 78.17% ± 0.29 points clipped. The earlier v2 best-checkpoint figures use different settings; preserve their separate identity. |
| Trained-model recovery | `encoder-flower-trained-recovery-v2`: client 1 killed during attempt 2; four survivors committed with weights 0.25; all five rejoined with fresh state in attempt 3. Collection took 10.27/6.83/1.78 seconds under a 60-second deadline. Validation stayed 95.5%. |
| Adapter variant | Shared frozen-A MPNet B-plus-head CPU/GPU invariants passed. Further federated/centralized adapter trials did not exceed 95.5% validation; the federated trial selected its initial checkpoint. |
| Offline query demo | Pinned local MPNet and selected Flower head, published aggregate confirmation, recorded client evidence, and optional frozen clarification rule. CPU inference exactly reproduced the 200 cached validation predictions. |

The WSL suite now passed **246 tests, zero failures and zero skips**. See the
[latest verification](results/demo-extension-v1/verification.json). The seven
dashboard checks are included, not additional tests. These checks do not score
confirmation queries. Only one physical laptop is available.

The [automated preflight](../docs/demo-preflight.md) passed against the confirmed
offline model and the existing matching dashboard. A
[short narrated recording](../presentation/narrated-demo.md) now supplements the
original deck and silent backup. The [nine-run resource study](results/resource-frontier-v1/README.md)
held total optimizer work fixed: four local steps between averages saved 75%
estimated raw payload but reduced validation accuracy by 2 points in every seed.
It is a cached-feature development comparison, not a new test-set result.

The original experiment used a central public feature cache. The new
[local preparation](results/encoder-local-v1/README.md) runs independently encode
each participant's own records in isolated, networkless Linux processes. Both
five- and three-client layouts passed feature parity and produced zero changed
predictions on all 2,960 training/validation rows per layout. Portable role
configuration and a code-only Flower app bundle were also exercised; the
selected head's parameters remained identical during identity migration.

The separate [locked scope study](results/scope-v1/README.md) scored 2,279 new
unsupported BANKING77 test queries. The strictest existing margin falsely
accepted 311; adding the fixed scope detector reduced this to 57. This is a
negative-only result, not a new twenty-intent accuracy score. Scope-only positive
coverage is 190/200 on reused validation. The detector remains separate from
the confirmed demo until fresh supported-query evaluation is available.

The [complete encoder matrix](results/encoder-matrix-v1/README.md) contains
33/33 completed cells and 16,500 committed attempts, with matched seeds and
all selected/final outcomes. Expanded-data methods finish at 95.5% under both
equal and unequal budgets. Scripted unavailability costs 0/0.5/0 final accuracy
points relative to its control across seeds 42/43/44. These are sequential
mathematical federation runs with real optimization, not 33 Flower deployments.

The new [role-isolated TLS rehearsals](results/remaining-work-v1/README.md) pass
three-client `[3,2,3,1,3]` and five-client `[5,4,5,2,5]` contribution sequences.
Each below-quorum attempt keeps the checkpoint, and every accepted update
starts from the expected current head hash. Fresh three-client offline startup
also passed in a network namespace containing only loopback. These are real
same-host Flower process tests with locally prepared features; physical LAN
failure and three distinct laptops remain untested.

The [representation diagnostic](results/encoder-representation/README.md) is a
separate centralized development experiment. Three- and ten-epoch variants
peaked at 96% reused-validation accuracy; the larger ten-epoch model finished
at 94.5%. It has no new independent score or
federated transport result. The demonstrated Flower checkpoint remains unchanged.

## Original SmolLM2 benchmark

The following preserves the separate B-only LoRA implementation and evidence.

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

The earlier original-benchmark WSL2 suite passed **115 tests with no skips** on 3 November 2026; the later 157-test suite above includes encoder checks. The original checks cover interrupted-attempt recovery, stable dashboard run selection, transient progress-file replacement handling, and validation-reference/candidate-recall safeguards. Optional integration tests need their corresponding packages. Unit tests and runtime probes are distinct from language-model experiments.

The seed-42 pilot has completed frozen-base, centralized B-only, sample-weighted,
equal-weighted, equal-plus-clipping and scripted-dropout runs. Final-test accuracy
is respectively 32%, 40.6%, 24.6%, 35.4%, 34.4% and 33.4%. The dropout run committed
all twenty attempts, rejecting the killed client's work on attempts 4/8/12 and
accepting fresh rejoining on 5/9/13. The completed omission run scores 34.8%
test accuracy with unchanged 32% verification-domain accuracy, so weak-client
benefit is not demonstrated in this seed. All seven pilot configurations are
complete; the full multi-seed program remains incomplete. See the [results register](results/README.md)
for work, identities and limitations.

A separate [accuracy feasibility study](accuracy-99-feasibility.md) reaches
95.5% top-1 validation accuracy using centralized frozen MPNet retrieval with
expanded training data. Its 99% top-2 candidate recall is not classification
accuracy. These reference scores are centralized validation results; the
separate final federated model and confirmation result are reported above.

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
| Frozen comparative study | All seven original LoRA pilot configurations completed. The encoder's 33-cell three-seed matrix is now complete; the original LoRA full multi-seed program remains distinct and incomplete. |
| Useful weak-client participation | All three matched encoder seeds show an 18-point validation benefit from including the one-step verification client. Its exclusive four-label ownership limits generalization; the original LoRA pair was null/unfavorable. |
| Actual interruption and recovery | Real Ray rehearsal passed process kill, below-quorum B retention, and all-five-client fresh rejoining. [TLS rehearsal](results/tls-node-recovery.json) also passed full SuperNode process-tree termination, two-survivor commit, and fresh rejoin. These are same-host tests. The 60-second Ray deadline missed cold/queued work, so the frozen comparative pilot declares 300 seconds. |
| Fresh offline startup | Fresh CPU inference and fresh three-node local TLS deployment passed inside a loopback-only network namespace. The deployment committed with two accepted updates; one client missed its deadline. See [offline deployment evidence](results/offline-deployment.json). Physical-laptop deployment remains untested. |
| Three physical laptops | Measured CPU budgets and whole-process memory, LAN/TLS/known-node setup, real device loss/rejoin, and a fresh start with internet disconnected. Logical clients on one laptop do not satisfy this gate. |
| Morning hands-on demo | Inspect the real run, score fresh queries, exercise replay, and perform the unplugged-internet restart checklist in [the morning guide](../docs/morning-demo.md). |

The same-host deployment launcher and transport probes are implemented. Their use of TLS or multiple processes does not prove the three-laptop target. The live layout is three clients with quorum two; the benchmark layout is five with quorum three. The [manual deployment guide](../docs/deployment.md) records the required LAN/TLS/configuration preparation and current limitations; that recipe has not been rehearsed across physical laptops.

See [the tested environments](../requirements/README.md), [study instructions](../docs/study-runner.md), [evaluation contract](evaluation.md), and [roadmap](roadmap.md) for reproducibility and acceptance criteria. Keep this status aligned with the results register as further evidence completes.
