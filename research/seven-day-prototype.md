# Solo seven-day working-prototype plan

Planning update: **2 November 2026**. The user confirmed one builder and a working-prototype/demo checkpoint in one week, rather than an abstract deadline. Day numbers are relative work days; the exact checkpoint date/time has not been supplied. No training or federation result exists yet.

This stages the work in the existing [full evaluation plan](evaluation.md); it does not claim that a smaller pilot completes the full study. The selected model, twenty intents, frozen A, B-only training, and equal/median-clipped main strategy stay unchanged. The uploaded abstract's centralized comparison is included explicitly; other design changes are documented in [abstract-alignment.md](abstract-alignment.md).

## What must work at the checkpoint

The target is a complete path from fixed data to real learning and recorded evidence:

1. Five Flower clients train on distinct BANKING77 partitions with actual unequal step budgets, using the same frozen base/A and versioned global B.
2. Sample-weighted FedAvg, plain equal averaging, and equal averaging with median clipping run as separate matched experiments. A centralized B-only reference and frozen-base evaluation provide context.
3. Held-out quality, completed work, memory, communication, coefficients, and time are recorded with run/configuration identities. Improvement must be observed before claiming the abstract's learning requirement is met.
4. A real training process disappears. Sufficient valid survivors commit within the bounded collection window; insufficient survivors leave the checkpoint unchanged; a returning client starts from current B.
5. A small interface shows real run events and a twenty-intent query result with its authored response. Any replay is labeled and comes from an actual run.

Five simulated clients are the chosen benchmark. The problem statement permits three to five, but silently changing to three would change this project's domain mapping and quorum. If five is infeasible, record a new configuration and label the reduced scope before running it. Five processes on one computer are not five physical devices.

The core week excludes FedProx tuning, rank heterogeneity, quantization experiments, new optimizers, secure aggregation, differential privacy, broad model comparisons, and an elaborate frontend. Three physical laptops remain the deployment target after access and timing gates; they are not a prerequisite for the simulation minimum.

## Start from what actually exists

| Available observation | What it supports | Still unverified |
| --- | --- | --- |
| Python 3.11.15 Windows environment and initial package | Existing tensor-state checks can run | A complete ML/Flower dependency stack |
| Host has about 16 GB RAM and RTX 3050 Laptop GPU with 6 GB VRAM | A plausible small-model development machine | Actual training/evaluation peak memory and throughput |
| Ubuntu WSL2 starts and sees the GPU; about 7.4 GiB assigned RAM and 2 GiB swap | A candidate Flower simulation host | Reliable five-client scheduling without memory pressure |
| WSL Python is 3.12.3; `uv` is available | An isolated Python 3.11 environment can be prepared | A tested CPU/CUDA environment lock |
| State/schema/hash helpers and 16 dependency-free tests | Initial serialization/tensor invariants | Dataset, protocol, model, Flower, learning, and deployment behavior |

These are environment observations from this research session, not CPU/GPU training measurements. Keep a separate WSL environment; do not reuse a Windows virtual environment from Linux. The selected initial framework baseline is Flower 1.40.0. Its [versioned Grid reference](https://flower.ai/docs/framework/1.40/en/ref-api/flwr.serverapp.Grid.html) documents separate TTL and collection timeout. Current unversioned documentation displays 1.42.x, so test against 1.40 and freeze the working stack rather than copying commands across versions.

## Daily deliverables and stop/go gates

| Day | Build and measure | Evidence required to move on | If blocked |
| --- | --- | --- | --- |
| 1 | Prepare one isolated training environment, cache model/tokenizer, create common A, implement tokenization and a tiny real CPU learning diagnostic | B-only optimizer/gradients; base/A unchanged; real EOS and completion labels preserved; memory and step timing | Fix loading, masks, gradients, or memory before adding networking |
| 2 | Freeze data manifests; validate twenty-way scoring; short centralized diagnostic; GPU smoke; freeze tested environments | Disjoint counts/hashes, stable scoring under padding/batching, actual local learning, validation cost, dependency identities | Diagnose the local task; no claim that federation can repair a broken local pipeline |
| 3 | Implement the three B-delta strategies, request/reply contract, checkpoint commit/skip, and Flower ClientApp/ServerApp integration | Shared-state mathematics; invalid/stale/duplicate rejection; all accepted clients start from the same committed B | Keep UI/deployment work deferred; finish one real end-to-end attempt |
| 4 | Run five real model-training clients with unequal budgets; add timeout, quorum and worker cleanup; rehearse process kill/rejoin | Bounded attempt, checkpoint identity below quorum, current B on return, no abandoned worker holding the GPU | Repair failure handling; record operational limits rather than animate success |
| 5 | Run the frozen pilot comparison and centralized reference; collect validation and final-test results using the declared rule | Full records for all methods, including unfavorable results; observed work/traffic/time; no final-test tuning | Stop adding configurations; publish completed evidence and identify unmet learning gates |
| 6 | Add small event-driven dashboard/query interface; run omission/dropout quality comparisons if affordable; attempt physical deployment only after core gates | Real inference and metrics, optional comparisons separately labeled; real device inventory if deployed | Use completed simulation evidence; mark physical deployment incomplete |
| 7 | Fresh offline start and repeated rehearsal; freeze code/configuration; record a real backup demo | Repeatable start, labeled replay, bounded live failure demo, clear result/limitation statement | Use the last verified build and disclose the live failure; do not make last-minute algorithm changes |

Each day produces a working artifact, not merely a checklist tick. Debugging has priority over additional experiments. If local learning is still broken at the end of day 2, the remaining schedule is at risk; a polished dashboard cannot recover the missing scientific evidence.

## Exact pilot and workload forecast

Use the existing **90% imbalance plus heterogeneous work** condition first. It makes sample and equal weighting differ and includes genuinely unequal work. Keep client sizes `[10, 360, 10, 10, 10]`, initial optimizer-step budgets `[2, 16, 8, 4, 2]`, seed 42, the frozen twenty-intent manifests, and all training/scoring settings from [evaluation.md](evaluation.md). This is a combined stress condition, not an isolated test of data imbalance.

After calibration and before principal evaluation, freeze a common horizon. The target is twenty attempts per federated method. If measured cost makes that infeasible, declare a shorter pilot horizon, use it for every comparator, and retain the full twenty-attempt requirement as outstanding. Do not shorten runs after looking at favorable test scores. Diagnostic prefixes may be reused only when their complete initialization, data, randomness, configuration and continuation state match.

| Run | Initial checkpoint scope | Purpose |
| --- | --- | --- |
| Frozen base | One shared evaluation | Initial quality under the identical scorer |
| Centralized B-only reference | Same processed-work blocks and initialization as the no-dropout condition | Honor the abstract's reference; check that the adaptation task can learn |
| Sample-weighted FedAvg | Seed 42, up to 20 attempts | Naive coefficient policy |
| Plain equal FedAvg | Same seed, manifests, budgets and horizon | Isolate the coefficient change |
| Equal plus median clipping | Identical contract | Isolate clipping relative to equal averaging |
| Main with client-1 dropout | Optional same-horizon run, reusing the matching main control | Measure learning cost of scripted failure, separate from protocol correctness |
| Main without client 4 | Optional same-horizon run, reusing the matching main control | Measure the total utility effect of that participant |

The core federated trio contains at most 60 scheduled attempts; the two optional runs add at most 40. The centralized reference is additional work, not included in those attempt counts. Its per-client minibatch replay and optimizer reset rule are specified in [abstract-alignment.md](abstract-alignment.md). Match actual processed steps/tokens and disclose any misses. Fit any TF-IDF contextual classifier only on the same training union, selecting settings on validation.

The original full matrix remains 480 principal attempts **before** ablations and omission. Running a 200-query, twenty-candidate validation after every one of 480 commits would require 1,920,000 candidate-completion scores. This is arithmetic, not a runtime prediction. It is not credible to assign the full matrix to two days without measurement.

Measure a full validation pass early, not just a training step. Forecast from observed model load, local work, serialization, queue, reply collection, aggregation, checkpointing and evaluation time. With one GPU worker, queued client times largely add; do not divide by five. On concurrent physical clients, collection follows response/deadline behavior instead. Reserve at least one work day for integration failures and rehearsal when choosing the experiment horizon.

Evaluate the final committed checkpoint after the declared attempt horizon. Retain skipped attempts in counters and time. Freeze the scorer and selection rule before final-test evaluation. If results motivate further tuning, use validation; report that an earlier test set was already inspected rather than presenting it later as untouched confirmation.

## Acceptance is a set of claims, not a single green badge

| Claim | Passing evidence | Honest report if it fails or is deferred |
| --- | --- | --- |
| Adapter-only learning works | Real CPU/GPU B changes, fixed base/A, correct masks and serialization | Local training remains incomplete |
| The model learns under heterogeneity | Actual held-out loss and intent-quality trajectories, compared with frozen base | Operational prototype; learning criterion not met or inconclusive |
| The coefficient rule is implemented | Exact survivor coefficients, clip threshold/factors, finite state, valid quorum | Aggregation/protocol defect; fix before comparative claims |
| The method improves utility | Matched overall and domain scores, including a plain-equal control | Mixed/negative/inconclusive result; no superiority claim |
| A weak participant is useful | Positive, meaningful paired omission effect, with global/domain tradeoffs | Inclusion works mechanically; utility benefit not established |
| Dropout is bounded and recoverable | Actual interrupted process, bounded collection, unchanged below-quorum state, fresh return | Failure handling still unresolved |
| Physical deployment works offline | Fresh isolated start on the stated three machines, cached assets, measured network run | Simulation-only evidence; no three-laptop claim |

Predeclare overall macro F1 as the pilot's primary task metric, and report each domain, worst-domain accuracy, domain gap, completion-only validation loss, participation, and resource costs alongside it. The supplied challenge gives no universal accuracy floor. Do not invent an organizer target. Positive held-out improvement is required for the abstract's learning claim; one seed and one partition only support preliminary observations. Statistical confirmation and practical effect thresholds belong in the preregistered later study.

The optional omission run changes data availability, membership, coefficients, and median clipping. Its conclusion is about this participant in this protocol; it does not isolate weak hardware as a cause. If omission is deferred, keep the benefit claim explicitly unproven.

## Failure rehearsal and offline packaging

Use five logical clients with quorum three. A single killed client ordinarily leaves four possible contributors; also test enough missing clients to leave only two valid replies. Count validated responses, not registered nodes. Set TTL and collection timeout, reject old/duplicate replies, and verify timed-out workers release scarce compute. Protocol tests do not substitute for a real model-training process interruption.

Calibrate the live workload toward a 60-second collection deadline. The single-worker simulation deadline must include queued execution; it may need a different configured value from the physical live run. Collection is only part of end-to-end duration: show evaluation and checkpoint overhead too. Fresh rejoining begins from the latest committed B, never a partially trained stale state.

For physical deployment, first verify access to three laptops and compatible CPU environments. Host: SuperLink/evaluation plus lifecycle/payments client; laptop 1: transfers/cash; laptop 2: verification. Use quorum two and separate run IDs. TLS, known-node authentication, private keys under ignored `certificates/`, and offline model/dependency caches remain required. A same-host process demonstration is useful but must be labeled accordingly.

Save weights/datasets in ignored `assets/`, run outputs in ignored `runs/`, and publish only reviewed result summaries/configurations with reproducible identities. A fresh offline start must neither fetch dependencies nor resolve uncached model/tokenizer files. Record exact launch commands only after they are tested with the locked runtime.

## Three-minute evidence sequence

| Time | Show |
| --- | --- |
| 0:00–0:25 | Five clients, their data/work budgets, and the exact intent-routing task |
| 0:25–1:10 | Completed matched results, domain tradeoffs, traffic and actual work; label seed-42 results preliminary |
| 1:10–2:25 | Interrupt a real process, show accepted survivors and commit/skip; show fresh rejoin if it completes within the calibrated sequence |
| 2:25–3:00 | Actual intent query, authored response, and the strongest measured conclusion plus key limitation |

Do not attempt an entire training experiment live. If recovery does not fit the slot, show its real recorded trace as replay and keep the live portion bounded. The backup recording must come from the same identified build/run. No fabricated metrics or animation should be labeled live.

## Work after the checkpoint

Complete the retained twenty-attempt, three-seed study and weak-client/dropout quality comparisons as measured runtime permits. Address the data/work confounds with explicitly designed controls before causal claims. Confirm the physical three-laptop target and expand beyond the single partition before generalizing. Only then consider new algorithms or larger models.

The next implementation action is **one real CPU B-only learning/scoring smoke test**, followed by the manifest and compatibility gates. The current repository's [status record](implementation-status.md) distinguishes this plan from completed functionality.
