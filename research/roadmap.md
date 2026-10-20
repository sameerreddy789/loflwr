# Implementation and demonstration roadmap

This is the approved delivery order for roughly one week or more of implementation. Dates below are work stages rather than claims that the stages have already been completed. See [implementation-status.md](implementation-status.md).

## Research delivery

Publish the root README, six-part dossier, project skills, initial state scaffolding, and meaningful dependency-free checks. Link primary sources and record their review depth. Verify the GitHub commit and repository navigation. Exclude model weights, datasets, credentials, and local caches.

## Days 1–2: one-client learning and reproducibility

Prepare isolated Python 3.11+ CPU and CUDA environments. Flower 1.40.0 is the selected starting framework version; package availability is not proof of compatibility. Freeze complete environments after smoke tests, not before discovering which stack actually works.

Download the selected model/tokenizer revision into `assets/`, build the centrally initialized A, and save its identity contract. Freeze base and A; train B only. Build and hash deterministic BANKING77 manifests using the exact counts in [evaluation.md](evaluation.md).

Complete a real CPU and GPU smoke test. Check trainable names/count, frozen hashes, completion/EOS masking, serialization, loss computation, peak RSS/VRAM, and optimizer-step timing. Reset optimizer state between attempts.

**Gate:** actual B learning works within the laptop memory budget. If memory or task quality fails, first adjust sequence length/work limits or reconsider the validated small-model choice using validation evidence; document changes before freezing principal experiments.

## Days 3–4: five-client federation and failures

Implement Flower ClientApp and ServerApp around the shared training code. Add versioned request/reply contracts, sample-weighted/equal/clipped strategies, explicit deadlines, quorums, validation, and atomic checkpoints.

Use WSL2 for Ray simulation and start with one GPU training worker. Account for queued execution in the simulation deadline. Exercise application-level dropout, actual process interruption, stale rejoin replies, duplicate replies, invalid tensor payloads, and below-quorum attempts.

**Gate:** all clients start from the same committed B; only compatible successful replies count; enough survivors commit within the bounded attempt; a rejoining node receives current state. A toy aggregation test alone does not pass this gate.

## Days 5–6: measured comparisons and deployment

Run the principal 20-attempt experiments with seeds 42/43/44 and matched manifests, budgets, and schedules. Add the equal-weight ablation and weak-client omission study. Tune optional B-coordinate FedProx only on validation. Publish unfavorable as well as favorable results.

Prepare three separate laptops using Flower Deployment Runtime. The host provides SuperLink, evaluation, a client, and a dashboard. The other two laptops contribute CPU clients. Use a dedicated hotspot/LAN, TLS, authenticated known nodes, locally cached assets, and a tested offline wheel/environment bundle.

**Gate:** real client training and communication work over the local network. Measure server/evaluation memory separately from client memory. Record actual transport bytes and timing rather than replacing them with raw tensor estimates.

## Day 7: offline rehearsal and judging

Disconnect internet and perform a fresh start. Demonstrate a real client disappearing mid-training, survivors meeting quorum, and the returning client synchronizing fresh state. Rehearse recovery if the quorum is missed.

Build one small dashboard around recorded run events and real inference. Show client status, completed steps, attempts/versions, round duration, bytes, coefficients, validation curves, domain scores, and a banking-query interface. Clearly label any replay; never display invented live metrics.

**Gate:** the three-minute demonstration uses completed comparison results and a bounded live attempt. Keep a clearly labeled replay of a real recorded run for demonstration continuity, while reporting any live failure honestly.

## Three-minute presentation

| Time | Demonstration |
| --- | --- |
| 0:00–0:30 | Explain data locality, weak compute, unequal participation, and the bounded banking task |
| 0:30–1:15 | Show completed baseline/main comparisons, domain scores, work, and measured traffic |
| 1:15–2:30 | Show a live dropout, survivor coefficients/quorum, checkpoint commit or skip, and fresh rejoin |
| 2:30–3:00 | Submit a banking query, show predicted intent and authored response, and state evidence limits |

The live deadline is initially 60 seconds. Calibrate work before rehearsal; the presentation cannot depend on completing the full 20-attempt benchmark live.

## Extensions after the core gates

| Extension | Why defer it |
| --- | --- |
| Larger models | Additional CPU time, memory, and baseline cost before proving the small model works |
| Heterogeneous adapter ranks | Requires a different parameter-alignment/aggregation contract |
| Buffered asynchronous aggregation | Adds staleness and availability/influence accounting |
| Secure aggregation | Changes which per-client updates and norms the server can inspect |
| Differential privacy | Requires a threat model, clipping/noise mechanism, accountant, and quality evaluation |
| Malicious-client defenses | Requires a distinct attack model and adversarial experiments |

The core deliverable remains a reproducible small-model system with measured evidence, not the number of additional algorithms integrated.
