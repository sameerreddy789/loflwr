---
name: banking77-experiments
description: Build frozen BANKING77 manifests and run or review matched FedLoRA-Edge strategy, dropout, and weak-client omission experiments with honest task metrics.
---

# BANKING77 experiments

Read [evaluation.md](../../research/evaluation.md) for the authoritative twenty intents, client IDs, sample counts, work limits, and scoring rule. Canonical support responses are authored application content, not dataset-provided answers.

Retain original split/row IDs, normalize for duplicate detection, stop on conflicting labels, and enforce disjoint held-out/client query sets within each scenario. Hash the source revision and frozen manifests. Verify available unique counts; never silently oversample to meet the design.

Validation is 200 queries, final test 500, and balanced training 800. In the 400-example imbalance, card-payments client ID 1 owns 360 examples and each other client owns 10 in a 3/3/2/2 allocation. Label this as an extreme few-shot regime.

Keep initialization/A, manifests, scoring/masking, optimizer settings, attempts, random streams, budgets, and dropout schedule matched. Training seeds are 42/43/44; initial heterogeneous optimizer-step limits are `[2,16,8,4,2]`. Derive client randomness independently so omission does not shift surviving clients' streams.

Compare frozen base, sample-weighted shared-A FedAvg, equal averaging, and equal averaging plus clipping. Optional B-coordinate FedProx uses validation-only mu selection. Complete 20 attempts per principal run; skipped attempts remain visible. Final-test access does not tune settings or select unpublished favorable runs.

Report accuracy, macro F1, per-domain/worst-domain accuracy, domain gap, validation task loss, work, participation, actual memory, traffic, and time. Attach seed/run/config/manifest/model identities. Publisher scores, tensor arithmetic, and toy state tests are not project training results.

The weak-client omission study requires matched runs and paired seed differences. Report negligible or unfavorable effects; nonzero deltas alone do not prove useful contribution. Keep the three-laptop experiment separate from the five-client benchmark.
