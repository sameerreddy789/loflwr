---
name: federated-rounds
description: Implement or review FedLoRA-Edge B-delta aggregation, reply identities, bounded collection, valid-update quorums, checkpoint commits, and dropout recovery.
---

# Federated rounds

Use [system-design.md](../../research/system-design.md) as the protocol and mathematics reference. All clients share frozen A and scaling, and replies are relative to one immutable incoming B snapshot.

Validate run/attempt/version, base-B hash, common-A/model/adapter identity, tensor names/shapes/dtypes, completion status, and finite values before counting a reply. Deduplicate by authenticated client identity. Reject late or stale work; a returning client synchronizes current B before training again.

Compute delta norms across all B tensors together. The main strategy clips to the median contributor norm and averages with equal `1/m` coefficients. Use deterministic even-count median semantics and record zero thresholds/deltas. Do not clip tensors independently or silently introduce a positive threshold.

Log declared sample size separately from actual completed work. The sample-weighted baseline uses manifest size; FedProx is a local penalty rather than an influence cap. Equal coefficients do not imply equal domain accuracy or remove participation bias. Two valid live survivors each receive 50%.

Set both Flower message TTL and response-collection timeout. Quorum is three valid updates for five-client simulation and two for three-laptop deployment. Below quorum, retain the committed state and log a skipped attempt. Count attempts separately from committed versions.

Include one-worker GPU queue time in simulation deadlines. Bound collection and arrange timed-out client-job cleanup so abandoned work does not hold compute indefinitely. Checkpoints use atomic replacement after a validated aggregate; interrupted writes preserve the previous committed state.

Verify mathematical round trips and actual failure behavior separately. Required cases include nonfinite/wrong-shaped payloads, missing tensors, duplicate/late/stale replies, zero deltas, and missed quorum. A real network/process interruption and fresh rejoin are required before claiming live recovery.

Keep results under `runs/`. Document the run contract, rejected reasons, coefficients, norm/clip telemetry, and participation; clipping alone establishes no malicious-client robustness guarantee.
