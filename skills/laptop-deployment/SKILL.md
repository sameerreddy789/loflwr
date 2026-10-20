---
name: laptop-deployment
description: Prepare and rehearse the three-laptop FedLoRA-Edge Flower deployment with cached offline assets, known-node authentication, measured CPU budgets, and real dropout/rejoin.
---

# Laptop deployment

Use [roadmap.md](../../research/roadmap.md) for stage gates and [system-design.md](../../research/system-design.md) for identities and failure contracts. Deployment follows successful real local training and protocol checks.

The host provides SuperLink/evaluation/dashboard and lifecycle/payments training. CPU laptop 1 covers transfers/cash; CPU laptop 2 covers verification. Require at least 8 GB RAM on CPU clients and measure whole-process memory, including base weights/runtime. Reserve additional host memory for evaluation.

Use the same ClientApp training function as simulation. The real run has three clients and quorum two; it is distinct from the five-client benchmark. Start with a 60-second collection deadline and calibrate actual optimizer-step budgets before rehearsal.

Prepare architecture/OS-appropriate tested dependencies, model/tokenizer revision, common A, manifest/config hashes, and transport credentials before going offline. Disable runtime downloads and perform a fresh start with internet disconnected. A warm process with cached objects is not the fresh-start acceptance check.

Use a dedicated LAN/hotspot, TLS, and authenticated known nodes. Keep private keys in ignored `certificates/`, public model/data assets in ignored `assets/`, and run outputs in ignored `runs/`.

Demonstrate actual mid-training process or network loss, bounded collection, survivor commit or below-quorum retention, and current-state synchronization on return. Reject old partial updates. Record wire bytes, work, timing, coefficients, and participation.

Dashboard replay is labeled and sourced from a real run. Current or completed measurements carry run identity; no invented metrics are inserted to make the demonstration appear complete. The query interface shows the allowed intent and reviewed authored response.

Data locality/TLS are not differential privacy or secure aggregation. Preserve those distinctions in the judging explanation and do not extend deployment to private production banking data merely because the public benchmark runs.
