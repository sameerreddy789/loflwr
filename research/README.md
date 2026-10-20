# FedLoRA-Edge research dossier

**Research and design delivery, 20 October 2026.** This dossier records the approved implementation target and primary-source review. It contains no completed training results.

## Recommendation

Build a small English banking-support system that trains across ordinary laptops, shares only adapter updates, exposes survivor aggregation coefficients, and recovers when a participant disappears during training. Use SmolLM2-135M-Instruct, Flower's ClientApp/ServerApp architecture, and a shared frozen-A adapter with B-only updates.

The bounded task has 20 BANKING77 intents across five domains. Predict an allowed intent by scoring its completion; map the prediction to a team-authored response. This makes task quality measurable and avoids treating unrestricted generated banking advice as the product.

The deployment target is three laptops with CPU clients having at least 8 GB RAM. The main benchmark has five simulated clients. These experiments must be reported separately. Use WSL2 and one GPU training worker at a time for the single-machine benchmark.

## Navigation and document purposes

| File | Purpose | Reader |
| --- | --- | --- |
| [landscape.md](landscape.md) | Explanation and comparisons of existing approaches | Team selecting methods and judges assessing novelty |
| [system-design.md](system-design.md) | Reference contract for training, aggregation, and failure handling | Implementers and reviewers |
| [evaluation.md](evaluation.md) | Reference protocol for reproducible comparisons | Experiment owners and result reviewers |
| [roadmap.md](roadmap.md) | Build and rehearsal guide with concrete gates | Team planning the implementation week |
| [sources.md](sources.md) | Source register, review depth, licenses, unresolved citations | Anyone verifying claims or reusing code/data |
| [implementation-status.md](implementation-status.md) | Delivery status and outstanding evidence | Contributors and judges |

## Corrections to the original proposal

1. **FedProx penalizes drift, not dataset-size influence.** With sample-weighted aggregation, a client owning 90% of the declared training samples may still receive a coefficient of 0.9. Equal weighting is a separate policy.
2. **Client-count settings are not response quorums.** A selected client can fail after it starts training. Set both message TTL and response-collection timeout, then count valid replies before committing a checkpoint.
3. **Freezing A and sharing B are established research.** Do not present FFA-LoRA as a new invention. Demonstrate measured usefulness under weak compute, low traffic, visible influence, and actual recovery.
4. **A small adapter does not make the whole model small in memory.** Frozen base weights, activations, gradients for B, optimizer state, tokenizer behavior, and runtime overhead still matter.
5. **Local raw data is not a formal privacy guarantee.** Updates can leak information; secure aggregation and differential privacy require additional mechanisms and separate evaluation.

## Evidence labels

- **Primary-source finding:** supported by a paper, model card, framework documentation, or repository reviewed at the depth shown in the source register.
- **Engineering decision:** a choice locked by the team for this build, not a universal research result.
- **Arithmetic estimate:** computed from tensor dimensions or bytes per element; not a measured memory/traffic result.
- **Experiment pending:** a hypothesis or acceptance criterion that still needs an actual model run.

Reading a paper's abstract and metadata does not establish replication of its results. An accessible repository is not evidence that its code runs on our laptops. Model-card scores are publisher results, not our BANKING77 results. Recent plan citations are checked individually and retained as unresolved candidates if their identity cannot be confirmed.

## Chosen boundaries

Keep adapter ranks fixed. Keep synchronous attempts with deadlines. Keep trusted known nodes and public benchmark data for the hackathon. Larger models, heterogeneous ranks, buffered asynchronous optimization, secure aggregation, and differential privacy remain documented extensions.

Success requires empirical evidence: useful intent prediction, measured convergence under heterogeneity, a weak-client omission comparison, and bounded dropout/rejoin behavior. Unfavorable comparisons must appear in the report. See [evaluation.md](evaluation.md) for exact conditions and [implementation-status.md](implementation-status.md) for current progress.
