# FedLoRA-Edge research dossier

**Initial delivery: 20 October 2026. Implementation update: 3 November 2026.** This dossier records the selected design and primary-source review. See the [results register](results/README.md) for measured CPU/GPU and Flower checks, and [implementation status](implementation-status.md) for remaining acceptance gates. The checkpoint constraint is one solo builder with a working demo due in one week.

Start with the [seven-day prototype plan](seven-day-prototype.md). It stages a preliminary comparison before the full evaluation program. The [ideation synthesis](ideation.md) explains the reasoning, and [abstract alignment](abstract-alignment.md) records changes from the newly reviewed GPT-2/WikiText-2/capped-weight proposal. The selected SmolLM2/BANKING77/shared-A design remains active.

The [accuracy diagnosis](accuracy-diagnosis.md) combines partial federated
run and code audits with primary-source research. A copied version-9 checkpoint
scores 34.5% under the original rule and 64.5% with content-free centering on
validation. The separate TF-IDF comparison scores 41.5% versus 87% across
imbalanced/balanced data. These are exploratory results from distinct methods
and data regimes, not final federated comparisons.

The follow-up [99% feasibility review](accuracy-99-feasibility.md) distinguishes
benchmark accuracy from selective precision, verifies published numerical
comparisons, and reports twelve completed reference checks. The best achieves
95.5% validation accuracy with frozen MPNet and 2,760 eligible training examples;
none reaches 99% top-1 accuracy. A reproduced post-hoc candidate diagnostic
finds 99% top-2 recall; this is not classification accuracy. The review also
records completed original-scoring federated
comparisons and suspected annotation issues, keeping these scopes distinct.

## Recommendation

Build a small English banking-support system that trains across ordinary laptops, shares only adapter updates, exposes survivor aggregation coefficients, and recovers when a participant disappears during training. Use SmolLM2-135M-Instruct, Flower's ClientApp/ServerApp architecture, and a shared frozen-A adapter with B-only updates.

The bounded task has 20 BANKING77 intents across five domains. Predict an allowed intent by scoring its completion; map the prediction to a team-authored response. This makes task quality measurable and avoids treating unrestricted generated banking advice as the product.

The deployment target is three laptops with CPU clients having at least 8 GB RAM. The main benchmark has five simulated clients. These experiments must be reported separately. Use WSL2 and one GPU training worker at a time for the single-machine benchmark.

## Navigation and document purposes

| File | Purpose | Reader |
| --- | --- | --- |
| [ideation.md](ideation.md) | Explanation of problem discovery, hypotheses, alternatives, and evidence gates | Solo builder and reviewers |
| [seven-day-prototype.md](seven-day-prototype.md) | Working-demo guide with a staged pilot and realistic cut criteria | Solo implementer |
| [abstract-alignment.md](abstract-alignment.md) | Uploaded proposal review, design changes, and supplied competition criteria | Builder and checkpoint reviewers |
| [landscape.md](landscape.md) | Explanation and comparisons of existing approaches | Team selecting methods and judges assessing novelty |
| [system-design.md](system-design.md) | Reference contract for training, aggregation, and failure handling | Implementers and reviewers |
| [evaluation.md](evaluation.md) | Reference protocol for reproducible comparisons | Experiment owners and result reviewers |
| [roadmap.md](roadmap.md) | Build and rehearsal guide with concrete gates | Team planning the implementation week |
| [sources.md](sources.md) | Source register, review depth, licenses, unresolved citations | Anyone verifying claims or reusing code/data |
| [implementation-status.md](implementation-status.md) | Delivery status and outstanding evidence | Contributors and judges |
| [accuracy-diagnosis.md](accuracy-diagnosis.md) | Current accuracy findings and ordered validation experiments | Builder and reviewers |
| [accuracy-code-audit.md](accuracy-code-audit.md) | Training, tokenization and scoring correctness checks | Implementers |
| [accuracy-run-audit.md](accuracy-run-audit.md) | Partial seed-42 quality, work, influence and data coverage | Experiment reviewers |
| [accuracy-literature.md](accuracy-literature.md) | Primary-source explanation of likely limitations and candidate improvements | Method selection |
| [accuracy-99-feasibility.md](accuracy-99-feasibility.md) | Evidence for the 99% aspiration and tests of stronger task references | Builder and reviewers |

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
