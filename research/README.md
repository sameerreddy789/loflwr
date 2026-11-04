# FedLoRA-Edge research dossier

**Initial delivery: 20 October 2026. Latest engineering review: 4 November 2026.** This dossier records the selected design and primary-source review. See the [results register](results/README.md) for measured CPU/GPU and Flower checks, and [implementation status](implementation-status.md) for remaining acceptance gates. The checkpoint constraint is one solo builder with a working demo due in one week.

The new [ML engineering guidelines](ml-engineering-guidelines.md) map primary
Google, scikit-learn and PyTorch guidance to this repository. The targeted
[efficiency paper review](efficiency-paper-review.md) connects seven papers and
the upstream batching implementation to an optional
[bucketed local preparation path](../docs/bucketed-feature-preparation.md).
The [model card](../MODEL_CARD.md) collects the confirmed model's intended use,
identity, domain scores and limitations in one report. These source reviews do
not themselves establish new speed or accuracy measurements.

The delivered accuracy variant uses frozen MPNet with a federated twenty-class
head. It reached **94.0% independent confirmation accuracy (282/300)** after
selection at **95.5% validation accuracy (191/200)**. All 500 actual Flower
attempts accepted five clients; a trained-model process dropout and fresh rejoin
also completed. The [final report](results/encoder-v2/README.md) includes adverse
results and all frozen selective policies: none reached 99%. The strictest
policy achieved 98.65% on answered queries at 74.33% coverage.

The [public-dataset review](public-dataset-evaluation.md) led to a separately
locked [CLINC150 external test](results/external-clinc-v1/README.md): **28/29
correct on one compatible intent**, and **5/120 unsupported banking false
accepts** under the strictest fixed margin rule. It does not replace the
all-twenty-intent result. No directly compatible full-task public test was found
among the reviewed candidates.

The [client-local preparation report](results/encoder-local-v1/README.md) now
checks both deployment layouts with isolated offline encoding and unchanged
train/validation predictions. The [scope-detector report](results/scope-v1/README.md)
adds a locked negative-only test on 2,279 previously unscored unsupported
BANKING77 queries: the strictest margin plus scope accepts 57, versus 311 with
margin alone. The scope detector is experimental; this does not establish new
all-intent accuracy or fresh supported-query coverage. Follow the
[encoder laptop guide](../docs/encoder-laptop-rehearsal.md) for the physical
rehearsal that remains pending.

The [seven-day prototype plan](seven-day-prototype.md), [ideation synthesis](ideation.md),
and [abstract alignment](abstract-alignment.md) preserve the design process.
The original SmolLM2/BANKING77/shared-A benchmark and its results remain intact.
The user-authorized MPNet head and B-plus-head variants have separate identities;
head-only training is not LoRA.

The [representation diagnostics](results/encoder-representation/README.md) test
larger transformer-layer updates using pooled training data and reused validation.
They are centralized development references. Use the
[judging presentation guide](../docs/judging-presentation.md) to demonstrate the
confirmed Flower head and clearly separate these research results.

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

The [accuracy recovery plan](accuracy-recovery-plan.md) records the rationale for
the now-implemented encoder/classifier experiments and internal quality gates. Final-checkpoint
scoring corrections reach 52%/63%/58.5% validation for sample/equal/clipped;
the [seven-configuration pilot](results/pilot-r2-20261103/README.md) is complete,
including an unfavorable/null weak-client omission comparison. Those original
LoRA results must not be mixed with the expanded-data MPNet result. The
[encoder three-seed matrix](results/encoder-matrix-v1/README.md) is now complete:
33 runs, matched weighting/clipping, budgets, imbalance and omission controls.
The original LoRA full multi-seed program and physical three-laptop deployment
remain separate pending work.

## Original recommendation and delivered variant

Build a small English banking-support system that trains across ordinary laptops, shares only adapter updates, exposes survivor aggregation coefficients, and recovers when a participant disappears during training. Use SmolLM2-135M-Instruct, Flower's ClientApp/ServerApp architecture, and a shared frozen-A adapter with B-only updates.

That original benchmark is implemented and preserved. For the working query
demo, use the delivered frozen-MPNet classifier head: 15,380 trainable parameters
and 61,520 raw FP32 bytes per direction. Its clients load separate feature
archives, but those public-data features were prepared centrally before
partitioning. This is not evidence of distributed feature preparation or
physical isolation. The separate frozen-A MPNet adapter trial did not improve
validation accuracy and remains reported.

The bounded task has 20 BANKING77 intents across five domains. The original
LoRA model scores intent completions; the delivered encoder uses a 20-class
head. Both map a predicted intent to a team-authored response. This makes task
quality measurable and keeps generated banking advice outside the product.

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
