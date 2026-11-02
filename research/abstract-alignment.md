# Abstract review and design alignment

Reviewed **2 November 2026** for a solo builder with a working-prototype checkpoint in one week. This is an explanation of the uploaded proposal and the active design, not a replacement submission or an organizer approval.

## What was reviewed

The user supplied `Cheap Fine-Tuning Across Weak, Scattered Devices Abstract.pdf`: four pages, 38,956 bytes, SHA-256 `ce7e2f358dceea96100841639aaa785c4bb93068925d8cc09add4664ebeaf46c`. All four pages were text-extracted and reviewed. The original remains a local user file, unmodified and outside this documentation commit. Its submission/acceptance status has not been independently established.

Pages 1–2 contain the eight-section abstract; page 3 contains architecture and tools; page 4 contains the build checklist, evaluation design, and risks. The first two extracted pages contain approximately 955 whitespace-separated words, including headings and repeated headers. The supplied rules suggest 600–900 words plus an optional architecture page. This is a reason to condense a future permitted revision, not evidence of automatic disqualification. Do not replace or resubmit an existing entry without checking the portal's rules.

## Assessment

The abstract correctly identifies three separate difficulties: heterogeneous local optimization, sample-count influence, and incomplete participation. It names a reproducible baseline, specifies held-out evaluation, and explicitly allows negative findings. Its privacy and fairness limitations are appropriately stated.

The main weakness is feasibility detail. It does not specify partitions, step budgets, response validation, a final cap algorithm, or numerical resource measurements. Its team section describes work divided among roles, while the current implementation team has one person. The checkpoint plan must acknowledge that constraint and prioritize a functioning learning pipeline.

The active design remains the SmolLM2/BANKING77/shared-A specification in [AGENTS.md](../AGENTS.md). The uploaded abstract is evidence of earlier commitments; it does not silently replace that specification or the separately preserved [legacy Qwen brief](reference-team-brief/README.md).

## Commitments and changes

| Uploaded abstract | Current design | Assessment and action |
| --- | --- | --- |
| GPT-2 small, TinyLlama conditional on hardware | SmolLM2-135M-Instruct | Explicit model change. Justify with instruction tuning, selected architecture, and measured CPU feasibility when available. GPT-2 small and SmolLM2-135M are similar in parameter scale; do not claim SmolLM2 is smaller than GPT-2. |
| WikiText-2 initial language-model corpus | Twenty BANKING77 intents | Material benchmark change: open-ended next-token modeling becomes closed-set intent completion. It makes per-domain utility testable; scores are not comparable to WikiText perplexity. |
| Frozen base, train/share LoRA adapters | Common frozen A, train/share B | A specified LoRA variant supported by FFA-LoRA. Verify actual shared tensors and effective-update equivalence. Cite the existing method. |
| Sample-count-weighted FedAvg | Shared-A sample-weighted FedAvg | Retained. All federated comparators use identical adapter coordinates. |
| Capped-weight stretch comparison | Equal survivor weights, with and without median norm clipping | A policy revision. Equal weights and clipping are not the abstract's capped sample-weight algorithm. Explain the revision; do not relabel it. |
| Centralized, FedAvg, and capped comparison | Centralized reference plus the active federated trio | Restore the promised centralized reference to the checkpoint's minimum evidence. Capped aggregation is revised/deferred, not completed by another method. |
| Three to five simulated clients | Five simulated clients | Preserves the abstract's minimum scope. The separate three-laptop deployment is additional scope. |
| Timeouts, minimum participants, skipped rounds | TTL, collection timeout, valid quorum, versioning, atomic commit, fresh rejoin | Engineering detail needed to make the original reliability promise concrete. |
| Measurable improvement on held-out data | Completion loss and intent metrics | Still required for the learning claim. Correct message exchange or changed tensors alone is insufficient. |
| Repeated seeds and varied conditions | Seed-42 checkpoint pilot; broader study retained | Stage the evidence explicitly. One seed is preliminary and does not fulfill the later repeated-run program. |

The recommendation is to keep the current design and disclose these refinements in the checkpoint explanation. Whether a model/dataset/policy refinement is acceptable under the competition's continuity rule is an organizer decision, not established by this review. Work on the common federated-learning pipeline can proceed without assuming that decision.

## Why the cap description needs repair

The abstract says weights receive an upper limit and are then normalized. This does not generally cap the **final** coefficient. For sample fractions `[0.9, 0.025, 0.025, 0.025, 0.025]`, clipping at `0.4` gives a sum of `0.5`; normalization then assigns the rich client `0.8`.

A true capped distribution needs nonnegative weights that sum to one and individually satisfy `w_i <= c`. It is feasible only when `m * c >= 1`, where `m` is the number of survivors. One possible rule for positive sample counts is proportional redistribution, `w_i = min(c, lambda * n_i)`, choosing `lambda` so the weights sum to one. With five survivors and a 0.4 cap, the example becomes `[0.4, 0.15, 0.15, 0.15, 0.15]`. With only two survivors a 0.4 cap is infeasible. A cap-changing or skip policy would have to be declared before experiments.

These are algebraic observations, not a proposed new algorithm or a request to add another strategy this week. The current equal rule assigns `1/m`: 20% with five survivors, about 33.3% with three, and 50% with two in the separate laptop configuration. Median **update-norm** clipping acts on B deltas, not sample coefficients. Neither mechanism guarantees equal accuracy or equal cumulative participation.

## Centralized reference that preserves a fair comparison

Use a separate experiment on the union of the same public training rows. Keep the base model, A/B initialization, precision, tokenizer/scorer, optimizer hyperparameters, and total processed-work budget matched. This public-data reference does not alter the federated client's no-raw-query transport contract.

For the checkpoint, declare a deterministic per-client minibatch replay schedule. Within each attempt-sized block, cycle through active client queues in client-ID order, consuming one optimizer step from each until its declared budget is exhausted. Reset the centralized optimizer at block boundaries, as clients reset at round boundaries. Use the same client minibatch streams as the federated run and log actual steps/tokens. The centralized model updates sequentially throughout the block; federated clients start from the same snapshot, so their optimization trajectories deliberately differ.

For the initial heterogeneous budgets, one successful no-dropout block contains `2 + 16 + 8 + 4 + 2 = 32` optimizer steps. A twenty-block reference therefore plans 640 steps. These are workload counts, not measured runtime. Compare no-dropout references first; losses to deadlines must be reported as actual work differences rather than hidden by matching only nominal budgets.

Call this a **centralized B-only reference**, not a guaranteed upper bound. A cheap TF-IDF classifier answers a separate task-efficiency question and does not fulfill the centralized fine-tuning comparison.

## Suggested checkpoint explanation

> We retain the original objective: lightweight federated adapter training across heterogeneous clients, evaluated against sample-weighted FedAvg with explicit failure handling. We refined the model and task to SmolLM2-135M-Instruct and twenty BANKING77 intents to make CPU feasibility and domain-level utility measurable. We use the existing shared frozen-A method and compare sample weighting, equal weighting, and equal weighting with update clipping. This explicitly revises the proposed capped-weight stretch strategy. The checkpoint prioritizes five simulated clients and a preliminary matched comparison including centralized B-only training; physical deployment and multi-seed confirmation remain separate milestones. Completed results and unresolved requirements will be reported individually.

Replace future-tense status with measured facts only after the runs exist. Do not imply the organizer has approved this wording.

## Competition criteria and clarification drafts

The weights below come from the user's supplied **Round 1** rules. They have not been independently verified as the finals scoring rubric.

| Criterion | Supplied weight | Evidence the prototype can make concrete |
| --- | --- | --- |
| Problem understanding | 20% | Separate data skew, local work, availability, and update influence |
| Technical soundness and feasibility | 25% | Real B-only optimization, compatible replies, bounded attempts, measured resources |
| Innovation | 20% | A precisely evaluated participation/influence policy and failure behavior; credit existing LoRA/FL methods |
| Clarity | 10% | One architecture, labeled results, honest description of the twenty-intent task |
| Evaluation rigor | 15% | Matched centralized/federated comparisons, plain-equal ablation, domain metrics, negative findings |
| Team readiness | 10% | A solo schedule with working artifacts, checkpoints, and rehearsal time |

The user supplied finals dates of 15–18 December 2026 and clarified that the one-week deadline is another working-demo checkpoint. Do not infer its exact submission time or treat it as Round 1. The supplied rules also describe a later one-page progress report; that does not replace the current working-demo requirement.

The following text is a **draft for the user to submit through the portal's Doubt section**. Nothing has been posted or emailed:

> Our proposal uses published LoRA/federated-learning methods and open-source libraries explicitly suggested by the problem statement. Does the originality clause permit original implementation and evaluation built on these attributed methods and appropriately licensed pretrained models/datasets? Which part of the submitted solution must be unpublished, and are there restrictions on public GitHub development?

If the uploaded PDF is the submitted version, a separate precise continuity question is:

> May we retain the same heterogeneous federated fine-tuning objective and FedAvg comparison while refining GPT-2/WikiText-2 to SmolLM2/BANKING77 intent classification and replacing the capped-weight stretch strategy with equal survivor weights plus update-norm clipping? We will document the changes, preserve centralized comparison, and report simulated and physical-device evidence separately.

The ambiguity calls for a factual clarification, not an unsupported claim that using the prescribed libraries is prohibited. See [the seven-day plan](seven-day-prototype.md) for implementation priorities and [sources.md](sources.md) for provenance.
