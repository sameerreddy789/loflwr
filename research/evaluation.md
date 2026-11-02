# Reproducible evaluation plan

This document fixes the benchmark before training. All model-performance and deployment acceptance checks below are pending. The existing tensor-state tests are not substitutes for these experiments.

**2 November checkpoint addendum:** [seven-day-prototype.md](seven-day-prototype.md) stages a seed-42 pilot for the solo working-demo deadline. It retains the selected manifests and methods, includes the uploaded abstract's centralized reference, and does not mark the full twenty-attempt/three-seed matrix complete. Any shorter pilot horizon must be fixed before comparison/test evaluation and reported explicitly.

## Task and fixed domains

Use queries and labels from the original BANKING77 dataset. Train a causal language model to complete the intent name. At inference, score the 20 permitted completions using the fixed rule in [system-design.md](system-design.md), select one, and map it to a team-authored response.

| Client ID | Domain | Four selected intents |
| --- | --- | --- |
| 0 | Card lifecycle | `card_arrival`, `card_delivery_estimate`, `activate_my_card`, `card_about_to_expire` |
| 1 | Card payments | `pending_card_payment`, `declined_card_payment`, `card_payment_not_recognised`, `transaction_charged_twice` |
| 2 | Transfers | `pending_transfer`, `transfer_not_received_by_recipient`, `transfer_fee_charged`, `cancel_transfer` |
| 3 | Cash withdrawals | `pending_cash_withdrawal`, `declined_cash_withdrawal`, `wrong_amount_of_cash_received`, `cash_withdrawal_not_recognised` |
| 4 | Account verification | `edit_personal_details`, `verify_my_identity`, `unable_to_verify_identity`, `why_verify_identity` |

Labels retain the dataset's original spelling. BANKING77 provides user queries and intent labels; it does not provide the canonical banking responses. Review templates separately and avoid promises about fees, refunds, delivery dates, or identity handling that the demonstration cannot substantiate.

## Data manifest

Pin a dataset repository revision and hash the source CSV files. Preserve source split and original row number in every selected record. Freeze the data-selection seed separately from training seeds.

| Split/scenario | Construction | Total |
| --- | --- | --- |
| Validation | 10 unique original-training queries per intent | 200 |
| Final test | 25 unique official-test queries per intent | 500 |
| Balanced train | 40 queries per intent; each client owns its four intents | 800, or 160 per client |
| 90% imbalance train | Client 1 gets 90 per intent; each other client gets 3/3/2/2 across its four intents | 400: 360 on client 1 and 10 on each other client |

Deduplicate using Unicode NFKC normalization, case folding, and whitespace collapse. Retain original text for training. Detect conflicting labels for the same normalized query and stop for review. This catches normalized duplicates, not every semantic paraphrase.

Choose official-test records first, remove their normalized matches from training eligibility, then select validation and training records. Enforce disjoint normalized query sets between validation, final test, and all clients within a scenario. Training overlap between balanced and imbalanced scenarios is allowed because they are alternative experiments; it must not leak into held-out splits.

Record per-intent counts, excluded duplicates, source identifiers, selection seed, normalization version, and stable hashes of each manifest. Verify that enough eligible unique examples exist for every required count; fail rather than silently sampling with replacement or changing an allocation. The 90% setup is intentionally an extreme few-shot condition for minority clients, not a general statement about realistic banking organizations.

## Matched training configuration

Pin the selected SmolLM2 model/tokenizer revision and common A tensors. Initialize B identically. Use rank 8, alpha 16, dropout 0.05 on `q_proj`/`v_proj`, learning rate `2e-4`, AdamW weight decay 0, gradient clipping 1, microbatch 1, accumulation 4, and maximum sequence length 128. Reset optimizer state each attempt.

Derive local randomness from run seed, attempt, and client ID so omitting one client does not change the other clients' random streams merely by changing execution order. Keep example order, optimizer-step budgets, evaluation rules, and dropout schedules matched between strategies.

Equal-budget runs use 8 optimizer steps per client per attempt. Initial heterogeneous budgets are `[2, 16, 8, 4, 2]` in the domain-table order. Record calibration changes before freezing comparison runs. Artificial sleep belongs in a separate network-delay test; it does not replace an actual compute/work limit.

## Baselines

| Configuration | Purpose |
| --- | --- |
| Frozen base | Establish initial 20-intent task quality under the same scoring rule |
| Centralized B-only reference | Same public training union, initialization, scoring, and declared processed work; preserve the uploaded abstract's promised comparison |
| Shared-A, sample-weighted FedAvg | Naive influence baseline; coefficients use manifest training sizes |
| Shared-A, equal averaging | Isolate the coefficient-policy change |
| Shared-A, equal weights and median clipping | Main proposed system |
| Main system with B-coordinate FedProx | Optional local-drift ablation |

All adapter baselines share A. Training both factors in only one baseline would confound factor aggregation with weighting and clipping. The centralized reference uses the minibatch replay and optimizer reset contract in [abstract-alignment.md](abstract-alignment.md); it is not a guaranteed upper bound. A TF-IDF or encoder classifier is separate task-efficiency context, not a substitute for the promised fine-tuning reference or the controlled federated comparisons.

For FedProx, tune mu using validation only from `1e-4`, `1e-3`, and `1e-2`. Keep the B-start reference detached and fixed. Account for that penalty consistently in logged losses; report task loss separately from penalized training objective.

## Principal runs

Run the sample-weighted baseline and main system for 20 **attempts** each, with seeds 42, 43, and 44, under each condition:

1. Balanced data, equal budgets.
2. Balanced data, heterogeneous budgets.
3. 90% data imbalance, heterogeneous budgets.
4. The same imbalance and budgets, with scripted mid-training dropout and rejoining.

This is 24 principal runs and 480 scheduled attempts. That count excludes the additional plain-equal control required for claims separating weighting from clipping, the centralized reference, the omission study, and optional FedProx ablations. Plain equal is required for mechanism attribution; FedProx is optional. A skipped attempt still consumes its attempt number and wall time; it must not appear as a completed training round.

Interpret this matrix carefully: balanced/imbalanced conditions use 800/400 total unique training queries; equal/heterogeneous work totals 40/32 optimizer steps per attempt; the largest dataset is also the fastest-budget client and owns card-payment intents. Matched strategy comparisons within a condition remain informative. Differences across conditions do not isolate data quantity, data skew, local work, and domain identity. Plain equal versus sample weighting is identical on balanced equal-size manifests, so that cell isolates clipping only when comparing with the main method.

For a reproducible initial dropout schedule, disconnect client 1 on attempts 4, 8, and 12 after two completed optimizer steps. It rejoins on the next attempt after fresh synchronization. Store the exact schedule and trigger in run configuration. Run an actual process/network interruption as well as a deterministic application-level simulation; a Python exception alone does not validate the networked demonstration.

Use validation for monitoring and optional hyperparameters. Evaluate the final committed checkpoint after the twentieth attempt on the final test, including runs with skipped attempts. Any alternative checkpoint-selection rule must be declared before final-test access.

## Weak-client omission study

Compare otherwise matched main-system runs with and without weak client 4, using the 90%/heterogeneous condition without scripted dropout to isolate its marginal contribution. Keep the other clients' samples, budgets, random streams, attempt count, and quorum rule fixed. Report verification-domain accuracy, overall macro F1, worst-domain accuracy, and wall time.

Give paired seed differences and uncertainty. A positive effect is an empirical requirement to support the claim that weak clients contribute usefully. If the effect is negligible or unfavorable, report it and investigate the budget/task choice; the existence of a nonzero update is not proof of benefit.

This estimates the total effect of including client 4 in the specified protocol. Omission changes available examples, survivor count, coefficients, and median clipping. It does not isolate slow hardware as the source of any benefit. A single-seed checkpoint omission run is preliminary and does not replace paired multi-seed confirmation.

## Metrics and result records

| Category | Required measurements |
| --- | --- |
| Task quality | Completion-only validation loss, test intent accuracy, macro F1, per-domain accuracy, worst-domain accuracy, and domain gap |
| Training progress | Attempt and committed-version counters, completed optimizer steps, microbatches, samples/tokens processed |
| Participation | Selected, successful, rejected, timed-out, and returning clients; per-client participation rates |
| Influence | Server coefficient, unclipped norm, clip threshold/factor, clipped norm, aggregate norm |
| Resources | Peak process RSS, measured GPU allocated/reserved peaks where available, hardware/runtime details |
| Communication | Raw B bytes, actual serialized/wire bytes by direction, initial A/asset transfer separately |
| Timing | Local training time, queue wait, collection/aggregation/evaluation time, and total wall time |

Publish run ID, seed, manifest/config/A hashes, model revision, dependency lock identity, hardware, strategy, deadline, and schedule with every result table. Report seed means and standard deviations and show individual runs. Distinguish declared sample ownership from actual processed work; client-reported telemetry is not verified proof of work.

## Required verification and acceptance

- **Real local smoke:** B changes after optimization on CPU and GPU; base and A hashes remain identical. Verify the selected tensor count and actual peak memory.
- **Masking:** prompt and padding labels are ignored; intent and genuine EOS remain supervised, including PAD/EOS aliasing and truncation cases.
- **Mathematics:** adapter round-trip identity, common-A effective-update equivalence, cross-tensor norms, median semantics, and clipping arithmetic pass checks.
- **Protocol:** wrong shapes, missing tensors, NaNs/Infs, wrong run/version/base hashes, duplicates, late replies, zero deltas, and missed quorum produce their specified outcomes.
- **Failure handling:** actual mid-training client loss results in a bounded attempt; sufficient survivors commit; insufficient survivors retain the checkpoint; returning clients receive fresh state.
- **Offline rehearsal:** a clean three-laptop start succeeds with internet disconnected and previously cached dependencies/assets.
- **Learning evidence:** report loss/quality trajectories under heterogeneity and final paired strategy comparisons. Claim improvement only when observed; falling validation loss alone is not a convergence theorem.
- **Weak-client benefit:** the matched omission comparison supports useful contribution before that claim is made.

No benchmark scores are prefilled. The system may fail these gates; that finding belongs in the report. Do not claim guaranteed convergence, superior accuracy, differential privacy, or malicious-client robustness from these experiments.

## Three-laptop experiment

Map lifecycle/payments to the host client, transfers/cash to CPU laptop 1, and verification to CPU laptop 2. Require two valid replies and calibrate work to a 60-second live deadline. Domain ownership, client count, and compute differ from the five-client benchmark, so use distinct run IDs and report those results separately.
