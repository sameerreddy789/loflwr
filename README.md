# FedLoRA-Edge

**Current measured result, 3 November:** the frozen-MPNet/federated-head variant
scored **94% on independent confirmation (282/300)** and **95.5% on validation
(191/200)**. Its actual five-client Flower run completed 500 attempts. Unequal-budget
and clipping comparisons also reached 95.5% validation; a real process dropout
and rejoin preserved that validation score. See the
[complete evidence](research/results/encoder-v2/README.md) and
[remaining-error analysis](research/encoder-error-analysis.md).

The model and clarification thresholds were locked in commit `7413e6b` before
confirmation. The strictest frozen rule answered 223/300 queries and got 220
correct: **98.65% accuracy on answered queries at 74.33% coverage**. It declined
the other 77. Neither full-set nor selective 99–100% accuracy was achieved.

A separate [external CLINC150 check](research/results/external-clinc-v1/README.md)
now scores **28/29 (96.55%) on one compatible intent**, with all twenty labels
still competing. The strictest existing margin rule also falsely accepted
**5/120 unsupported banking requests**. This is a narrow transfer/scope test;
the all-twenty-intent confirmation remains 94%. See the
[public-dataset review](research/public-dataset-evaluation.md) for compatibility,
deduplication and source limitations.

The latest [client-local preparation checks](research/results/encoder-local-v1/README.md)
now pass in both five- and three-client layouts, with each isolated process
reading its own records and **zero changed predictions across 2,960 train/validation
examples per layout**. A separate [locked scope-detector study](research/results/scope-v1/README.md)
reduces unsupported-request false accepts from **311/2,279 to 57/2,279** under
the strictest fixed margin. It remains experimental: this negative-only test
does not change the confirmed 94% intent accuracy or establish fresh positive coverage.
The new [isolated Flower rehearsals](research/results/remaining-work-v1/README.md)
also pass three- and five-client dropout, below-quorum retention, fresh rejoining,
and fresh three-client startup in a loopback-only network namespace.

Before presenting, run the [demo preflight](docs/demo-preflight.md) on the
prepared laptop:

```powershell
.\tools\check_demo.ps1
```

It checks dependencies, published evidence, the dashboard port, the confirmed
checkpoint and actual offline inference. If it says `open_existing`, open the
reported URL. If it says `launch`, start the query demo:

```powershell
.\tools\start_encoder_demo.ps1
```

Open <http://127.0.0.1:8511>. The [encoder demo guide](docs/encoder-demo.md)
explains live classification, recorded evidence and supported intents.
Use the [three-minute judging script](docs/judging-presentation.md) for the
spoken presentation, exact demo sequence, and answers to likely judge questions.
The [updated presentation figure](research/results/remaining-work-v1/comparison.png)
visualizes the three-seed imbalance/omission comparisons and separate scope test.

The [six-slide PowerPoint](presentation/README.md),
[short narrated backup](presentation/narrated-demo.md), and
[automated preflight](docs/demo-preflight.md) are ready.
The sidebar now has **Live banking query**, **Recorded recovery**, **Experiment
comparisons**, and **Custom run**. Live mode enables the fixed clarification rule
by default. The [fresh-query kit](docs/fresh-query-collection.md) creates empty
blinded collection packets and checks readiness; human submissions and physical
multi-laptop testing remain pending.

The [fresh evaluation runner](docs/fresh-evaluation-runner.md) is now implemented.
It requires an eligible human collection, a published frozen plan, unchanged
inputs and a fresh GitHub publication check before claiming queries for one-time
scoring. Today's empty pack correctly blocks it. A
[fresh-server browser rehearsal](research/results/rehearsal-evaluator-v1/README.md)
also passed ten functional observations, including clarification, recorded quorum
retention/rejoining and backup-video playback. The human presentation remains to
be rehearsed.

Federated adapter fine-tuning across ordinary laptops for IIT Bombay's **Efficient AI at Scale** hackathon problem.

The original SmolLM2 experiment trains B-only LoRA adapters. The accuracy-recovery variant trains a small classifier head on frozen MPNet features; a separate MPNet B-plus-head trial is also implemented. The server validates updates, exposes contribution weights and commits only when enough valid replies arrive before the deadline. Head-only training is not called LoRA.

**Current delivery: a runnable one-laptop prototype.** Deterministic BANKING77 preparation, real CPU/GPU frozen-A training, Flower applications, checkpointed rounds, a frozen-study launcher, and a dashboard are implemented. Actual process interruption, below-quorum retention, rejoining, and fresh offline deployment have been exercised. The full WSL environment passes 260 tests with no skips. See the measured results for learning comparisons; successful execution alone does not establish convergence or deployment across three physical laptops.

Start with the [morning demo guide](docs/morning-demo.md) on the prepared laptop. The [implementation status](research/implementation-status.md) distinguishes tested behavior from remaining gates. Measured evidence and its limits belong in the [results register](research/results/README.md).

## Test statistics and measured results

Verified on **4 November 2026** using the [tested environments](requirements/README.md).
The [handoff record](research/results/encoder-v2/handoff-verification.json)
contains the run/checkpoint identities and hashes of the original check logs.
The [latest delivery verification](research/results/rehearsal-evaluator-v1/verification.json)
records 260 passing tests, a browser rehearsal, the empty-collection refusal and
unchanged original, CLINC and scope evaluation locks. The
[previous extension](research/results/demo-extension-v1/verification.json)
records 246 passing tests, the preflight, narrated backup and resource study. The
[previous demo delivery](research/results/demo-delivery-v1/verification.json)
preserves the 225-test result and original slide/recording hashes.
The [previous runtime verification](research/results/remaining-work-v1/verification.json)
preserves the 203-test baseline.
It covers local ownership, portable configuration, rejoin readiness, bounded
startup retries, orphan-process cleanup and cancellation. The prior
[representation verification](research/results/encoder-representation/verification.json)
records three earlier checks for training data boundaries and frozen layers.

| Verification | Observed result | Scope |
| --- | --- | --- |
| Full automated suite, Ubuntu WSL2 | **260 passed, 0 failed, 0 skipped** in 26.109 s | Existing checks plus fresh-evaluation publication, record integrity, once-only inference, metrics and FP32 boundary fixtures |
| Fresh-server browser rehearsal | **Ten functional observations passed** | Live/blank/ambiguous input, recorded three/five-client recovery, comparisons, supported intents and browser video playback; no human rehearsal or new accuracy score |
| New evaluation gate | **Empty real collection refused with exit 2** | Zero human queries; no evaluation lock, inference or exposure claims created |
| Automated demo preflight | **All five checks passed** | Dependencies, published evidence, matching dashboard configuration, confirmed checkpoint and actual offline functional inference |
| Encoder dashboard checks, Windows CPU | **7 passed, 0 failed, 0 skipped** in 8.033 s | A separately run subset of the full suite, not 7 additional unique tests |
| Locked artifact integrity | **24/24 original, 36/36 external, 40/40 scope file hashes matched** | Overlapping locks, not 100 unique files; assets, checkpoints, sources and evaluation configurations |
| Live CPU query check | Passed; a fresh pending-card-payment query returned `pending_card_payment` | Functional inference check, not an accuracy benchmark |
| Actual five-client Flower training | **500/500 attempts committed** with all five contributors | `encoder-flower-balanced-v2`, selected version 500 |
| Trained-model dropout and recovery | **3/3 attempts committed**; 5 → 4 → 5 accepted clients | One laptop; a client process was killed during attempt 2 and rejoined with fresh state in attempt 3 |
| Client-local preparation | **Five- and three-client layouts passed**; zero changed predictions on 2,960 train/validation rows each | Separate mount/PID/network namespaces, pinned offline encoder, own records only; not physical laptops |
| Portable setup | **Four role configs and code-only Flower FAB built** | Selected head hash preserved; configuration check uses fixture node IDs |
| CPU query timing | **152 ms median / 229 ms p95**, 30 warm timing probes | Two CPU threads; 735.03 MiB peak process RSS; authored probes provide no accuracy measurement |
| Repeated encoder study | **33/33 runs completed**, 16,500 committed attempts across seeds 42/43/44 | Real head optimization with sequential mathematical aggregation; separate from Flower runtime evidence |
| New isolated TLS recovery | **3 → 2 → 3 → 1 → 3** and **5 → 4 → 5 → 2 → 5** accepted contributors | Real SuperNode process loss; below-quorum attempts skipped, current state checked on every accepted reply; one physical laptop |
| Fresh offline encoder deployment | **All 3 clients accepted** in a fresh loopback-only network namespace | External connectivity unavailable, dependencies already installed, runtime downloads disabled; not physical LAN evidence |

Timings are observations from this laptop, not controlled performance comparisons.
The full suite includes the dashboard tests. Missing optional dependencies can
produce skips elsewhere; a smaller passing suite does not reproduce this result.

| Model-quality measure | Result | Evaluation boundary |
| --- | --- | --- |
| Independent confirmation top-1 accuracy | **94.0% — 282/300 correct** | All 300 queries; checkpoint and scoring frozen before access |
| Confirmation macro F1 / worst-domain accuracy | **0.93914 / 88.33%** | Transfers was the weakest domain |
| Validation accuracy | **95.5% — 191/200 correct** | Reused development set that guided model selection |
| Strictest frozen clarification policy | **98.65% — 220/223 answered correctly** | **74.33% coverage**; 77/300 queries declined |
| External CLINC150 mapped-class top-1 | **96.55% — 28/29 correct** | One compatible intent only; not another twenty-intent score |
| External unsupported banking, strictest frozen margin | **5/120 false accepts (4.17%)** | 115 declined; unrelated-topic group separately had 0/120 false accepts |
| New unsupported BANKING77 scope test | **57/2,279 false accepts (2.50%)**, versus **311/2,279 (13.65%)** without scope | 57 unsupported intents; separate experimental detector; scope-only positive validation coverage 190/200, not fresh combined-system coverage |

See [all confirmation results](research/results/encoder-v2/confirmation-result.json)
and the [complete experiment report](research/results/encoder-v2/README.md),
including negative adapter results and matched imbalance/omission comparisons.
Passing software tests is separate from predicting banking intents correctly.

The [three-seed study](research/results/encoder-matrix-v1/README.md) gives an
**18-point paired validation benefit** from including the one-step verification
client in every seed: 95.5% versus 77.5%. It owns the only examples of four
intents. Under the separate 400-query 90%-imbalance condition, final accuracy
averaged **62.5% sample-weighted, 76.5% equal, and 78.17% ± 0.29 points clipped**.
All expanded-data methods finished at 95.5%. These are development results;
all selected/final outcomes, including regressions, are published.

The new [fixed-work study](research/results/resource-frontier-v1/README.md)
completed **nine runs across three seeds**. Each used 500 optimizer steps per
client on the same cached MPNet features and reused validation. Synchronizing
every four local steps reduced estimated raw payload by **75%**, with a
**2 percentage-point validation cost** in every seed:

| Local steps between averages | Rounds | Final validation accuracy | Estimated total raw payload | Observed process wall time, mean ± sample SD |
| --- | --- | --- | --- | --- |
| 1 | 500 | 95.5% | 307.6 MB | 85.50 ± 11.01 s |
| 2 | 250 | 93.5% | 153.8 MB | 48.48 ± 3.98 s |
| 4 | 125 | 93.5% | 76.9 MB | 28.37 ± 2.32 s |

MB is decimal. Payload is calculated across five clients, upload plus download;
it is not measured network traffic. Timings cover fresh serial CPU processes
optimizing the head, excluding feature preparation and Flower transport. This
study does not change the confirmed query model or its **94%** independent score.

## Future plans

These milestones are planned work, not completed results. Their order prioritizes
the weak-device/federation challenge alongside task quality.

| Priority | Next milestone | Completion evidence |
| --- | --- | --- |
| 1 | Rehearse the prepared deck, preset dashboard and recorded fallback with the presenter | Human three-minute rehearsal, fresh launch, live queries, clearly labeled event replay, and launch recovery |
| 2 | Test three physical laptops | Measured CPU/RAM budgets, authenticated TLS, bounded dropout/rejoin, and fresh startup without internet; only one laptop is available now |
| 3 | Confirm the combined classifier/scope policy on fresh supported queries | Independent coverage and accuracy; the new negative-only result alone does not justify deploying it |
| 4 | Improve accuracy with new development data and a new independent test | Train-only improvements selected on validation; freeze the next model and policies before scoring fresh held-out queries |
| 5 | Measure physical-network efficiency and evaluate later extensions | Wire traffic, hardware-specific memory/latency and repeated deployment runs; separate evidence for adapters, secure aggregation or differential privacy |

Client-local preparation and the 33-cell encoder study have moved from planned
work to measured results. The [remaining-work checklist](docs/remaining-work-plan.md)
and [physical laptop guide](docs/encoder-laptop-rehearsal.md) identify the next gates.

The [detailed future plan](docs/future-plans.md) gives dependencies and acceptance
criteria. The current 300-query confirmation set is now evaluation evidence;
it cannot serve as an untouched test for the next model-selection cycle.
**99–100% remains an unachieved target, not a promised milestone.**

## Accuracy research and original results

The new [representation diagnostic](research/results/encoder-representation/README.md)
tests transformer-layer adaptation on training data and reused validation only.
Both three-epoch and ten-epoch comparisons peaked at **96% validation (192/200)**;
the twelve-layer ten-epoch run ended at 94.5%. These are centralized development
references with much larger trainable states; they do
not replace the independently confirmed Flower head or establish 99–100% accuracy.
The [fresh-evaluation guide](docs/fresh-evaluation.md) describes the new data
needed to verify a subsequent model without reusing a spent test set.

The [99% feasibility investigation](research/accuracy-99-feasibility.md) reports
**95.5% validation accuracy** from a separate centralized MPNet reference using
2,760 training queries. None of its twelve fixed reference checks reached 99%
top-1 accuracy. A post-hoc check reached 99% **top-2 candidate recall**, meaning
the correct label is among two suggestions; no reranker has achieved 99% accuracy.
Those were centralized reference checks. In the historical SmolLM2 LoRA
benchmark, the completed seed-42 original-scoring test gives 24.6% for sample
weighting, 35.4% for equal weighting, and 34.4% for equal weighting plus clipping.
The [scoring investigation](research/accuracy-diagnosis.md)
also records a separate validation correction from 34.5% to 64.5% at version 9.

All seven configurations in the [seed-42 pilot](research/results/pilot-r2-20261103/README.md)
have now completed. The same scoring correction on final sample/equal/clipped
checkpoints reaches 52%/63%/58.5% validation accuracy. The
[accuracy recovery plan](research/accuracy-recovery-plan.md) led to the separately
measured encoder/classifier variant reported above. The original B-only contract
and results are preserved. Training data and architecture changed, so the old
and new scores are not a matched aggregation comparison.

## Open the original LoRA console

From PowerShell on the current laptop:

```powershell
Set-Location D:\Hackathons\loflwr
.\tools\start_demo.ps1
```

Open [the local dashboard](http://127.0.0.1:8510). If it is already running, open that address directly. Select a recorded run, inspect **Attempts & clients**, then use **Ask a query** with a saved checkpoint. The dashboard reads actual files and performs local CPU inference; it does not launch training. Keep the terminal open while using it.

The launcher detects `C:\Users\samee\.venvs\loflwr-cpu\Scripts\python.exe`. To choose another prepared environment:

```powershell
.\tools\start_demo.ps1 -Python C:\path\to\environment\Scripts\python.exe -Port 8510
```

The GPU training environment is separate: Ubuntu under WSL2, using `/home/sameer/.venvs/loflwr/bin/python`. See the [morning guide](docs/morning-demo.md) for exact checks and a short training run, and the [dashboard guide](docs/dashboard.md) for replay and inference behavior. The [manual three-laptop guide](docs/deployment.md) describes an untested next rehearsal; `deploy-local --layout three` still launches all three clients on one host.

## Original LoRA benchmark

| Component | Decision |
| --- | --- |
| Model | [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Task | Score 20 allowed BANKING77 intent completions, including EOS, then display a team-authored support response |
| Adapter | Shared, frozen LoRA A; train and communicate B only |
| Configuration | `q_proj` and `v_proj`, rank 8, alpha 16, dropout 0.05 |
| Aggregation | Equal coefficients among valid contributors; median-norm clipping of their B deltas |
| Benchmark | Five logical clients; matched balanced/imbalanced and equal/unequal work conditions |
| Initial heterogeneous budgets | `[2,16,8,4,2]` optimizer steps, with independent client random streams |
| Framework | Python 3.11, Flower 1.40.0; WSL2 for the tested Flower/Ray environment |
| Recovery contract | Message TTL, bounded reply collection, valid-update quorum, stale-update rejection, and fresh synchronization |
| Physical deployment target | Three laptops on a dedicated LAN/hotspot; CPU clients with at least 8 GB RAM; still requires rehearsal |

The prepared adapter has **184,320 trainable B parameters** and **737,280 raw FP32 bytes (0.703 MiB) per direction**. Flower records serialized message bytes separately. Neither quantity is whole-network traffic including TLS framing and retransmissions.

The design follows [FFA-LoRA](https://arxiv.org/abs/2403.12313). Federated LoRA and freezing A are existing methods. The intended contribution is measured behavior under constrained hardware and unequal client budgets, with explicit influence and bounded dropout handling. The earlier Qwen/77-intent/capped-weight proposal is preserved only as [reference material](research/reference-team-brief/README.md).

## Set up a clean environment

Use Python 3.11+. Follow [the tested environment instructions](requirements/README.md) for the separate Windows CPU/dashboard and WSL2 CUDA/Flower pins. The broad extras in `pyproject.toml` describe package inputs; they are not substitutes for the tested pins. The current Windows CPU environment does not include Flower.

Once the chosen environment is installed, run these commands from the repository root using its Python executable. Asset downloads require internet access:

```sh
python -m fedlora_edge.data download
python -m fedlora_edge.data prepare
python -m fedlora_edge.data verify
python -m fedlora_edge.training download
python -m fedlora_edge.training prepare
python -m fedlora_edge.training smoke --device cpu --steps 2 --output runs/smoke-new-cpu.json
```

Preparation freezes the manifests and saves the actual common-A tensors and initial B. Prepare once, then distribute the same assets to clients; do not independently regenerate A on every laptop. Model loading and inference use local assets. Keep downloads in ignored `assets/`, measurements in ignored `runs/`, and private node keys in ignored `certificates/`.

The frozen data contains 200 validation queries, 500 test queries, and separate balanced (800) and imbalanced (400) training scenarios. The [manifest record](research/reproducibility/README.md) publishes hashes and counts without publishing queries. Keep final-test access behind the declared study horizon; calibrate with validation only.

## Run checks and experiments

```sh
python -m unittest discover -s tests -v
python tools/check_docs.py
python tools/run_study.py --list
```

The standard-library suite covers state, aggregation, protocol/checkpoints, manifests, result parsing, and study configuration. Optional Torch, Flower, and Streamlit tests skip when their dependencies are absent. The suite also includes small trainer fixtures; real model optimization and Flower runtime checks are separate commands. A passing unit suite is not a task-quality result.

Use [the frozen-study guide](docs/study-runner.md) to freeze a pilot or full matrix, run a bounded invocation, resume unchanged plans, and inspect incomplete runs. The pilot includes frozen-base and centralized references, sample-weighted/equal/clipped strategies, dropout, and weak-client omission. A horizon other than twenty is labeled calibration and disables final-test access. [Project utilities](tools/README.md) lists the individual commands.

## Navigate the project

| Location | Purpose |
| --- | --- |
| `fedlora_edge/` | Data preparation, training, aggregation, protocol, Flower runtime, reports, and dashboard |
| `tests/` | Unit tests and optional dependency integration checks |
| `tools/` | Demo launcher, study runner, reports, and runtime/source checks |
| `docs/` | Practical demo, dashboard, collection, and study instructions |
| `presentation/` | Six editable slides, evidence sources, and recorded backup demo |
| `requirements/` | Tested environment pins and installation notes |
| `research/` | Design rationale, evaluation contract, implementation status, and evidence |
| `skills/` | Four project workflows preserving training and deployment invariants |

The delivered implementation is in [encoder training](fedlora_edge/encoder_head.py),
[Flower applications](fedlora_edge/encoder_flower.py),
[local inference](fedlora_edge/encoder_inference.py), and the
[query dashboard](fedlora_edge/encoder_dashboard.py). Start experiments with the
[reproduction configurations](configurations/encoder/README.md). The original
[frozen-A LoRA trainer](fedlora_edge/training.py) remains available separately.

The [research overview](research/README.md), [system design](research/system-design.md), and [evaluation contract](research/evaluation.md) explain the decisions and full acceptance criteria. The [solo seven-day plan](research/seven-day-prototype.md) and [abstract alignment](research/abstract-alignment.md) preserve the checkpoint scope and proposal changes.

## Evidence and privacy boundaries

Equal survivor coefficients address server weighting. They do not guarantee equal domain accuracy or remove participation bias. FedProx is a local drift penalty, not an influence cap. Below-quorum attempts must remain visible and retain the previous committed model.

Both models rank twenty supported labels. The encoder console can ask for more
detail on low-margin queries when the frozen policy is enabled; that is not a
general unsupported-request detector. Logits and completion scores are not
calibrated confidence. Responses are demonstration content; the application
has no bank account access and performs no banking actions.

Data locality is the privacy boundary. Adapter updates can leak information. Differential privacy, secure aggregation, and malicious-client robustness are not established. Do not commit credentials, model weights, datasets, caches, or raw run directories. Published measurements must carry their run/configuration/model/manifest identities and include unfavorable results.

Project repository: [sameerreddy789/loflwr](https://github.com/sameerreddy789/loflwr). BANKING77 is distributed under CC BY 4.0; the selected SmolLM2 model card declares Apache 2.0. See the [source register](research/sources.md) for attribution and review depth.
