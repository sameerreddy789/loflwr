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

To open the upgraded query demo on the prepared laptop:

```powershell
.\tools\start_encoder_demo.ps1 -PolicyFile research/results/encoder-v2/final-policy.json
```

Open <http://127.0.0.1:8511>. The [encoder demo guide](docs/encoder-demo.md)
explains live classification, recorded evidence and supported intents.

Federated adapter fine-tuning across ordinary laptops for IIT Bombay's **Efficient AI at Scale** hackathon problem.

The original SmolLM2 experiment trains B-only LoRA adapters. The accuracy-recovery variant trains a small classifier head on frozen MPNet features; a separate MPNet B-plus-head trial is also implemented. The server validates updates, exposes contribution weights and commits only when enough valid replies arrive before the deadline. Head-only training is not called LoRA.

**Current delivery: a runnable one-laptop prototype.** Deterministic BANKING77 preparation, real CPU/GPU frozen-A training, Flower applications, checkpointed rounds, a frozen-study launcher, and a dashboard are implemented. Actual process interruption, below-quorum retention, rejoining, and fresh offline deployment have been exercised. The full WSL environment passes 157 tests with no skips. See the measured results for learning comparisons; successful execution alone does not establish convergence or deployment across three physical laptops.

Start with the [morning demo guide](docs/morning-demo.md) on the prepared laptop. The [implementation status](research/implementation-status.md) distinguishes tested behavior from remaining gates. Measured evidence and its limits belong in the [results register](research/results/README.md).

## Test statistics and measured results

Verified on **3 November 2026** using the [tested environments](requirements/README.md).
The [handoff record](research/results/encoder-v2/handoff-verification.json)
contains the run/checkpoint identities and hashes of the original check logs.

| Verification | Observed result | Scope |
| --- | --- | --- |
| Full automated suite, Ubuntu WSL2 | **157 passed, 0 failed, 0 skipped** in 19.297 s | State, aggregation, protocol, manifests, trainers, Flower integration fixtures, dashboards, and evaluation locks |
| Encoder dashboard checks, Windows CPU | **7 passed, 0 failed, 0 skipped** in 8.033 s | A separately run subset of the 157 tests, not 7 additional unique tests |
| Locked artifact integrity | **24/24 file hashes matched** | Model/tokenizer assets, selected checkpoint, sources, and evaluation configuration |
| Live CPU query check | Passed; a fresh pending-card-payment query returned `pending_card_payment` | Functional inference check, not an accuracy benchmark |
| Actual five-client Flower training | **500/500 attempts committed** with all five contributors | `encoder-flower-balanced-v2`, selected version 500 |
| Trained-model dropout and recovery | **3/3 attempts committed**; 5 → 4 → 5 accepted clients | One laptop; a client process was killed during attempt 2 and rejoined with fresh state in attempt 3 |

Timings are observations from this laptop, not controlled performance comparisons.
The full suite includes the dashboard tests. Missing optional dependencies can
produce skips elsewhere; a smaller passing suite does not reproduce this result.

| Model-quality measure | Result | Evaluation boundary |
| --- | --- | --- |
| Independent confirmation top-1 accuracy | **94.0% — 282/300 correct** | All 300 queries; checkpoint and scoring frozen before access |
| Confirmation macro F1 / worst-domain accuracy | **0.93914 / 88.33%** | Transfers was the weakest domain |
| Validation accuracy | **95.5% — 191/200 correct** | Reused development set that guided model selection |
| Strictest frozen clarification policy | **98.65% — 220/223 answered correctly** | **74.33% coverage**; 77/300 queries declined |

See [all confirmation results](research/results/encoder-v2/confirmation-result.json)
and the [complete experiment report](research/results/encoder-v2/README.md),
including negative adapter results and matched imbalance/omission comparisons.
Passing software tests is separate from predicting banking intents correctly.

## Future plans

These milestones are planned work, not completed results. Their order prioritizes
the weak-device/federation challenge alongside task quality.

| Priority | Next milestone | Completion evidence |
| --- | --- | --- |
| 1 | Rehearse the prepared demo and record a fallback | Fresh offline launch, live queries, clearly labeled event replay, and a reproducible launch checklist |
| 2 | Prepare encoder features independently on each client | Local query-to-feature preparation using the same pinned encoder; the server receives only model updates |
| 3 | Test three physical laptops | Measured CPU/RAM budgets, authenticated TLS, bounded dropout/rejoin, and fresh startup without internet |
| 4 | Complete matched experiments across seeds 42/43/44 | Equal/sample/clipped aggregation under heterogeneous budgets, imbalance, dropout, and paired weak-client omission; report variability and unfavorable outcomes |
| 5 | Improve accuracy with new development data and a new independent test | Train-only improvements selected on validation; freeze the next model and policies before scoring new held-out queries |
| 6 | Profile efficiency and evaluate later extensions | End-to-end memory, traffic and latency measurements; separate evidence before claiming benefits from adapters, secure aggregation, or differential privacy |

The [detailed future plan](docs/future-plans.md) gives dependencies and acceptance
criteria. The current 300-query confirmation set is now evaluation evidence;
it cannot serve as an untouched test for the next model-selection cycle.
**99–100% remains an unachieved target, not a promised milestone.**

## Accuracy research and original results

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
| `docs/` | Practical demo, dashboard, and study instructions |
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
