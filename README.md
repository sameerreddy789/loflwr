# FedLoRA-Edge

Federated adapter fine-tuning across ordinary laptops for IIT Bombay's **Efficient AI at Scale** hackathon problem.

Each client specializes a small language model on its local banking-intent data. The server combines validated B-only adapter updates, shows each contributor's coefficient, and commits only when enough valid replies arrive before the deadline.

**Current delivery: a runnable one-laptop prototype.** Deterministic BANKING77 preparation, real CPU/GPU frozen-A training, Flower applications, checkpointed rounds, a frozen-study launcher, and a dashboard are implemented. Real CPU/GPU optimization checks, a five-client one-round GPU run, scorer parity, and isolated persistent-worker recovery checks have completed. These checks establish execution and training invariants; they do not establish convergence, improvement over the frozen base, or successful deployment across three physical laptops.

Start with the [morning demo guide](docs/morning-demo.md) on the prepared laptop. The [implementation status](research/implementation-status.md) distinguishes tested behavior from remaining gates. Measured evidence and its limits belong in the [results register](research/results/README.md).

## Open the prepared demo

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

## Selected design

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

The [research overview](research/README.md), [system design](research/system-design.md), and [evaluation contract](research/evaluation.md) explain the decisions and full acceptance criteria. The [solo seven-day plan](research/seven-day-prototype.md) and [abstract alignment](research/abstract-alignment.md) preserve the checkpoint scope and proposal changes.

## Evidence and privacy boundaries

Equal survivor coefficients address server weighting. They do not guarantee equal domain accuracy or remove participation bias. FedProx is a local drift penalty, not an influence cap. Below-quorum attempts must remain visible and retain the previous committed model.

The query interface always selects one of twenty labels, including for unrelated text. Its completion scores are not calibrated confidence. Responses are demonstration content; the application has no bank account access and performs no banking actions.

Data locality is the privacy boundary. Adapter updates can leak information. Differential privacy, secure aggregation, and malicious-client robustness are not established. Do not commit credentials, model weights, datasets, caches, or raw run directories. Published measurements must carry their run/configuration/model/manifest identities and include unfavorable results.

Project repository: [sameerreddy789/loflwr](https://github.com/sameerreddy789/loflwr). BANKING77 is distributed under CC BY 4.0; the selected SmolLM2 model card declares Apache 2.0. See the [source register](research/sources.md) for attribution and review depth.
