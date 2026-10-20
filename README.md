# FedLoRA-Edge

Federated adapter fine-tuning across ordinary laptops for IIT Bombay's **Efficient AI at Scale** hackathon problem.

The project will specialize a small banking-support model using data held by separate participants. Each client trains a small adapter and shares its update. The server exposes each contributor's aggregation coefficient and continues an attempt when enough valid replies arrive before its deadline.

**Current delivery: research dossier, project workflows, and initial Python tensor-state scaffolding.** Local language-model training, Flower applications, experiment results, the dashboard, and three-laptop deployment still need implementation and verification. This repository does not yet demonstrate convergence or a live dropout recovery.

## Selected design

| Component | Decision |
| --- | --- |
| Model | [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Task | Score 20 allowed BANKING77 intent completions, then display a team-authored canonical support response |
| Adapter | Shared, frozen LoRA A; train and communicate B only |
| Configuration | `q_proj` and `v_proj`, rank 8, alpha 16, dropout 0.05 |
| Aggregation | Equal coefficients among valid contributors; median-norm clipping of their B deltas |
| Simulation | Five clients with distinct domains and unequal optimizer-step budgets |
| Deployment target | Three laptops on a dedicated LAN/hotspot; CPU clients with at least 8 GB RAM |
| Framework baseline | Python 3.11 and Flower 1.40.0; full dependency environments still need compatibility tests and locks |
| Recovery | Message TTL, collection timeout, valid-update quorum, version checks, and fresh synchronization on rejoining |

The chosen adapter contains an estimated **184,320 trainable B parameters**, equivalent to **737,280 raw FP32 bytes (0.703 MiB) per direction**. Those are arithmetic estimates. Actual process memory, transport overhead, and training time must be measured.

The design follows [FFA-LoRA](https://arxiv.org/abs/2403.12313). Federated LoRA and freezing A are existing methods. Our intended contribution is measured behavior on weak laptops: useful contributions under unequal budgets, transparent influence, bounded dropout handling, and low update traffic.

## Read the project dossier

| Document | What it answers |
| --- | --- |
| [Research overview](research/README.md) | Why this design and what the hackathon must demonstrate |
| [Landscape](research/landscape.md) | Which models, PEFT methods, optimizers, and existing systems fit the constraints |
| [System design](research/system-design.md) | How shared-A aggregation, local training, identity, and recovery work |
| [Evaluation](research/evaluation.md) | Exact domains, splits, baselines, metrics, and acceptance checks |
| [Roadmap](research/roadmap.md) | Build order, hardware gates, deployment preparation, and judging demo |
| [Sources](research/sources.md) | Primary references, review depth, publication and license status |
| [Implementation status](research/implementation-status.md) | What exists and which claims still require experiments |

## What exists now

`fedlora_edge/state.py` provides a dependency-free FP32 tensor representation, schema and payload-size helpers, serialization, state hashes, finite-value checks, and adapter-state subtraction. Standard-library tests exercise these state contracts. They are **not language-model or federated-training experiments**.

The [project skills](skills/README.md) preserve the training, federation, experiment, and deployment invariants for future work. [Project tools](tools/README.md) describe documentation and source-review utilities.

```text
loflwr/
  README.md
  AGENTS.md
  pyproject.toml
  fedlora_edge/        # Initial tensor-state implementation
  research/           # Dossier and source-review metadata
  skills/             # Four project-specific SKILL.md workflows
  tools/              # Documentation and source-review utilities
  tests/              # Dependency-free state-contract checks
```

The repository stays named `loflwr`. The former `IIT-B` wrapper has been removed; project content belongs directly in these folders. [The legacy inspection record](research/reference-team-brief/README.md) explains why the standalone Qwen checkpoint was unnecessary for the selected design.

## Run the existing checks

Use Python 3.11 or newer. For a standard Windows Python installation:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools\check_docs.py
```

With an installed `uv` runtime, the equivalent environment setup is `uv venv --python 3.11 .venv`. On Linux/WSL2, use `python3.11 -m venv .venv` and `.venv/bin/python` for the check commands.

The base checks require no third-party packages. The optional `training`, `flower`, and `dashboard` dependencies in `pyproject.toml` are planned environment inputs, **not a tested complete training stack**. Freeze separate CPU and CUDA environments after actual smoke tests. Use WSL2 for Flower/Ray simulation; Flower documents native Windows Ray support as experimental.

## Evidence and privacy boundaries

FedProx controls local drift; it does not automatically stop dataset-size-weighted aggregation from assigning 90% influence to one participant. Equal survivor coefficients address that weighting choice. They do not guarantee equal accuracy or eliminate participation bias.

Minimum-client settings alone do not handle dropout during training. The server also needs bounded reply collection, successful-update quorums, stale-update rejection, and recovery synchronization.

Data locality is the current privacy boundary. Adapter updates can leak information. Differential privacy, secure aggregation, and malicious-client robustness are extensions; none is established by this initial delivery. Canonical responses are demonstration content, not an authoritative banking service.

Keep model/data downloads under ignored `assets/`, local outputs under ignored `runs/`, and private node keys under ignored `certificates/`. Credentials, weights, datasets, and caches do not belong in source-control commits.

Project repository: [sameerreddy789/loflwr](https://github.com/sameerreddy789/loflwr). BANKING77 is distributed under CC BY 4.0; the selected SmolLM2 model card declares Apache 2.0. See the source register for third-party attribution and remaining license checks.
