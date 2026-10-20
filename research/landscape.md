# Research landscape and design choices

Primary-page review date: **20 October 2026**. See [sources.md](sources.md) for actual review depth, publication/license status, and unavailable checks. Comparisons below distinguish source findings from engineering choices; none is a measured FedLoRA-Edge training result.

## The problem has several separate constraints

Parameter efficiency reduces trainable state and communication, but the full base model and forward/backward activations still occupy memory. Heterogeneous local work changes optimization. Data imbalance changes aggregation coefficients if they use sample counts. Dropout changes which clients participate. Data locality does not establish confidentiality of updates.

The core design therefore combines a small model, common frozen-A coordinates, actual optimizer-step limits, explicit server coefficients, norm clipping, and deadline/quorum recovery. Each element addresses a different issue; one optimizer cannot replace them all.

## Small-model choices

| Candidate | Approximate model scale | Fit for this build | License/access observation |
| --- | --- | --- | --- |
| GPT-2 | 124M nominal | Simple established CPU baseline, but not instruction-tuned; would need task-specific prompting/training | MIT model-card metadata |
| DistilGPT2 | 82M nominal | Lower-memory diagnostic baseline; task quality remains an experiment | Apache 2.0 metadata |
| SmolLM2-135M-Instruct | 135M | Selected: compact, English instruction model with conventional attention targets | Apache 2.0; ungated |
| SmolLM2-360M-Instruct | 362M stored model parameters | Fallback if validation quality is inadequate and measured CPU budget permits | Apache 2.0; ungated |
| Gemma 3 270M IT | 268M stored model parameters | A small alternative requiring its own tokenizer/architecture and adapter checks | Gemma terms; model API reports manual gating |
| Qwen2.5-0.5B-Instruct | 494M stored parameters | More capacity but larger memory/time burden; not needed for the first CPU-focused build | Apache 2.0 for this particular checkpoint |
| LFM2-350M | 354M stored parameters | Different architecture and adapter-target/support checks; no assumed CPU speedup | Custom `lfm1.0` license metadata; terms require review |
| TinyLlama-1.1B-Chat | 1.10B stored parameters | Much larger frozen-weight memory; a later GPU/reference option after the small-model gates | Apache 2.0 |

These scales are not measured training-memory requirements. Publisher safetensors counts can include buffers or repeated/tied tensors, so nominal GPT-2/DistilGPT2 scales are not inferred blindly from those totals. A larger model may improve task quality, but that hypothesis does not justify starting with higher hardware demands.

The selected SmolLM2 checkpoint reports 134,515,008 stored parameters and revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. Its model card describes English strengths and factual limitations. Publisher general benchmarks do not predict our 20-intent accuracy. The bounded completion-and-template task is intentionally narrower than unrestricted banking conversation.

The old standalone Qwen file was incomplete without config/tokenizer assets and did not match the selected design. Removing it simplifies the repository; it does not establish that Qwen can never be useful. See [the inspection record](reference-team-brief/README.md).

## PEFT, memory, and communication

| Method | Published mechanism | Implication for our implementation |
| --- | --- | --- |
| LoRA | Freeze base weights and train low-rank A/B factors | Strong baseline, but independently averaging both factors need not average their products |
| QLoRA | Quantized frozen base with low-rank adaptation and memory-oriented training machinery | Valuable GPU option; validate CPU/kernel support and optimizer behavior before assuming the same benefit on weak laptops |
| FFA-LoRA | Freeze common randomly initialized A; train zero-initialized B | Selected coordinates allow linear, exact averaging of effective adapter deltas under shared A/scaling |
| LoRA-FA | Frozen-A local fine-tuning reduces adapter-related activation requirements | Relevant memory reasoning; local savings are not proof of federated stability or overall peak RAM |
| DoRA | Separate weight magnitude and direction during adaptation | Additional state and nonlinear reconstruction change the B-only aggregation contract |
| AdaLoRA | Allocate low-rank capacity adaptively across parameters | Rank/mask variation needs coordinated schema and aggregation support |
| VeRA | Shared frozen random matrices with trainable scaling vectors | Attractive smaller trainable state, but a different parameterization and aggregation experiment |
| GaLore | Project gradients to reduce optimizer-state memory during broader weight training | Not interchangeable with freezing the base and sending B; communication design needs separate treatment |
| MeZO | Zeroth-order optimization using forward evaluations rather than backpropagation | Potential activation-memory reduction; CPU work, variance, and convergence must be measured |

The FFA-LoRA abstract reports benefits in privacy-preserving FL and describes reduced factor communication. We adopt its coordinate choice without adopting a differential-privacy mechanism. Sharing only B saves a shape-dependent fraction of adapter traffic, not a universal 50% for every projection layout. In our grouped-query model, Q and V output widths differ.

The chosen B has 184,320 elements and 737,280 raw FP32 bytes. Adapter bytes do not include frozen-base storage, activation graphs, the Python tensor-state representation, serialization overhead, or the initial A distribution. Report each separately.

## Federated optimization and influence

| Method/policy | What it addresses | What it does not establish |
| --- | --- | --- |
| FedAvg | Aggregate local model updates, commonly with sample weights | A data-rich client can dominate coefficients; unequal local work can introduce drift |
| FedProx | Penalize departure from the incoming model during local training | It does not replace sample-size coefficients with equal or capped coefficients |
| SCAFFOLD | Client/server control variates correct local drift | Additional state/communication and implementation complexity; no automatic node-availability policy |
| FedNova | Normalize contributions for heterogeneous local optimization work | Not an automatic equal-influence or worst-domain-accuracy guarantee |
| q-FFL | Optimize a loss-based fairness objective across clients | Requires comparable client loss information and differs from equal coefficients |
| FedBuff | Buffered asynchronous aggregation | Needs a staleness/participation contract distinct from strict fresh-version synchronous attempts |
| Equal survivor coefficients | Make `1/m` explicit for accepted contributors | Does not ensure equal domain quality or equal participation over time |
| Median update-norm clipping | Limit update magnitude relative to current contributors | Not a Byzantine robustness or formal privacy guarantee |

The main comparison isolates sample weighting, then equal weighting, then clipping. Optional FedProx applies to B coordinates with a fixed incoming reference. In the 90% scenario, the sample-weighted rich client may receive 0.9; under equal weighting it receives one fifth if all five replies are valid, or its share among survivors if membership changes.

A repeatedly absent domain can lose influence despite equal coefficients in every completed attempt. Participation and per-domain accuracy must appear alongside global averages. With two live survivors, both coefficients are 0.5; claiming a permanent 20% cap would be false.

## Federated adapter aggregation research

| Work | Source-backed idea | Scope decision |
| --- | --- | --- |
| FLoRA | Stacking-based aggregation represents heterogeneous factor products without the naive product-of-averages mismatch | Important baseline research; growing/redistributed factors require a different implementation contract |
| FlexLoRA | Form full adapter weights and use SVD to redistribute different local ranks | Supports resource/task heterogeneity at additional reconstruction/decomposition cost |
| HetLoRA | Local rank self-pruning and sparsity-weighted aggregation | A separate heterogeneous-rank strategy, not a direct replacement for fixed-B equal averaging |
| PreLort, 2026 preprint | Prefix-nested ranks, segment-wise aggregation, and training across rank truncations | Promising heterogeneous-rank extension; outside the fixed-rank minimum viable system |
| FedLore, 2026 preprint | Share a gradient low-rank basis within a round and refresh it across rounds | Different optimization state; theoretical claims have specific assumptions and do not transfer to our protocol |
| FedFit, 2026 preprint | Shared vector banks, alternating updates/residual spectral aggregation, and quantized vectors with error feedback | Communication-oriented extension with a different reconstruction and aggregation contract |

All three 2026 arXiv IDs in the approved plan were confirmed to match their stated research identities. The accessed PreLort version was submitted 14 June 2026; FedLore and FedFit were submitted 1 October 2026. Their abstracts/metadata were reviewed; their code, proofs, and reported benchmarks were not reproduced. Their paper-reported superiority or convergence is not our result.

## Existing systems

| System | Useful starting point | Review and integration limit |
| --- | --- | --- |
| Flower / FlowerTune LLM | ClientApp/ServerApp architecture, simulation/deployment sharing, LLM examples | Simulation documentation reviewed; custom B-only contracts, response quorums, and real CPU compatibility still need implementation |
| FederatedScope-LLM | Larger research framework and LLM support | Repository metadata/license checked; no local execution or CPU benchmark |
| OpenFedLLM | Existing federated LLM experiments and algorithms | Repository metadata/license checked; does not implement our exact benchmark by inclusion alone |
| Microsoft FLUTE | Federated experimentation infrastructure | MIT repository is archived; prefer current Flower support for the chosen build |
| FedLLM-Bench | Broader heterogeneous LLM evaluation reference | Overall license is CC BY-NC 4.0 with sub-dataset-specific terms; review before copying or redistributing |

Flower documents Linux/macOS support and experimental native Windows Ray support, recommends WSL2, and explains queued, ephemeral ClientApps and soft resource reservations. One GPU worker avoids assuming that all simulated clients fit concurrently. Server evaluation needs additional resources beyond worker reservations.

Minimum-client selection settings do not prove a successful-reply quorum. The implementation needs TTL, a response-collection bound, reply validation, atomic checkpoint decisions, and fresh rejoin. See [system-design.md](system-design.md).

## Privacy and honest claims

Data locality, TLS, known-node authentication, clipping, secure aggregation, and differential privacy protect different things. Updates may reveal training information even when raw queries never traverse the network. Clipping without calibrated noise/accounting does not provide differential privacy.

Secure aggregation can hide individual updates from the server under its threat model, but then the server cannot simply inspect each client's norm as this prototype plans. Differential privacy introduces a quality/budget tradeoff and requires an accountant. Both remain extensions with separate designs and experiments.

The hackathon contribution is the measured small-model system: actual weak-client work, transparent coefficients, bounded failures, low observed traffic, and domain-level quality. See [evaluation.md](evaluation.md) for the tests required before claiming those outcomes.
