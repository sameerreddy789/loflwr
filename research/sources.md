# Sources, review depth, and licenses

Access date for this delivery: **20 October 2026, Asia/Calcutta**. This register separates verified page identity, published claims, engineering reasoning, and pending reproduction.

## Review method and limits

Primary arXiv pages were read through the agent-reach Jina Reader route. Titles, abstracts, submission histories, and available publication comments support the method summaries. Full proofs/appendices and experimental code were not reproduced. A venue is marked verified only where the reviewed page explicitly supplied it; absence of a verified venue here does not mean a work was never published.

[source-review.json](source-review.json) preserves the batch metadata, errors, and supplemental publisher checks. Later reader requests encountered HTTP 429. Earlier successful standalone reads of the SmolLM2 card and Flower simulation guide are recorded separately; publisher APIs and GitHub metadata/license files supplement the rate-limited checks. An unavailable batch request is not silently relabeled as a successful read.

Model API metadata verifies card license/access fields and revision identity, not trainability or accuracy. GitHub repository metadata verifies existence, archive state, and detected license; README/license review is distinguished from source-code execution. Arithmetic in the system design is our calculation, not a publisher benchmark.

## PEFT and memory references

All entries below had title/abstract metadata reviewed on the access date. Paper posting rights and implementation-code licenses are different; code licenses were not audited for these paper implementations.

| Reference | Publication status from reviewed evidence | Claims supported |
| --- | --- | --- |
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | arXiv version reviewed; venue not rechecked in this pass | Frozen base with trainable low-rank factors |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | arXiv page mentions NeurIPS submission; acceptance not independently checked here | Quantized-base adapter fine-tuning and memory-oriented machinery |
| [Improving LoRA in Privacy-preserving Federated Learning / FFA-LoRA](https://arxiv.org/abs/2403.12313) | **ICLR 2024**, explicitly stated in paper comments | Freeze common A, train B, motivation around factor aggregation and heterogeneous/DP FL |
| [LoRA-FA: Efficient and Effective Low Rank Representation Fine-tuning](https://arxiv.org/abs/2308.03303) | arXiv 2023; later venue unverified | Frozen-A local memory approach |
| [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) | **ICML 2024 oral**, page comments | Magnitude/direction decomposition |
| [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512) | **ICLR 2023**, page comments | Adaptive allocation of adapter capacity |
| [VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/abs/2310.11454) | **ICLR 2024**, page comments | Random shared matrices with trainable vectors |
| [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507) | **ICML 2024 oral**, page comments | Gradient/optimizer-state projection |
| [Fine-Tuning Language Models with Just Forward Passes / MeZO](https://arxiv.org/abs/2305.17333) | **NeurIPS 2023 oral**, page comments | Zeroth-order fine-tuning without backpropagation activation storage |

The project uses frozen-A mathematics, not paper-reported accuracy or privacy guarantees. Optional method integrations require their own implementation and comparison.

## Federated optimization

| Reference | Publication/review status | Claims supported |
| --- | --- | --- |
| [Communication-Efficient Learning of Deep Networks from Decentralized Data / FedAvg](https://arxiv.org/abs/1602.05629) | arXiv abstract/metadata; venue not rechecked | Local optimization and model averaging |
| [Federated Optimization in Heterogeneous Networks / FedProx](https://arxiv.org/abs/1812.06127) | **MLSys 2020**, page comments | Proximal local objective for heterogeneity; does not imply equal server coefficients |
| [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378) | arXiv metadata explicitly mentions ICML version | Control-variate correction of client drift |
| [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization / FedNova](https://arxiv.org/abs/2007.07481) | arXiv abstract/metadata; venue not rechecked | Normalization for heterogeneous local optimization |
| [Fair Resource Allocation in Federated Learning / q-FFL](https://arxiv.org/abs/1905.10497) | **ICLR 2020**, page comments | Loss-based fairness objective |
| [Federated Learning with Buffered Asynchronous Aggregation / FedBuff](https://arxiv.org/abs/2106.06639) | **AISTATS 2022**, page comments | Buffered asynchronous federation |

Equal coefficients, median-norm clipping, deadline/quorum rules, and the selected B-coordinate proximal ablation are project engineering choices. No convergence theorem for their combination is asserted.

## Federated adapters and 2026 preprints

| Reference | Identity/status verified | Review and use |
| --- | --- | --- |
| [FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations](https://arxiv.org/abs/2409.05976) | arXiv 2024 identity; venue unverified | Abstract describes stacking-based aggregation; code not run |
| [Federated Fine-tuning of Large Language Models under Heterogeneous Tasks and Client Resources / FlexLoRA](https://arxiv.org/abs/2402.11505) | arXiv version 2, 30 May 2024; venue unverified | Abstract describes full-size reconstruction/SVD redistribution |
| [Heterogeneous LoRA for Federated Fine-tuning of On-Device Foundation Models / HetLoRA](https://arxiv.org/abs/2401.06432) | arXiv version 2, 20 February 2024; venue unverified | Abstract describes rank self-pruning and sparsity weighting |
| [PreLort: Prefix-Nested LoRA for Federated Fine-Tuning under Rank Heterogeneity](https://arxiv.org/abs/2606.15963) | **2026 preprint**, accessed version 1 submitted 14 June 2026 | Title/abstract/history verified; prefix ranks and segment aggregation; no code/proof reproduction |
| [FedLore: Communication and Memory Efficient Federated Learning via Shared Gradient Low-Rank Projection](https://arxiv.org/abs/2610.01620) | **2026 preprint**, version 1 submitted 1 October 2026 | Shared/refreshed per-round optimization basis; theorem conditions not inherited by our system |
| [FedFit: Federated Fine-Tuning of LLMs via Vector-Bank Parameterization and Quantization](https://arxiv.org/abs/2610.01537) | **2026 preprint**, version 1 submitted 1 October 2026 | Vector banks, alternating optimization, residual spectral aggregation, and quantization; paper page identifies CC BY-SA 4.0 posting rights, not a code license |

The three recent IDs from the approved plan match the stated works. They remain comparison/extension references; publisher compression, perplexity, and convergence claims are not project measurements.

## Models and licenses

Publisher API card/license metadata was checked for every model below. Only the selected SmolLM2 model's card, pinned configuration, and tokenizer configuration received the additional detailed review described after the table. No candidate was trained or benchmarked here.

| Primary model page | Card license | Access observation |
| --- | --- | --- |
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Apache 2.0 | Ungated; selected revision matches publisher API |
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | Apache 2.0 | Ungated |
| [GPT-2](https://huggingface.co/openai-community/gpt2) | MIT | Ungated; card also read through Jina |
| [DistilGPT2](https://huggingface.co/distilbert/distilgpt2) | Apache 2.0 | Ungated |
| [Gemma 3 270M IT](https://huggingface.co/google/gemma-3-270m-it) | Gemma terms | Publisher API reports manual gating; terms not accepted by this review |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Apache 2.0 for this checkpoint | Ungated; do not generalize this license to every Qwen release |
| [LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) | Custom `lfm1.0` / `other` metadata | Ungated; full license terms remain to be reviewed |
| [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | Apache 2.0 | Ungated |

Selected-model immutable files: [config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac/config.json) and [tokenizer_config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac/tokenizer_config.json). They verify 30 layers, hidden size 576, nine attention heads, three KV heads, and PAD/EOS token ID 2. The chat template has no assistant-generation-mask block. These support adapter-size arithmetic and masking requirements; they do not establish memory consumption or task quality.

## Dataset, libraries, and existing systems

| Source | License/status checked | Review depth and supported claim |
| --- | --- | --- |
| [Original BANKING77 repository](https://github.com/PolyAI-LDN/task-specific-datasets) | **CC BY 4.0**, LICENSE read | README confirms 10,003 train, 3,080 test, 77 intents; full manifest preparation still pending |
| [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807) | Repository citation identifies NLP for ConvAI / ACL 2020 workshop | Citation read from dataset README; paper not separately reviewed in this pass |
| [Hugging Face PEFT](https://github.com/huggingface/peft) and [LoRA guide](https://huggingface.co/docs/peft/main/developer_guides/lora) | Apache 2.0 repository metadata | Library reference; guide's batch reader request was rate-limited; installed-version compatibility pending |
| [Flower framework](https://github.com/flwrlabs/flower) | Apache 2.0 repository metadata | Architecture source and selected framework; not installed/run with our training code |
| [Flower simulation guide](https://flower.ai/docs/framework/how-to-run-simulations.html) | Official documentation; standalone page read | WSL2 recommendation, experimental Windows/Ray, queued/ephemeral clients, soft resource reservations |
| [Flower Grid API](https://flower.ai/docs/framework/ref-api/flwr.serverapp.Grid.html) | Official reference retained; current batch read returned 429 | Implement and verify TTL/collection timeout behavior against the pinned runtime |
| [FlowerTune LLM](https://flower.ai/docs/examples/flowertune-llm.html) | Flower ecosystem reference; batch page returned 429 | Example integration starting point; no source-code/runtime verification in this delivery |
| [FederatedScope](https://github.com/alibaba/FederatedScope) | Apache 2.0; repository not archived | Repository existence/license only; LLM implementation not executed |
| [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM) | Apache 2.0; repository not archived | Repository existence/license only |
| [Microsoft FLUTE](https://github.com/microsoft/msrflute) | MIT; **archived** | Repository metadata only; maintenance status supports preferring current Flower |
| [FedLLM-Bench](https://github.com/rui-ye/FedLLM-Bench) | **CC BY-NC 4.0 overall**, LICENSE read; sub-datasets have additional terms | Canonical repository found via GitHub search; no benchmark reproduction or redistribution |

## Privacy references and extensions

- [Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935): abstract/metadata reviewed. Supports the warning that updates can reveal training information; it is not a measured attack on this B-only system.
- [Practical Secure Aggregation for Privacy-Preserving Machine Learning](https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/): primary reference retained; reader returned 429, so full-page and implementation review remain pending.
- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133): extension reference only, not reviewed in this pass. A DP mechanism/accountant and its utility evaluation are outside the current implementation.

The project does not implement differential privacy or secure aggregation and does not claim malicious-client robustness. Private production banking data is not part of this public benchmark delivery.

## Before incorporating third-party code or publishing results

Inspect the exact implementation/license and pin its commit. Record CPU/CUDA package locks after smoke tests. Cite BANKING77 and retain its attribution when distributing derived data. Review gated/custom/noncommercial model and benchmark terms separately. Attach configuration, seed, manifest, checkpoint, hardware, and run identity to our own measurements.
