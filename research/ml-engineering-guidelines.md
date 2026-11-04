# ML engineering guidance applied to FedLoRA-Edge

Reviewed **4 November 2026**, against repository baseline
`ad32d6c3fabee57d9cfed0105c9ea8afeaf683f2`. This is a targeted source and
implementation review, not a new model evaluation, complete compliance audit,
or reproduction of the cited research.

The immediate priority is to measure and remove redundant inference work while
preserving the confirmed model and all evaluation boundaries. The reference is
the frozen MPNet encoder plus federated linear head: **94% (282/300)** on the
already-scored confirmation set. Head-only training is not LoRA. The original
SmolLM2 shared-A/B-only benchmark and centralized representation studies retain
their separate identities and unfavorable results.

## Rules mapped to this repository

The source guidance below informs engineering decisions; it does not override
[AGENTS.md](../AGENTS.md), the original [evaluation contract](evaluation.md), or
the separately identified [accuracy recovery plan](accuracy-recovery-plan.md).
“Present” means the indicated implementation or published evidence was inspected.
It does not mean every execution path was independently tested in this review.

| Rule and primary guidance | Present evidence | Gap or concrete next action |
| --- | --- | --- |
| **Keep a simple measured reference; test infrastructure separately from model quality.** Google Rules 2, 4 and 5 [S1]. | The confirmed head, original LoRA results, protocol tests, real Flower runs and one-laptop recovery evidence have distinct descriptions in the [README](../README.md). | Retain the confirmed head as the control. A faster function, passing unit test or successful process rejoin cannot establish higher intent accuracy. Physical-laptop recovery remains a separate gate. |
| **Split before fitting any data-dependent transformation.** scikit-learn leakage guidance [S2]. | [Encoder data preparation](../fedlora_edge/encoder_data.py) checks source identity, normalized duplicates and held-out exclusion. The fresh head/scope runners share an exposure ledger. | Confirmation, CLINC and scope test sets are spent. Do not select thresholds, labels, features or exclusions from their errors. New model selection needs separately identified development data and a fresh independent evaluation. A `Pipeline` helps contain fitting but cannot make a previously inspected test set untouched. |
| **Preserve the same feature recipe at training and serving; compare actual outputs.** Google Rules 5, 32 and 37 [S1], consistent preprocessing [S2]. | [Local encoding](../fedlora_edge/encoder_local_data.py) and the [confirmed predictor](../fedlora_edge/encoder_inference.py) use right padding, maximum length 384, attention-masked mean pooling, L2 normalization and FP32. The [paired evaluator](../tools/evaluate_fresh_scope.py) shares one feature pass. | The confirmed predictor and local encoder still contain separate tokenization/pooling paths. Any new implementation must compare token tensors, truncation counts, features, logits and decisions with its declared reference. Batch shape, ordering, precision or pooling changes can alter numerics; an unchanged checkpoint hash alone does not prove output equivalence. Preserve locked source files. |
| **Use evaluation mode and disable autograd for inference.** PyTorch tuning and autograd guidance [S3, S4]. | The confirmed predictor calls `eval()`, freezes encoder parameters and enters `inference_mode()`; the head prediction path also uses evaluation/inference mode. | This optimization is already present. Do not report adding it as a new benefit. `eval()` and gradient modes serve different purposes. Do not wrap adapter/head optimization in inference mode or reuse inference tensors in a later gradient-tracked computation without a valid conversion boundary. |
| **Cache only features whose generating model and preprocessing remain fixed.** Reusable preprocessing [S1, S2]; project-specific application. | Client-local preparations bind records, encoder identity, feature settings and archives through hashes. [Head training](../fedlora_edge/encoder_head.py) consumes frozen features. | Keep caches local to each client. Invalidate them when effective encoder weights, tokenizer or feature recipe changes. Encoder-LoRA and transformer-layer adaptation require fresh forward passes; a frozen-feature cache cannot represent a changing encoder. Locality does not establish differential privacy. |
| **Record effective CPU parallelism, not only the requested value.** PyTorch `set_num_threads` and CPU tuning [S3, S5]. | `EncoderPredictor(cpu_threads=2)` sets Torch threads, then its default `HeadTrainer` sets the count to **one**. The [fresh paired protocol](../docs/fresh-scope-evaluation.md) explicitly verifies this effective count. | Preserve that locked behavior. A separate efficiency path should set and record its intended process settings, verify `torch.get_num_threads()` after construction, and benchmark thread choices in isolated processes. More threads are not a demonstrated improvement. Avoid changing a process-wide setting concurrently with another workload. |
| **Warm up, repeat, control runtime settings and distinguish measurement scopes.** PyTorch benchmark guidance [S6]. | The [fixed-work study](results/resource-frontier-v1/README.md) separates optimizer, aggregation, evaluation and process time; it reports actual process RSS and explicitly calculated payload. | For inference changes, publish paired reference/candidate runs on the same probes, hardware and settings, with warm-up count, repetitions, order and individual observations. Separate cold load, warm query latency, throughput and process memory. A microbenchmark excludes work outside its timed region. Raw tensor bytes are not wire traffic. |
| **Seeds are necessary controls, not cross-platform reproducibility guarantees.** PyTorch reproducibility [S7]. | Head initialization preserves other RNG streams; local training uses a private generator. The project records seeds, model/configuration hashes and [tested environment pins](../requirements/README.md). | Record actual package builds, device, thread settings, batch shape and relevant backend settings for each new comparison. Keep seeds 42/43/44 paired where the study calls for them. CPU/GPU or library-version changes need their own smoke/parity evidence; do not promise bitwise reproducibility across platforms. |
| **Report intended use, evaluation conditions and subgroup limitations with each model.** Model Cards [S8]; metric design [S1]. | Published reports separate overall/per-domain quality, selective accuracy and coverage, one-intent CLINC evidence, and unsupported-query false acceptance. The dashboard labels logits as uncalibrated. | Keep selective accuracy beside its answered/total denominator and coverage. Scope can decline a prediction; it does not improve the head's top-1 label. Per-domain results do not establish demographic fairness or real-traffic performance. A future model release should carry these conditions and limitations in one concise model report. |
| **Measure observed failures without silently broadening data collection.** Google Rules 27 and 37 [S1]; project privacy boundary. | The project has explicit local ownership, hash-only public evaluation plans and a [blinded collection workflow](../docs/fresh-query-collection.md). | Use authored, unlabeled probes for engineering parity and timing. They are not fresh accuracy data. Google's suggestion to log serving features is not authorization to upload users' banking queries or client embeddings; keep diagnostics within the established data boundary. |

The existing communication experiment already shows why these distinctions
matter: four local steps between averages reduced **calculated raw head payload
by 75%**, with a **2 percentage-point reused-validation accuracy cost** in each
of three seeds. This is the published `resource-frontier-v1` result, not a new
finding here, measured network traffic, or a reason to replace the confirmed demo.

## Acceptance criteria for an efficiency change

These are this review's engineering recommendations, not publisher guarantees.

1. Create an explicitly identified implementation/benchmark outside the locked
   evaluators. Keep model/tokenizer/checkpoint, labels, scoring rules and existing
   lock bytes unchanged. Removing repeated tokenization or other repeated
   inference-only work is a candidate; changing model capacity is a separate
   experiment.
2. Use declared authored probes covering ordinary, empty/invalid, Unicode,
   mixed-length and truncated inputs. Compare accepted-input behavior, token
   tensors, order, truncation accounting, scores and decisions. Keep exact
   comparisons where the reference contract is exact. Any permitted numerical
   tolerance needs an explicit justification and cannot hide a changed decision.
3. Measure the complete intended workload with fixed probes and warmed models,
   repeated reference/candidate observations and controlled effective threads.
   Record cold loading separately. Use fresh serial processes when comparing
   process-global settings or memory; disclose host activity and timing noise.
   Functional parity on these probes is limited evidence, not accuracy or a
   universal equivalence proof.
4. Publish run/configuration/code/input hashes, actual runtime/hardware, sample
   counts, latency distribution, throughput and memory scope. Preserve neutral
   or slower results. A speed claim must identify which workload improved and
   which costs were excluded.
5. Run `python -m unittest discover -s tests` and report pass/failure/skip counts,
   plus the focused real-model functional check appropriate to the change.
   Passing fixtures does not substitute for actual-model execution, fresh model
   evaluation, Flower transport, or physical-laptop deployment evidence.

## Primary-source register and review depth

All sources below were accessed **2026-11-04** through the agent-reach skill's
Jina Reader route using `curl.exe`. Extracted pages stayed in an OS temporary
directory; no paper or full website copy is added to the repository. The
`agent-reach` and `mcporter` executables are unavailable on this host, so no CLI
doctor, Exa search or update check completed. All listed page fetches succeeded.
This targeted official-source review did not survey social discussion.

| ID | Primary source | Material actually reviewed and limits |
| --- | --- | --- |
| S1 | Google, [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) | Relevant rule text on metrics, simple baselines, independent infrastructure checks, measured failures, shared training/serving code and skew; especially Rules 2, 4, 5, 27, 29, 32 and 37. Engineering guidance, not a theorem or a banking benchmark. |
| S2 | scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) | Inconsistent preprocessing, leakage, train-only fitting/feature selection and pipeline examples. No scikit-learn experiment was run; the examples' scores are not project results. |
| S3 | PyTorch, [Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) | General gradient/data-loading guidance and CPU/OpenMP sections. Workload/hardware-specific suggestions require measurement; GPU/NUMA/compiler recommendations were not validated on this laptop. |
| S4 | PyTorch 2.8, [`inference_mode`](https://docs.pytorch.org/docs/2.8/generated/torch.autograd.grad_mode.inference_mode.html) and [Autograd mechanics](https://docs.pytorch.org/docs/2.8/notes/autograd.html) | Inference API plus the autograd sections on freezing, grad modes, inference-tensor restrictions and evaluation mode. Full autograd implementation not audited. |
| S5 | PyTorch 2.8, [`torch.set_num_threads`](https://docs.pytorch.org/docs/2.8/generated/torch.set_num_threads.html) | Complete short API description: intra-op CPU parallelism and setting it before eager/JIT/autograd work. Does not prescribe the fastest thread count for MPNet. |
| S6 | PyTorch 2.8, [Benchmark utilities](https://docs.pytorch.org/docs/2.8/benchmark_utils.html) | `Timer` rationale/parameters and `blocked_autorange`: warmups, controlled threadpool, accelerator synchronization where needed, replicates and timing overhead. `Timer` defaults to one thread; report the actual choice. Callgrind internals were not reviewed or executed. |
| S7 | PyTorch 2.8, [Reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html) | Reproducibility limits, Torch/Python/NumPy RNG guidance and deterministic-algorithm tradeoffs. No cross-platform reproducibility study performed. |
| S8 | Google Research, [Model Cards for Model Reporting](https://research.google/pubs/model-cards-for-model-reporting/) (2019) | Primary publication page, metadata and abstract: intended uses, evaluation conditions and relevant subgroup reporting. Full paper/examples and venue were not independently reviewed in this pass. |

The initially accessed PyTorch `stable` API pages resolved to **2.15**. The
versioned **2.8** API/autograd/reproducibility/benchmark pages above were then
checked to match the project's Torch 2.8.0 pins. This review does not authorize
dependency upgrades. The tuning tutorial is a current general guide, not a
version-locked compatibility guarantee.
