# Targeted efficiency paper review

Accessed **4 November 2026**. This review supports the next local efficiency
experiment. It does not add an accuracy result or replace the earlier
[accuracy review](accuracy-literature.md).

The best next experiment is **length-aware batching for participant-local frozen
MPNet feature preparation**, with the original row order restored before caching.
Keep the encoder, tokenizer, maximum length, FP32 precision, pooling and trained
head fixed. Measure it against the existing preparation path on the same approved
training records. This targets unnecessary padded computation while leaving the
federated optimization schedule intact. It is an implementation hypothesis, not a
speedup established by the papers.

The confirmed model remains frozen `sentence-transformers/all-mpnet-base-v2`
revision `e8c3b32edf5434bc2275fc9bab85f82640a19130` plus a federated affine
classifier head. Head-only training is not LoRA. The original SmolLM2/shared-A
benchmark and the MPNet B-plus-head variant keep their separate identities.
Confirmation remains the already-scored 282/300 result; confirmation, CLINC and
scope data are not inputs to this proposed efficiency experiment.

## What the primary papers support here

| Source and verified publication | Relevant finding | Decision for this project |
| --- | --- | --- |
| Sun, Li, Li and Ding, [*Improving LoRA in Privacy-preserving Federated Learning*](https://arxiv.org/abs/2403.12313), ICLR 2024; [paper](https://arxiv.org/pdf/2403.12313) | FFA-LoRA fixes the nonzero-initialized low-rank factor and updates the zero-initialized factor. It addresses the mismatch between jointly optimizing two factors locally and averaging them separately. | This motivates the existing shared frozen A, B-only design. It does not establish an efficiency gain for the confirmed classifier-head model, which has no trainable LoRA factors. The paper's reported factor-communication saving is not a universal whole-system saving; compute it from the actual A/B shapes. We do not inherit a differential-privacy guarantee. |
| McMahan et al., [*Communication-Efficient Learning of Deep Networks from Decentralized Data*](https://proceedings.mlr.press/v54/mcmahan17a.html), AISTATS 2017, PMLR 54; [paper](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) | FedAvg performs several local SGD updates before model averaging. The paper studies local epochs, minibatch size and client participation; it also reports that excessive local epochs can plateau or diverge. | More local work can reduce communication rounds, but it changes the optimization trajectory. Keep total optimizer work and the weighting rule explicit. The paper's empirical speedups are not predictions for the BANKING77 head, equal survivor weights or this laptop. |
| Wang et al., [*Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization*](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html), NeurIPS 2020; [arXiv](https://arxiv.org/abs/2007.07481) | FedNova studies how unequal numbers of local updates can make naive aggregation target a mismatched objective, and introduces normalized averaging. | Relevant if slower participants perform different amounts of completed work. It does not automatically solve label skew, dropout bias or this project's equal-weight/clipped aggregation. Adding it requires a separate method identity and matched development runs. |
| Karimireddy et al., [*SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*](https://proceedings.mlr.press/v119/karimireddy20a.html), ICML 2020, PMLR 119; [paper](https://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf) | SCAFFOLD corrects client drift with server and client control variates. The paper shows that drift can persist even with full-batch gradients and full participation. | A credible later test when trying to recover the quality lost with multiple local steps. It adds persistent state and changes training, recovery and transport. Its convergence statements are conditional mathematical results, not guarantees for the current implementation. |
| Reimers and Gurevych, [*Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*](https://aclanthology.org/D19-1410/), EMNLP-IJCNLP 2019 | Independently computed sentence embeddings can be reused for similarity and transfer tasks instead of repeatedly running a pairwise cross-encoder. | This supports the architectural rationale for reusable sentence features. It does not itself validate the particular MPNet checkpoint, BANKING77 head, caching recipe or federated accuracy. Caching helps training; a new interactive query still needs an encoder forward pass. |
| Song et al., [*MPNet: Masked and Permuted Pre-training for Language Understanding*](https://arxiv.org/abs/2004.09297), NeurIPS 2020; [paper](https://arxiv.org/pdf/2004.09297) | MPNet combines masked and permuted language-modeling ideas for pretrained representations. | This is the backbone paper. The actual sentence encoder has additional sentence-pair contrastive training, documented in its model card below. Do not attribute all-mpnet-base-v2's full training recipe or our task performance solely to this paper. |
| Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), NIPS 2017; [paper](https://arxiv.org/pdf/1706.03762) | Table 1 lists self-attention complexity as `O(n² d)` in sequence length `n` and representation dimension `d`. | Reducing each batch's padded sequence length reduces a relevant attention-work proxy. Projection and feed-forward work, tokenizer overhead, CPU kernels and memory movement also matter. The paper does not imply that a 30% proxy reduction produces a 30% wall-time reduction. |

The algebra behind common frozen A is narrow and useful: with common scale `s`,
`sum_i w_i (s B_i A) = s (sum_i w_i B_i) A`. It says that averaging B and
averaging these effective adapter matrices agree. It says nothing by itself
about the accuracy of the resulting predictor, fairness, privacy or convergence.

## The direct implementation evidence

The [model card at the project's pinned revision](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/blob/e8c3b32edf5434bc2275fc9bab85f82640a19130/README.md)
describes a 768-dimensional sentence embedding, attention-mask-aware mean pooling,
L2 normalization and a default maximum of 384 word pieces. It states that
`microsoft/mpnet-base` was further trained on sentence pairs with a contrastive
objective. These are more direct sources for the deployed feature recipe than
the MPNet pretraining paper alone.

[Transformers 4.57.1 padding documentation](https://huggingface.co/docs/transformers/v4.57.1/en/pad_truncation)
states that `padding=True` pads to the longest sequence in the batch. The existing
[participant-local encoder](../fedlora_edge/encoder_local_data.py) batches in
record order and makes two tokenizer calls: one for original lengths and another
for truncated, padded tensors. Variable-length records can therefore cause short
examples to run with the longest example's tensor width.

The [sentence-transformers implementation at commit
3eeec5b5a912809e1034847f9d6d3d4a2d0a5f62](https://github.com/huggingface/sentence-transformers/blob/3eeec5b5a912809e1034847f9d6d3d4a2d0a5f62/sentence_transformers/sentence_transformer/model.py#L948)
sorts inputs by its `_input_length` helper before batching (lines 948–955), and
restores the original order afterward (line 995). This is implementation
precedent, not a paper claim or a dependency upgrade recommendation. Using exact
token lengths in this repository would be our own implementation choice; the
review does not claim that the upstream helper measures exact token lengths.

For a batch with `b` examples and padded width `L`, record `b × L` padded token
positions and `b × L²` attention-score positions. Sum them across batches and
report their units explicitly. These are deterministic work proxies, not measured
FLOPs, memory usage, CPU utilization or elapsed time. Preserve the count of
queries that exceed 384 tokens before truncation. When reusing tokenized inputs,
retain the tokenizer's truncation and special-token behavior; slicing an
untruncated token list is not automatically equivalent.

## Why this comes before another optimization method

The [five-participant preparation evidence](results/encoder-local-v1/five.json)
already records tens of seconds per participant for encoding, asset verification
and model loading together. Those combined historical timings are not a clean
encoding baseline. The affine head has `(768 + 1) × 20 = 15,380` FP32 parameters,
or **61,520 raw bytes**. That makes local encoding a useful first target on the
available laptop, while repeated round serialization can still be important.

The [completed fixed-work study](results/resource-frontier-v1/README.md) is a
concrete counterexample to assuming communication reduction is free: at 500
optimizer steps per client, four steps between averages used 75% less calculated
head payload, but scored 93.5% instead of 95.5% on reused validation across seeds
42/43/44. Those are cached-feature numerical-client results, with no measured
wire bytes or new test accuracy. Preserve that tradeoff rather than switching the
confirmed schedule based on a general paper recommendation.

Lossless transport compression remains a separate systems option.
[RFC 1951](https://www.rfc-editor.org/rfc/rfc1951) specifies lossless DEFLATE and
explicitly notes that no lossless codec compresses every possible input. This is
a format standard, not an ML paper. A transport experiment should compare the
existing serialized message, raw FP32 payload and compressed representations;
check byte-for-byte decoded tensors and hashes; and measure codec CPU time,
actual serialized bytes and end-to-end transfer time. Compressing JSON overhead
must not be presented as reducing the mathematical model's parameter count.
Payload reduction on one host cannot establish a physical-network latency gain.

## Minimal empirical gate

1. Use a new efficiency run identity and approved training records, explicitly
   reused development validation, or authored probes. Retain the original path
   as the comparison. Do not infer the scored held-out queries or change their
   locked evaluators, models, thresholds or data. Reading existing model/source
   hashes for identity checks does not create a new accuracy observation.
2. Hold model/tokenizer assets, record order, batch size, FP32, CPU threads,
   maximum length and pooling fixed. Restore output order before feature/label
   alignment. Record input, source, configuration and output hashes.
3. Test mixed lengths, ties, a short final batch and long truncation-boundary
   probes. Compare token IDs/masks, truncation counts, feature norms and maximum
   feature error; also check the fixed head's logits and predictions on the
   development/probe inputs used for parity. Rebatching can change floating-point
   rounding, so do not promise bitwise equality without measuring it.
4. Time tokenization, encoder work and full preparation separately. Include
   sorting/preparation overhead in the end-to-end comparison. Alternate run order
   with repeated serial measurements and no competing benchmark work; report
   individual observations, summary statistics and sampled process RSS honestly.
5. Claim an efficiency gain only if measured elapsed time improves under the
   stated parity gate. A proxy reduction alone is not sufficient. This measures
   local batch preparation, not faster single-query inference, faster head SGD,
   Flower scaling or physical-laptop recovery.

## Source access and limits

Primary sources were accessed with the agent-reach skill's Jina Reader route;
GitHub source/metadata used its `gh` route. The `agent-reach` and `mcporter`
executables were unavailable, so backend diagnostics, Exa search and the optional
update check could not run. No installation or environment-pin change was made.

PMLR and ACL metadata/abstract pages were readable. Jina PDF text was readable for
FFA-LoRA, FedAvg, SCAFFOLD, MPNet and Attention; the targeted sections above were
checked, not the proofs independently reproduced. FedNova's venue and abstract
were checked against the NeurIPS proceedings and the author's repository. MPNet,
FFA-LoRA and Attention venues were also explicit in the PDF text. The pinned
model card and versioned padding documentation were readable.

The arXiv HTML and ar5iv HTML attempts for FFA-LoRA and Attention returned SVG
image descriptions instead of paper text, so those responses were not used as
evidence. An OpenReview request returned a browser challenge. The unversioned
Hugging Face padding page pointed to a missing version and was replaced by the
readable 4.57.1 page. The initial sentence-transformers `master`/old filename
lookups failed; repository metadata and the pinned tree resolved the current
`main` path cited above. An initial arXiv candidate `2401.09670` was verified to
be *DistServe* and discarded rather than cited as FFA-LoRA.

Temporary retrieved text is outside the repository. This file contains the
source links, supported claims and access status; no raw paper corpus or new
benchmark metrics are added here.

The subsequent [measured implementation report](results/encoder-bucketing-v1/README.md)
records the six paired timings and complete migration check. Median encoding
times improved, with two slower repetitions. The full check preserved all
top-1 predictions and passed actual head-update comparisons, but one stored-cache
clarification decision changed. Its strict compatibility result remains failed;
the new recipe is experimental and the confirmed preprocessing remains intact.
