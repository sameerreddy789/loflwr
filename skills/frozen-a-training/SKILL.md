---
name: frozen-a-training
description: Prepare or review FedLoRA-Edge model loading, common-A identity, B-only training, completion masking, and local CPU/GPU smoke checks.
---

# Frozen-A training

Use [the training contract](../../research/system-design.md) when changing model preparation, adapter serialization, optimization, or intent scoring. The selected model is SmolLM2-135M-Instruct revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`; the legacy Qwen proposal is reference material.

Keep rank 8, alpha 16, dropout 0.05 on `q_proj`/`v_proj` until a measured, documented configuration change is authorized. Centrally initialize and save A, distribute its actual tensors, and verify A/model/tokenizer/schema/scaling identities. Seed equality alone is not tensor identity.

Freeze base and A explicitly; inspect the optimizer's parameter list to confirm only B is trainable. Verify the realized B count against the 184,320-parameter estimate. Incoming global B replaces local B before each attempt, and optimizer state resets each attempt.

Mask prompts and padding while preserving intent and true EOS labels. PAD and EOS share `<|im_end|>` in the selected tokenizer; token-ID equality cannot identify padding. Determine explicit completion boundaries and use attention masks. Report truncation that removes supervision. If activation checkpointing is enabled, verify actual B gradients with the frozen base rather than assuming they propagate.

CPU uses FP32. Mixed precision requires a verified GPU smoke path. Run real optimization and compare before/after hashes: B changes, base/A do not. Record task loss, completed work, actual memory, and tensor identities. A serialization example or random delta is not training evidence.

The initial local settings are AdamW `2e-4`, zero weight decay, microbatch 1, accumulation 4, sequence length 128, gradient clipping 1. Optional proximal loss references a detached, fixed incoming B and does not alter server coefficients.

Keep caches under `assets/` and measured outputs under `runs/`. Publish source/configuration/run identities with results. Follow [evaluation.md](../../research/evaluation.md) for permitted intent scoring and comparisons.
