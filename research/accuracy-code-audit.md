# Accuracy audit: training, tokenization and scoring

**Reviewed 3 November 2026.** No implementation defect explaining the low
accuracy was demonstrated by this audit. The evidence supports investigating
data coverage, unequal aggregation influence, and label-score preferences before
changing the model or optimizer. This is a bounded correctness review, not proof
that every training path is correct or that the task can reach 100% accuracy.

The review uses the actual cached tokenizer, the 400-example imbalanced training
manifests, 200 validation queries, saved results through attempt 7 of
`pilot-r2-20261103-sample`, and source inspection. Its primary record is
[accuracy-tokenizer-audit.json](results/accuracy-tokenizer-audit.json). No model
was loaded, trained or scored for that tokenizer audit, and its final-test
manifest, queries and results were not opened. A later scoring experiment is
separate evidence; see the [diagnosis](accuracy-diagnosis.md).

## Correctness checks and their limits

| Area | Finding | Evidence and scope |
| --- | --- | --- |
| Query/label source mapping | Zero mapping or client-ownership errors in the inspected manifests | Original training row identifiers and the twenty-label domain map were checked |
| Train/validation separation | Zero normalized-query overlaps | NFKC/case/whitespace normalization; does not rule out semantic paraphrases |
| Prompt/completion boundary | All twenty candidate boundaries agree with concatenated tokenization | Actual tokenizer; the native chat template agrees through EOS |
| EOS and padding | Both use token ID 2, but completion EOS remains supervised and padding is masked | Explicit completion positions and `-100` labels, rather than token-ID equality |
| Supervision | Zero supervision errors over the 600 inspected training/validation records | Prompt labels are ignored, full label and genuine EOS are retained |
| Truncation | 0/400 training and 1/200 validation queries truncate | Common prompt budget is 117 tokens, preserving the longest 11-token completion within length 128 |
| Frozen parameters | Code enables gradients only for B and checks the optimizer parameter set | Existing real CPU/GPU smoke evidence independently records changing B with unchanged base/A |
| Local restart | Incoming B is assigned before each local call and a fresh AdamW optimizer is created | Code inspection of `LocalTrainer.train` |
| Causal scoring | Logits at position t predict labels at t+1; masked rows contribute zero | Code inspection and existing scorer-parity evidence |
| Completed local work | All 35 accepted client replies through attempt 7 report nonzero B gradients | Saved real-run telemetry; this is not independent verification of remote work |

The audit record reports **18 training tests and eight data tests passed, with
zero skips**, in the prepared CPU environment. Those include small trainer
fixtures; they are not a new BANKING77 model-performance experiment. The present
write-up reviewed that saved evidence without rerunning model execution.

## Relevant implementation paths

- In [training.py](../fedlora_edge/training.py), `CompletionEncoder` builds a
  common chat prompt for all candidates, appends explicit label/EOS tokens and
  masks prompt labels. `collate` pads labels with `-100` and attention with zero.
  This avoids accidentally removing genuine EOS supervision when PAD equals EOS.
- `_add_adapter` freezes parameters except `.lora_B.`. `LocalTrainer` checks the
  realized 184,320 B parameters and saved model/tokenizer/common-A identity.
  `_new_optimizer` checks that AdamW owns precisely those B parameters.
- `train` validates common A, replaces B from the incoming state, derives a
  client-specific random stream and creates a fresh optimizer. `_train_batches`
  divides each microbatch objective by the accumulation factor, rejects missing
  or nonfinite gradients, clips the B gradient norm and applies an optimizer
  step only after a complete accumulation group. Incomplete accumulated gradients
  are discarded when the local deadline interrupts that group.
- `_scores_for_examples` uses `logits[:, :-1]` against `labels[:, 1:]`, computes
  cross entropy only at supervised positions, and returns the negative sum.
  `score` and `evaluate` choose the greatest of the same twenty completion sums.
  The [prior real-model scoring check](results/scoring-optimization.json) records
  parity across the optimized and reference implementations; it does not establish
  that the chosen scoring rule is best for intent classification.
- In [data.py](../fedlora_edge/data.py), source records retain split/row identity,
  duplicate normalization detects conflicting labels, and selection verification
  enforces counts, label ownership and disjoint query sets. This audit inspected
  the imbalanced training and validation records, not final-test content.

The optional proximal reference is detached and fixed within `_train_batches`.
The active sample run has `proximal_mu=0`, so that penalty cannot explain its
observed accuracy. Per-step gradient clipping is active; server median clipping
is not active in this sample-weighted baseline.

## Scoring preferences are plausible, but not yet a proven cause

The system prompt says to return the exact intent label but does not enumerate
the allowed labels or define them. Fine-tuning therefore has to teach the model
the desired label strings. That is the declared task design, not a tokenizer
bug. Adding a long label list would require a new prompt contract and a careful
sequence-budget check rather than an untracked edit to the running comparison.

Completion lengths including EOS vary substantially:

| Examples | Tokens including EOS |
| --- | ---: |
| `pending_transfer`, `cancel_transfer` | 4 |
| `card_arrival` | 5 |
| `pending_card_payment`, `verify_my_identity` | 6 |
| `wrong_amount_of_cash_received`, `cash_withdrawal_not_recognised` | 10 |
| `transfer_not_received_by_recipient` | 11 |

At the initial checkpoint, `cancel_transfer` accounts for 69 of 200 predictions.
At version 7, `pending_card_payment` accounts for 62 and `pending_transfer` for
52. Each true label has ten validation queries. The two ten-token cash labels
receive zero predictions at both snapshots. These facts justify measuring label
priors and length sensitivity; they do not prove that length causes every error.
A longer completion can still outrank a shorter one.

Across the twenty labels, the descriptive Pearson correlation between completion
length and prediction count is about -0.644 initially and -0.486 at version 7.
This is a small, confounded label-level association, not an independent test of
scoring bias. Label semantics, base-model preferences, sample counts and local
work all vary. A copied-checkpoint comparison of raw sums, mean-token scores and
fixed content-free centering is the appropriate next diagnostic. Report both
corrected and newly harmed predictions under each rule.

The model's validation loss falls from 5.1089 to 2.3252 through version 7, while
accuracy rises from 31.5% to 37.5%. These metrics answer different questions:
true-label token probabilities can improve without the true label becoming the
highest-scoring candidate. Neither the gap between these metrics nor a falling
loss is itself evidence of a causal-shift bug. The
[run audit](accuracy-run-audit.md) covers data exposure and server coefficients.

## Reproducibility and outstanding checks

The source files reviewed still match the frozen training/data checksums:

| File | SHA-256 |
| --- | --- |
| `fedlora_edge/training.py` | `92e07a87beb0fe06bc7862d1518d00c9c771072227bdb8404a6ce208662e40b8` |
| `fedlora_edge/data.py` | `e669eacba2802574a678c84888d1b77cae01e3108194ccdc897e7bcd63bd0b8c` |

The [exported audit](results/accuracy-tokenizer-audit.json) retains source-byte
hashes, tokenizer hashes, manifest identity, exact label counts, validation B
hashes and the regression counts. Its recorded study-plan identity is distinct
from byte-level file hashes; do not substitute one for the other.

Remaining checks include a balanced training-only fit diagnostic, the separately
identified validation scoring ablation, and completed matched federated versus
centralized results. Existing [CPU](results/smoke-cpu.json) and
[GPU](results/smoke-gpu.json) smoke checks demonstrate actual optimization and
frozen-state behavior, but do not show that the model can fit all twenty labels
or generalize well. No source, manifest, optimizer or live scorer was changed
as part of this audit.
