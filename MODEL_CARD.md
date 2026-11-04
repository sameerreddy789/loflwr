# Model card: confirmed FedLoRA-Edge banking classifier

Updated **4 November 2026**. This report describes the existing confirmed demo,
following the reporting principles summarized in the
[ML guidance](research/ml-engineering-guidelines.md). It does not announce a new
checkpoint or a new accuracy result.

## Model and intended use

The model classifies English banking-support queries into the project's
[twenty supported BANKING77 intents](research/evaluation.md), then displays a
team-authored canonical response. It is a hackathon demonstration of data-local
federated head training, explicit aggregation weights and bounded dropout
handling. It has no bank-account access and performs no banking actions.

The sentence encoder is frozen
[`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2),
revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`. The federated component is a
768-to-20 affine classification head: **15,380 trainable parameters**, or
**61,520 raw FP32 bytes per direction**, before transport overhead. This
head-only variant is separate from the preserved SmolLM2 LoRA and MPNet LoRA
experiments.

The selected checkpoint is `encoder-flower-balanced-v2`, version 500, head hash
`d85f8e14a62ebaa092c7a1e173e74e5c35a3c6dd59258c08252626f7d6cd7312`.
Its actual five-client Flower run used equal averaging. Clipping and other
weighting rules have separately identified comparison runs.

Inference uses FP32, right padding, a 384-token maximum, attention-masked mean
pooling and L2 normalization. Logits are uncalibrated scores. The implemented
predictor requests two Torch threads, but the head constructor sets its
effective thread count to one. Tested runtime pins are recorded in
[requirements](requirements/README.md).

## Data, selection and evaluation

The expanded public training set contains **2,760 queries** across five
domain-specific client partitions. The reused **200-query validation** split
was used for development and checkpoint/policy selection. The later
**300-query confirmation** contains fifteen queries per intent. The model and
all clarification thresholds were fixed in commit `7413e6b` before confirmation
was scored. These confirmation queries are now spent evaluation evidence.

The original generative LoRA study used different data sizes, scoring and
training. Its quality difference from this model is not an isolated effect of
changing aggregation. The original unfavorable results remain in the
[results register](research/results/README.md).

| Confirmation metric | Observed result |
| --- | --- |
| Full-set intent accuracy | **94.0%**, 282/300 correct, 18 errors |
| Macro F1 | **0.93914** |
| Account verification | 58/60, 96.67% |
| Card lifecycle | 56/60, 93.33% |
| Card payments | 58/60, 96.67% |
| Cash withdrawals | 57/60, 95.0% |
| Transfers, the weakest domain | 53/60, 88.33% |
| Gap between best and worst domains | 8.33 percentage points |

The strictest pre-fixed clarification threshold is **2.1202011108398438** on
the top-two FP32 logit margin. It answered 223 of the 300 queries and got 220
right: **98.65% accuracy on answered queries, at 74.33% coverage**, with three
wrong answers and 77 declines. This does not establish 99–100% accuracy.
See the [full confirmation report](research/results/encoder-v2/confirmation-result.json)
for the complete per-intent results and nominal confidence intervals.

## Scope and limitations

The independent evidence is a small, class-balanced public benchmark. It does
not establish performance on real bank traffic, all 77 BANKING77 intents,
other languages or demographic groups. Transfer requests were the weakest
measured domain; the model's intended twenty-intent boundary is not equivalent
to a reliable detector of every unsupported topic.

The separate CLINC150 evaluation measured 28/29 correct on one compatible intent,
not all twenty. The separate experimental scope detector reduced unsupported
false accepts from 311/2,279 to 57/2,279 under the strict margin. It remains
outside the confirmed demo because fresh supported-query coverage is unknown.
The [paired evaluator](docs/fresh-scope-evaluation.md) is ready for a future
reviewed collection; no such human collection has been scored.

Repeated client absence can bias participation. Equal coefficients do not
guarantee equal accuracy or all notions of fairness. All deployment and recovery
evidence currently comes from one laptop with multiple processes. Three physical
laptops, independent human-query evaluation and a presenter rehearsal remain
separate completion gates.

## Resources, privacy and changes

The frozen encoder still requires storage, RAM and computation. The small head
payload is not total memory or measured wire traffic. Frozen client features can
be cached locally, but must be regenerated if the encoder or preprocessing
changes. The optional [bucketed preparation](docs/bucketed-feature-preparation.md)
has its own dataset/recipe identity and does not replace this confirmed predictor.

The tested data boundary is participant-local preparation and head-update
exchange, with TLS in the same-host deployment checks. Updates and metadata can
still leak information. Differential privacy, secure aggregation and protection
against malicious participants have not been established.

Keep source/configuration/data/model identities with every result. Test accuracy
must not be improved by relabeling or filtering spent queries after viewing
errors. Future model changes require development-only selection and a fresh,
locked evaluation. Changing preprocessing needs new parity evidence and an
explicit identity even when weights stay fixed.

Repository code, upstream model and dataset licensing are separate. See the
[repository license](LICENSE), upstream model card, BANKING77's CC BY 4.0
attribution and the [source register](research/sources.md). No upstream paper's
reported accuracy, privacy or convergence guarantee is adopted as a project result.
