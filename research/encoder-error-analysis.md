# What still prevents 99% accuracy

On 3 November 2026, the improved federated classification head reached
**191/200 (95.5%) validation accuracy**. It matches the centralized class-balanced
logistic C10 reference's predictions on every validation query. Increasing the
head-training horizon from the selected attempt 500 to attempt 1000 did not
change those predictions. This comparison uses the already reused development
set; it is not a new test result.

## Error overlap

We compared the saved head, logistic C10, logistic C100, a 128-unit MLP, and the
nearest-example reference, all using the same frozen MPNet features and 2,760
training examples. Each method misses nine of the 200 validation queries.
Seven are wrong for **all five methods**; twelve are wrong for at least one.
An oracle choosing a correct prediction whenever any of those five has one
would reach only **193/200 (96.5%)**. This is an upper bound for choosing among
these predictions, not a trained ensemble result or a bound on all algorithms.

The shared error IDs are `train:1087`, `train:3865`, `train:3956`, `train:5924`,
`train:5937`, `train:6000`, and `train:9777`. They cover:

- Card arrival versus delivery estimate when the wording supplies little timing context.
- How to verify identity versus why verification is required.
- A verification-failure label on wording that appears affirmative.
- A declined transaction with no indication of card payment versus cash withdrawal.
- Payment clearing or posting language that overlaps pending and failed transactions.

These are signs of ambiguity and possible annotation inconsistency, not an
independently adjudicated label audit. We retain the original queries and labels.
Changing labels, dropping these examples, or writing special rules from their
IDs would invalidate the current benchmark comparison.

## Consequence for the implementation

More optimization of this linear head is unlikely to resolve the shared
representation errors: its selected predictions already match a separately
solved logistic reference. Subsequent trials adapted MPNet with frozen-A LoRA B
plus the head, starting from the stronger federated head. They used training
text only and validation for selection, and did not exceed 95.5% validation.
Those negative results remain in the [completed report](results/encoder-v2/README.md).
The transformer-layer diagnostic below tests a different, larger update space.

For a real banking assistant, asking a clarifying question on ambiguous requests
is useful. Accuracy after user clarification, top-two recall, or accuracy on
answered queries must be reported separately from automatic top-one accuracy
over every query. None is a valid substitute for a claimed 99–100% full-set score.

The original 500 test queries were already inspected. The newer 300-query
confirmation set was subsequently scored after locking the model and rules in
commit `7413e6b`: the selected Flower head achieved 94% (282/300). It is now
evaluation evidence and cannot be reused as an untouched test for further
selection. Even a perfect finite sample would not prove perfect future accuracy.

The later [representation diagnostic](results/encoder-representation/README.md)
tests a different mechanism: adapting transformer layers with a persistent
optimizer. Its first two three-epoch variants reached 96% reused-validation
accuracy, one query above the frozen-head result. This does not change the
published independent score or establish a 99–100% path.

## Evidence

The analysis reconstructs the selected head from
`runs/encoder-balanced-v2/balance-fast-long/selected/checkpoint.json` and verified
`validation.npz`; it does not fit a model. Stored reference predictions come from
`runs/encoder-reference-ablation-v1/summary.json` and
`runs/accuracy-feasibility-20261103/summary.json`.
The [overlap export](results/encoder-v2/error-overlap.json) retains source hashes,
source IDs and label comparisons, without raw queries.
