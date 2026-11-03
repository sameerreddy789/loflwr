# Locked external CLINC150 evaluation protocol

Declared 3 November 2026 before downloading query data or running inference.
This file, evaluator code, source bytes, retained manifest, original checkpoint,
inference implementation and all three original margin policies are hashed in
the preparation lock. Commit that lock before the first inference.

## Question and scope

Does the unchanged deployed model transfer to independently collected declined-card
queries, and how often does its fixed clarification policy answer unsupported
requests? This is a **one-intent transfer test plus unsupported-topic stress
tests**, not an independent test of all twenty BANKING77 intents. No model update,
candidate-label restriction, threshold selection or response-template scoring.

Use the official [CLINC150 repository](https://github.com/clinc/oos-eval), revision
`828f8093932c8fe6ca7936c3d2e52903b1c523de`, `data/data_full.json`, official `test`
split. The [EMNLP-IJCNLP 2019 paper](https://aclanthology.org/D19-1131/) and README
describe crowdworker collection distinct from BANKING77. Independence from the
encoder's pretraining corpus has not been established. License: CC BY 3.0
Unported; retain the original README and LICENSE with cached source data.

## Fixed selection and mapping

Include every official test row for the following source labels, before the
fixed exact-duplicate exclusions. Expected source count: 30 per class, 270 total.

| Group | CLINC150 labels | Evaluation meaning |
| --- | --- | --- |
| Mapped intent | `card_declined` | Map to `declined_card_payment`; retain all twenty competing model labels |
| Unsupported banking | `routing`, `pin_change`, `interest_rate`, `credit_score` | None of these tasks is supported by our twenty-intent application |
| Unrelated topics | `weather`, `play_music`, `recipe`, `book_flight` | Clearly outside banking-support scope |

The single mapping follows the credit-card ontology. Different collection
instructions and granularity can still cause semantic mismatch; retain all
source-labeled examples and disclose this limitation rather than manually
relabeling or removing examples after outcomes. Do not map generic `transfer`,
`report_fraud`, `replacement_card_duration` or `expiration_date` onto a narrower
banking label. Do not treat CLINC's generic `oos_test` as automatically outside our
twenty labels: its original OOS definition is relative to CLINC's own ontology.

## Preparation before scoring

1. Verify the original 24-file confirmation lock and unchanged published policies.
2. Download pinned CLINC source, domains, README and LICENSE to ignored `assets/`;
   verify Git blob IDs against repository metadata and record SHA-256 hashes.
3. Validate schema/counts. Use official zero-based test indices as stable source IDs.
4. Normalize with original NFKC/casefold/whitespace rules. Exclude exact matches
   against **all 13,083 original BANKING77 train and test rows, all 77 labels**.
   Remove repeated selected test text by keeping its first source row. Fail on
   conflicting labels or an empty retained class. Publish every exclusion reason.
5. Count overlaps with CLINC train/validation for disclosure; these splits are
   not used to train or calibrate our model. Diagnose token-set Jaccard >= 0.8
   against BANKING77, without additional exclusions. This lexical check is not
   exhaustive semantic/pretraining decontamination.
6. Freeze retained IDs, text/labels, counts, code, model and policy hashes. Keep raw
   queries and predictions in ignored directories; publish aggregate evidence.

## Scoring and reporting

Use the original offline CPU FP32 MPNet encoder, masked-mean/L2 pooling, max length
384, and selected Flower head version 500. All twenty logits participate in
argmax. The original three margin thresholds are 0.9581952095031738,
1.669316291809082, and 2.1202011108398438. Answer iff top1 minus top2 >= threshold.
No new threshold search. Evaluate once, retain all policies and negative results.

Report mapped-class top-1 correct/count, accuracy and nominal Wilson 95% interval;
for each fixed policy report correct/wrong answers, declines, coverage and
accuracy among answered mapped-class queries. For each unsupported group and
source class report false accepts and declines with denominators. A decline is
only a clarification request, not demonstrated successful OOS identification or
resolution. Do not pool supported and unsupported examples into one accuracy.

Even 30/30 on the mapped class would not establish 99% population accuracy or
twenty-intent performance. The previously locked **94% (282/300)** confirmation
remains the all-twenty-intent estimate. A better fresh all-intent test still needs
compatible independent queries with blinded labels and a frozen evaluation plan.
