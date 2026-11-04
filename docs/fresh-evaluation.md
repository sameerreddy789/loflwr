# Collect a fresh evaluation set

The [collection kit](fresh-query-collection.md) and
[separate evaluation runner](fresh-evaluation-runner.md) are implemented. Real
evaluation is still blocked: there are zero fresh human submissions. The first
runner supports the unchanged confirmed classifier and frozen clarification
rule; it does not evaluate the experimental combined scope detector.

The original 500 test queries and the later 300 confirmation queries have already
been scored. Reusing them can check regressions, but cannot provide a new
untouched estimate after further model selection. Do not replace the published
94% result with a score selected from those sets.

The [public-dataset review](../research/public-dataset-evaluation.md) found
CLINC150 suitable for a limited transfer/scope check. Its
[completed evaluation](../research/results/external-clinc-v1/README.md) covers
one compatible intent (28/29 correct) and eight unsupported classes. These
269 queries are now scored too; they are not a fresh twenty-intent collection.
None of the reviewed sources supplied a verified directly compatible full-task
test. The collection procedure below remains the route to that missing evidence.

## Collection procedure

1. Fix the twenty-intent scope, target population, collection instructions,
   sample count, and ambiguity policy before collecting answers. Use English
   banking-support requests, without real names, account numbers or card details.
2. Ask independent contributors to write natural queries without seeing model
   outputs or old benchmark errors. Avoid paraphrasing the known test questions.
   Record collection source and date. Model-generated questions can be a separate
   development stress set; do not call them independent human evaluation.
3. Have two annotators label each question independently using the intent
   definitions. Resolve disagreement before inference, with a third adjudicator
   or a predeclared rule. Labelers must not see the candidate's predictions.
4. Mark genuinely underspecified or out-of-scope requests using the declared
   policy. Publish their counts and evaluate appropriate clarification/rejection
   behavior separately. Do not remove difficult questions after seeing errors.
5. Deduplicate against all existing train/development/test records and within
   the new set using normalized text plus a manual near-duplicate review. Preserve
   source IDs, provenance, exclusion reasons, and hashes of the final manifest.
6. Keep this final collection separate from newly collected training/development
   data. Select the candidate and any thresholds using development data only.
   Freeze checkpoint, tokenizer, preprocessing, label mapping, scoring, and
   stopping/selection rules before scoring the final set once.

## Suggested initial scope

For a manageable pilot, target **400 independently written, valid in-scope
queries: 20 per intent**. This is a suggested new protocol, not existing data or
a guarantee that 400 examples establish production quality. Record ambiguity
and exclusions separately rather than hiding them in the denominator.

With 400 scored questions, 99% means at least 396 correct; 100% means 400 correct
on that finite set. Neither proves perfect accuracy on future users. Report a
confidence interval, macro F1, each domain's accuracy, worst-domain accuracy, and
all-query performance. Selective answering must also report coverage and the
number of wrong and declined answers.

## Review template

Keep the collection in ignored `assets/`; do not put raw contributor queries or
personal information in Git. A JSONL record can use this schema:

```json
{"source_id":"fresh:0001","query":"<independently written query>","label":"<one of the twenty allowed intents>","collection_batch":"<batch ID>","annotator_labels":["<label 1>","<label 2>"],"adjudication":"<agreement or resolution>","status":"in_scope"}
```

This template is not yet accepted by the existing fixed BANKING77 confirmation
utility. Use the collection kit's stricter reviewed CSV schema and the separate
runner to freeze a qualifying collection; a new dataset identity and published
evaluation lock are still required. Changing a file path does not establish independence.

No contributors have been contacted and no new human evaluation queries have
been collected by this project yet. The [presentation guide](judging-presentation.md)
uses the existing independently measured model until a new result is available.
