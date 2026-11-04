# Fresh supported-query collection

Human collection is **pending**. The new offline tool creates empty templates,
separate blind labeling packets and a readiness report. It does not generate
queries, contact people, train, load a model, score questions or establish a new
independent benchmark. The proposed first gate is **20 valid independently
written supported queries per intent: 400 total**. Extra valid submissions are
retained; do not select favorable rows to force exactly 400.

This implements the collection part of [fresh-evaluation.md](fresh-evaluation.md).
The original 500-query test, 300-query confirmation, 269 retained CLINC queries
and 2,279 unsupported scope queries have already been scored. They and every
training/development collection are comparison sources for deduplication only.
Their models, evaluators, locks and results remain unchanged.

## Prepare an empty collection

Run from the repository root with Python 3.11 or later; this tool uses the standard
library only. It works offline on the available single laptop.

```powershell
python tools/fresh_query_collection.py init --output runs/fresh-bank20-v1 --batch-id fresh-bank20-v1
python tools/fresh_query_collection.py ready-check --pack runs/fresh-bank20-v1
```

The initial readiness check deliberately exits **2**, with zero submitted queries
and 400 missing supported queries. Output directories must resolve under ignored
`runs/` or `assets/`. `init` and packet creation refuse an existing destination.
Keep raw queries, reviews and pseudonym records private in these ignored folders.
No real contributor records or example questions are included in the pack.

Before inviting writers, review and freeze the collection protocol, intended
population and fixed ontology. Set `protocol_reviewed` in `collection-plan.json`
to JSON `true` only after that review. `target_per_intent` is fixed at 20 for this
protocol. A different sampling design needs a separately identified protocol,
not a lowered gate after observing results.

## People and submissions

Register pseudonymous participants in `participants.csv`. Give each person one
stable `p-` identifier followed by 4–24 lowercase letters/digits, such as the
placeholder pattern `p-<random-id>`; do not enter this literal placeholder.
Keep any identity mapping outside the collection and never store names, contact
details, account numbers, credentials or identity documents in these files.
Pseudonyms are not a substitute for verifying that reviewers are different people.

The `roles` column uses `writer`, `collector`, `labeler`, `adjudicator`, joined by
`|` for multiple roles. Set `curator_id` in the plan to a registered `collector`.
The CSV attestation fields accept the literal strings `true` or `false`:

| Field | Meaning |
| --- | --- |
| `no_model_outputs` | The participant did not see model predictions/answers while writing or reviewing this collection. Required for writers, both labelers and adjudicators. |
| `no_old_queries` | The writer did not consult, copy or paraphrase old benchmark queries/errors. Required for independent human writing. |
| `independent_reviews` | The labeler worked without seeing the other labeler's judgments. Required for both labelers. |

Writers see the ontology definitions and the collection instructions only. Ask
for natural English banking-support requests, varied phrasing and ordinary
uncertainty. Do not show model answers, known benchmark questions or their
errors. Do not ask a model to write or paraphrase the evaluation questions.
Spread writing across independent contributors and report the number and
concentration of writers; the software cannot verify human independence or
representativeness.

Record every submission in `queries.csv`:

| Field | Rule |
| --- | --- |
| `query_id` | Unique `q-` plus 1–60 lowercase letters, digits, underscores or hyphens. |
| `query` | Original submitted text, kept unchanged during labeling. No personal or financial identifiers. |
| `author_id` | Registered writer pseudonym; use the collector pseudonym for nonhuman/external submissions. |
| `source_id` | Unique `src-` plus 1–60 lowercase letters, digits, underscores or hyphens. Keep the original collection source identity stable. |
| `collected_at` | ISO calendar date `YYYY-MM-DD`, not in the future. |
| `collection_batch` | The plan's `batch_id`. |
| `provenance` | `independent_human`, `external`, `model_generated` or `unknown`. |
| `provenance_detail` | Required source/provenance explanation for anything other than independent human writing. |

External and generated records remain visible as secondary records and cannot
fill the 400-query human gate. Do not relabel their provenance to meet quotas.
Never replace a hard submission with an easier one after predictions are seen.
If private material is accidentally submitted, stop distributing it and resolve
privacy handling before review; the privacy checklist is manual, not a detector.

## Independent labeling and adjudication

Once the submission batch is complete, generate packets:

```powershell
python tools/fresh_query_collection.py prepare-labels --queries runs/fresh-bank20-v1/queries.csv --output runs/fresh-bank20-v1/label-packets-v1
```

Distribute **only** `reviewer-a/` to one labeler and **only** `reviewer-b/` to the
other. Each packet contains query IDs/text, blank annotation fields, the ontology
and instructions. Author identity, provenance, intended labels, the other
reviewer's labels and model answers are absent. Reviewers choose one of the 20
labels, `ambiguous` when the text supports multiple meanings or insufficient
detail, or `unsupported` for another topic. They fill `reviewer_id`, `label` and
optional `notes`, leaving text and IDs unchanged.

After both independent reviews are complete, return their `labels.csv` files as
the pack's `labels-a.csv` and `labels-b.csv`. Only then release them, the queries
and provenance to a separate adjudicator. Every row needs an entry in
`adjudications.csv`, including agreement, secondary records and proposed
exclusions. The adjudicator must differ from the author and both labelers.

| Field | Rule |
| --- | --- |
| `query_id`, `adjudicator_id` | Existing query and registered adjudicator pseudonyms. |
| `final_label` | One of the fixed labels, `ambiguous` or `unsupported`. |
| `decision` | `retain` or `exclude`. |
| `exclusion_reason` | Empty for retained records; `duplicate`, `near_duplicate` or `privacy` for exclusions. |
| `near_duplicate_of` | Empty unless a manual semantic duplicate exists. Use `fresh:<query_id>` or `old:<dataset_id>:<src-id>` to identify it without copying old text. |
| `privacy_checked`, `near_duplicate_checked` | Literal `true` only after both manual reviews. |
| `reason` | Required recorded rationale, including confirmation of agreement or resolution of disagreement. |

`ambiguous` and `unsupported` are retained secondary labels, not reasons to
exclude difficult requests. External/generative provenance is a retained flag.
Privacy exclusions remain counted. For exact fresh duplicates, the smallest
query ID is the deterministic representative; the other copies require explicit
adjudicated exclusions. Conflicting final labels on identical normalized text
block readiness even when one copy is excluded. Manual near-duplicate decisions
must be documented before inference. A future evaluator must preserve these
decisions, never use prediction errors to revise them.

## Compare with old records without scoring them

Complete `overlap-sources.json` using existing local paths. A source entry has
`dataset_id`, `category`, `path`, `text_field` and `records_key`. Paths are absolute
or relative to the pack. For a normal manifest use `text_field: "query"` and
`records_key: "records"`; for an original BANKING77 CSV use `text_field: "text"`.
JSON lists and JSONL need no wrapping key; nested JSON object keys can use dot
notation. Each source must have nonempty readable query records.

The required categories are `training`, `development`, `bank20-test-500`,
`bank20-confirmation-300`, `external-clinc-269`, and `scope-negative-2279`.
Use more than one entry with different `dataset_id` values when a category spans
several files. Include all prior training/development sources, not just the
selected model's most recent manifests, and add any further exposed datasets.
The curator must review the inventory's completeness and then set
`overlap_inventory_reviewed` to JSON `true`. Category names are attestations;
the tool cannot prove that a supplied file is the complete historical dataset.

The tool reads supplied old text only to compute Unicode NFKC normalization,
case folding, whitespace collapse and hashes. It never invokes old evaluators
or looks at old predictions. Reports contain source hashes/counts and overlap
IDs, not old query text. Exact matches require adjudicated exclusion; semantic
paraphrases still require human review. Do not show old questions to writers
or the two blind labelers. Raw source files are never modified.

## Readiness and future evaluation

```powershell
python tools/fresh_query_collection.py ready-check --pack runs/fresh-bank20-v1
```

The default private `ready-report.json` reports all submitted rows, supported and
secondary counts, exclusions, pending records, missing labels/adjudications,
disagreements, per-intent deficits, flags, provenance/source hashes and a
collection content hash. It contains no accuracy result. Unknown fields,
duplicate IDs, missing sources or malformed imports fail closed. For CSV/JSON
imports, use `--queries`, `--participants`, `--labels-a`, `--labels-b` or
`--adjudications` to override a file; JSON uses the same string fields as CSV,
either a list of objects, `{"records": [...]}`, or JSONL.

Exit **2** means unfinished or invalid. Exit **0** and `ready_for_lock` mean the
collection meets the software's proposed gate, including at least 20 supported
rows per intent; `evaluation_ready` remains **false**. Human attestations are
not independently verified by this tool. This collection tool has no scoring
command. The [separate evaluation runner](fresh-evaluation-runner.md) can lock and
score an eligible pack using the unchanged confirmed head; the current empty
pack is refused, and combined scope evaluation remains a separate extension.

The pack's `evaluation-handoff.json` lists the remaining requirements. Before
any new inference, make a separate public, committed and pushed lock for:

1. The exact candidate classifier and scope artifacts, tokenizer/configuration,
   preprocessing, ordered label mapping and inference source hashes.
2. The combined decision policy: the chosen classifier margin rule **and** scope
   threshold, numeric boundary behavior, rejection response and all prespecified
   comparisons. Select using development evidence only, never this collection
   or already-scored confirmation/CLINC/scope outcomes.
3. This collection's protocol, raw/review hashes, final row identities, every
   retained/excluded disposition and rationale, overlap-source hashes and the
   independently reviewed readiness report. Keep text/private metadata ignored.
4. A new separate evaluator with synthetic fixture tests, a one-time inference
   start marker, fixed metrics and failure/resume rules. Do not repurpose the
   locked old evaluators by changing their input paths.

Report full supported intent accuracy and macro F1, per-intent/domain and worst
domain performance, answered-query accuracy, supported coverage, wrong answers
and declined supported queries. Report unsupported false acceptance/rejection
and ambiguous clarification/rejection separately, with all-row accounting.
Declines are not correct intent answers. Include confidence intervals, writer
diversity/concentration and denominator definitions; a small incidental set of
unsupported submissions is not a representative negative benchmark. The proposed
400 supported queries cannot establish production quality or perfect future
accuracy. Threshold changes after scoring require fresh evaluation data.

The scope detector remains experimental and the confirmed demo model remains
separate. One-laptop data preparation does not provide evidence of physical
multidevice deployment. No new human data or new model scores exist from this
workflow until those activities are actually completed and documented.

Verification uses synthetic records only:

```powershell
python -m unittest discover -s tests -p test_fresh_query_collection.py
```
