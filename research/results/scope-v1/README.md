# Locked unsupported-intent evaluation

This is a separate centralized scope detector on frozen MPNet features. It is
not deployed in the confirmed demo, and it does not replace the twenty-intent
classifier. The new test contains only unsupported requests, so it can estimate
false acceptance but cannot establish new all-intent accuracy or positive-query
coverage.

## Measured result

The lock was committed and pushed in `8e3f241723d9fdba5d1915eb1865a8ecb33b2f1b`
before scoring. The [test result](negative-test-result.json) records all 2,279
unique queries, model/data identities, per-intent scope errors and artifact hashes.

| Fixed classifier margin | False accepts without scope | False accepts with scope |
| --- | --- | --- |
| No margin rejection | 2,279/2,279 (100%) | 265/2,279 (11.63%) |
| 0.9581952095031738 | 964/2,279 (42.30%) | 136/2,279 (5.97%) |
| 1.669316291809082 | 467/2,279 (20.49%) | 76/2,279 (3.33%) |
| 2.1202011108398438 | 311/2,279 (13.65%) | 57/2,279 (2.50%) |

Every row was declared before inference; none was chosen by test performance.
The 95% positive coverage figure belongs to the **scope detector alone on reused
validation**, not the combined margin-and-scope system on independent positives.
The lower false-accept rate is therefore not evidence of 97.5% or 100% overall
accuracy. The confirmed head remains 94% on its original locked 300-query set.

The scope detector remains an experimental component. Before deployment, measure
combined coverage and classification quality on fresh supported and unsupported
queries drawn from a declared application distribution. These 2,279 test queries
are now scored evidence and cannot become another untouched test after tuning.

## Before test inference

The [confirmation plan](confirmation-plan.json) locks 40 source/artifact files,
the selected detector, the unchanged confirmed head and three original margin
thresholds. It is committed before negative-test inference. The evaluator checks
the committed plan and refuses a changed lock or a silently restarted incomplete
evaluation.

Development uses 2,760 existing supported training features plus 1,425 original
training queries across BANKING77's other 57 intents: 25 per intent. Validation
uses the reused 200 supported queries and 570 unsupported training-source queries:
10 per intent. An initial 30-per-intent proposal exceeded available deduplicated
examples for one class and was reduced uniformly before fitting or test inference.

Logistic regression candidates use `C = 0.1, 1, 10`, balanced class weights and
liblinear. Each threshold retains at least 95% of supported validation queries.
Select the fewest unsupported validation accepts, then most supported accepts,
then lower C. No test predictions select these settings.

| C | Supported validation accepted | Unsupported validation falsely accepted |
| --- | --- | --- |
| 0.1 | 190/200 | 180/570 |
| 1 | 190/200 | 147/570 |
| 10 (selected) | 190/200 | 72/570 |

The selected decision threshold is `-0.4532273876866362`. It is a decision score,
not a calibrated probability. See [development settings](development-plan.json)
and [measured development result](development-result.json).

The final source has 2,280 official test rows in the other 57 intents. Normalized
deduplication retains **2,279**: `test:1462` duplicates `test:1442`. This exclusion
was determined before inference. The initial source-count assertion failed
before output creation because it incorrectly expected 2,280 unique queries;
the audited plan and regression fixture correct that accounting. No text is
excluded based on predictions. Development excludes normalized test text.

## Reproduction boundary

```sh
python tools/run_scope_study.py develop --output runs/scope-new
```

Use a fresh output for another development experiment. The currently committed
evaluator's publication destination is explicitly `research/results/scope-v1/`;
it is a fixed evaluation protocol, not a generic way to claim a fresh test by
changing the run name. Once scored, these 2,279 queries are spent evaluation
data. Keep raw queries, features and model weights in ignored directories.

The comparison applies each fixed original margin alone and together with the
fixed scope threshold. A reduced false-accept rate can cost supported coverage;
only reused-validation supported coverage is available here. Fresh independent
supported queries are required before deploying or claiming a new full-system
accuracy or coverage result.
