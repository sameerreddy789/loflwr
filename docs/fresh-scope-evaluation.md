# Evaluate the fixed classifier and scope detector together

The paired evaluator is implemented; **no fresh human data or new accuracy
result exists**. It compares the unchanged confirmed frozen-MPNet federated head
under two fixed rules in one evaluation. It does not deploy the scope detector.
The confirmed top-1 result remains **94% (282/300)**.

Use this runner instead of the [head-only runner](fresh-evaluation-runner.md)
when the next collection will evaluate scope. Decide before model inference.
The two runners share a query-exposure ledger: scoring with one makes those
queries spent for both. Do not score the baseline first and then fit or select
a combined rule on the same answers.

## Prepare one reviewed collection

Follow the [blinded collection workflow](fresh-query-collection.md), including
two blind reviews, independent adjudication, privacy and near-duplicate checks,
attested independence and the complete overlap-source inventory. This comparison
requires at least:

- **400 independent-human supported queries:** at least 20 per fixed intent.
- **200 independent-human unsupported queries:** requests outside the twenty
  supported intents, reviewed under the same collection protocol.

The unsupported minimum is an exploratory protocol choice, not a statistical
power calculation. Collect a plausible range of unsupported banking requests and
other topics, and document recruitment/sampling in the private collection notes.
A convenience sample does not establish real traffic prevalence. Ambiguous,
external, model-generated, unknown-provenance and excluded rows cannot fill the
minimum. Retain extra valid rows; report all dispositions and provenance groups.
Independence and sampling quality remain human attestations, not software proof.

With the currently empty collection, preparation exits **2**, creates no lock,
and performs no inference:

```powershell
python tools/evaluate_fresh_scope.py prepare --pack runs/fresh-bank20-v1 --output runs/fresh-scope-evaluation-v1 --public research/results/fresh-scope-evaluation-v1
```

When collection is complete, preparation freezes exact private records and
adjudications, all input/source/model hashes, cohort counts and observed mixture,
the environment, and both decision rules. Raw records/predictions stay under
ignored `runs/`; only the hash-only plan and aggregate result belong in Git.
Both private and public output directories must be new.

## Publish and score both policies once

Commit and push the exact public plan and locked sources to this repository's
`main`, then attest its full 40-character commit. Do not stage collection files.

```powershell
python tools/evaluate_fresh_scope.py attest-publication --output runs/fresh-scope-evaluation-v1 --commit <40-character-publication-commit>
python tools/evaluate_fresh_scope.py score --output runs/fresh-scope-evaluation-v1
```

Attestation checks GitHub `origin/main` and exact committed bytes. Scoring checks
live publication again before claiming any query or constructing a model. The
publication check needs internet; it sends Git references, no queries. Cached
model/tokenizer inference stays offline with downloads disabled. Keep the same
Python, package versions, NumPy build/backend and thread environment used when
preparing. If remote history advanced, fetch it before the publication check.

The fixed policies are:

1. **Baseline:** answer when the top-two logit margin is at least
   `2.1202011108398438`. Logits and subtraction use FP32; lowest-label-index ties
   reproduce the frozen evaluator.
2. **Combined:** answer only when baseline answers **and** the frozen scope
   score is at least `-0.4532273876866362`. The scope score uses FP32 features
   multiplied by the saved FP64 coefficients, plus the FP64 intercept. Keep its
   threshold and coefficients in FP64.

Both policies reuse a single MPNet feature pass: right padding, maximum length
384, masked mean pooling and L2 normalization in FP32, batch size 16 on CPU.
The unchanged predictor requests two Torch threads, then its `HeadTrainer`
sets the effective count to **one**. The runner verifies and records this behavior.
Historical scope scoring used batch size 8; this new fixed batch-16 comparison
does not recompute or replace that old result. It never trains either model.

The saved scope model is bound by its original lock
`d5944ea9ec76a14c67870421ba782425d65245e3ed35207ff83660ba3ac0c7aa`.
The unchanged head retains its original confirmation identity. Preparation and
scoring verify those saved bytes without using old test questions for tuning.

One joint claim covers the collection and every retained normalized query in
`runs/fresh-evaluation-exposures/`. Preserve the ledger. A crash, partial claim
or inference failure makes the attempt spent; disclose it and investigate rather
than deleting claims. There is no retry/reset command. If predictions were saved
but reporting was interrupted, report-only performs no inference:

```powershell
python tools/evaluate_fresh_scope.py report --output runs/fresh-scope-evaluation-v1
```

## Read the paired result

For the supported human cohort, each arm reports top-1 accuracy, macro F1, all
domains, worst-domain accuracy and selective accuracy with coverage, wrong answers
and declines. **Top-1 predictions are identical for both arms.** Scope can remove
an answer; it cannot make that head's top-1 classification more accurate.

Paired counts distinguish **correct answers lost** from **wrong answers removed**,
with the full supported denominator and coverage change.
The supported-domain breakdown also shows each domain's answered/declined counts,
coverage, wrong answers and correct answers lost, so uneven rejection costs remain
visible. For unsupported humans,
report each arm's false-accept rate and how many false accepts scope removed.
Ambiguous answers/declines and non-human provenance groups remain separate.
Human unsupported rows also appear in their provenance breakdown for accounting;
they are the same observations and must not be counted twice.

The result reports actual cohort composition, writer concentration, exclusions
and nominal Wilson intervals. It does not combine rejected unsupported requests
with intent predictions into a new overall accuracy. Nominal binomial intervals
ignore writer clustering and sampling bias. No score, 99–100% guarantee or
deployment approval follows from passing this runner's tests.

```powershell
python -m unittest discover -s tests -p test_fresh_scope_evaluation.py
```

These tests use synthetic fixtures solely to verify lifecycle, exact thresholds,
paired denominators, rejection and one-time scoring behavior.
