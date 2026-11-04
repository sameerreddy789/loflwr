# Prepare and evaluate the next fresh collection

This runner is implemented but **no fresh human collection exists and no new
model score has been produced**. Its automated tests use synthetic fixtures only.
The existing confirmed model remains at 94% (282/300). This first runner version
evaluates that unchanged MPNet head and its strictest already-frozen margin rule.
The experimental scope detector is deferred; this is not the combined-scope test.

## Before preparing

Complete the [collection procedure](fresh-query-collection.md): at least 20 valid
independently written supported queries per intent, two blind reviews, independent
adjudication, manual privacy/near-duplicate checks and the complete overlap-source
inventory. Extra valid queries are retained. Independent provenance is a human
attestation, not something this software can establish.

Use the standard filenames created by the collection kit. Keep its inputs and
all overlap sources inside this repository's ignored `runs/` or `assets/` folders.
This runner does not accept per-file overrides. Do not put raw questions in Git.

```powershell
python tools/evaluate_fresh_bank20.py prepare --pack runs/fresh-bank20-v1 --output runs/fresh-evaluation-v1 --public research/results/fresh-evaluation-v1
```

With today's empty pack, this command exits **2** and creates no evaluation lock.
When the pack qualifies, it rechecks readiness, freezes all input bytes, saves
private exact query strings and adjudications, and creates a public hash-only
`plan.json`. Excluded query text is not duplicated into the private scored snapshot;
all excluded identities, dispositions and reasons remain accounted for.

The public plan includes source/configuration/model/asset hashes, all row
commitments, collection counts, label order, fixed inference settings and policy.
Newline-stable UTF-8 files preserve byte hashes across Git checkouts. Each private
and public output directory must be new. Preparing does not load a model.

## Publish before scoring

Review the collection and plan. Commit the public plan and the exact evaluator,
test, collection, inference and policy sources it names, then push to this
repository's `main`. Do not stage private records. Use the full publication SHA:

```powershell
python tools/evaluate_fresh_bank20.py attest-publication --output runs/fresh-evaluation-v1 --commit <40-character-commit>
```

This command verifies the declared GitHub origin, observes its current `main`
tip, and compares the committed plan and relevant source blobs byte-for-byte.
An unpushed commit, changed source, wrong remote or caller-written `published`
boolean cannot pass. It writes an immutable private publication receipt with
the observed commit and source hashes. Internet access is needed for this step.

The receipt is point-in-time publication evidence generated locally, not an
independent timestamping/notary service. **Each new scoring attempt also checks
the live GitHub origin/main reference before claiming any query or loading the
model.** The receipt's commit must remain an ancestor of that public tip, whose
plan and relevant source blobs must still match. A locally present but unpublished
commit cannot authorize scoring. If main advanced elsewhere, fetch its Git history
before scoring so both commits are available locally. Human review of publication
history remains part of the research audit.

## Score once with cached assets

Use the same prepared Python environment as the lock. A small Git publication
check needs internet access; it sends repository references, **no query data**.
Model/tokenizer downloads are disabled. The selected head, tokenizer/preprocessing, ordered labels, FP32
CPU inference, batch size 16 and two CPU threads must still match the lock.

```powershell
python tools/evaluate_fresh_bank20.py score --output runs/fresh-evaluation-v1
```

Before constructing the model, the runner exclusively claims the collection and
every retained normalized query hash in ignored `runs/fresh-evaluation-exposures/`.
This shared ledger prevents a new output path or changed collection metadata from
silently scoring the same questions again. Keep the ledger with the project;
deleting claims invalidates the once-only audit. A crash during claims or inference
leaves the attempt spent. Investigate, preserve partial evidence and disclose the
failure; there is no automatic inference retry or reset command.

Raw queries are passed to inference unchanged; NFKC/case folding is only for
deduplication/exposure tracking. Tied scores choose the lowest index in the
twenty-label mapping. Logits and the top-two subtraction are rounded to IEEE754
FP32, preserving the original evaluator's behavior at the threshold; the plan
locks this arithmetic. The fixed policy answers iff the top-two raw-logit margin
is **greater than or equal to 2.1202011108398438**. It otherwise asks for clarification.
No policy fitting, threshold selection, exclusion adjustment or training occurs.

Successful scoring saves private, hash-bound predictions and a public aggregate
`result.json`. The collection and all model/source bytes are rechecked after
inference. If reporting is interrupted after predictions are safely saved, use:

```powershell
python tools/evaluate_fresh_bank20.py report --output runs/fresh-evaluation-v1
```

Report-only verifies the local receipt, committed blobs and saved prediction
identities offline, and never constructs the model.
It may reproduce an identical public result; a different existing result is rejected.

## Read the denominators

- **Primary human supported:** only `disposition: supported`, meaning independent
  human provenance and one fixed intent. Report top-1 accuracy, macro F1 across all
  twenty labels, every domain, worst domain, domain gap and nominal Wilson intervals.
- **Fixed clarification policy:** report answered accuracy together with coverage,
  wrong answers and declined supported queries. `correct_answers_per_supported_query`
  includes declined queries in its denominator. A decline is not a correct intent.
- **Secondary:** external/generated/unknown supported rows stay in separate
  provenance groups. Unsupported answers are false accepts; ambiguous answers and
  declines are recorded separately. Neither group is folded into primary accuracy.
- **Accounting:** all submitted, scored, excluded and secondary rows remain visible,
  with exclusion-reason/provenance counts and primary writer concentration.

Nominal binomial intervals ignore author clustering and sampling bias; they do not
prove population performance. A small incidental unsupported group is not a
representative negative benchmark. Results do not change the deployed model or
establish physical-laptop recovery, production banking quality or 100% future accuracy.

Run fixture verification with:

```powershell
python -m unittest discover -s tests -p test_fresh_evaluation.py
```

The CLI returns **0** for its completed action and **2** for a blocked prerequisite
or validation error. Actual inference failure also leaves the permanent start/claim
evidence; inspect the exception and private `failure.json` without rerunning inference.
