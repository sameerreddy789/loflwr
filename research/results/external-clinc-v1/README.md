# External CLINC150 transfer and unsupported-request results

**Measured 3 November 2026:** the unchanged deployed model classified **28/29
declined-card queries correctly (96.55%)**, choosing among all twenty labels.
This covers **one of twenty target intents**, not overall task accuracy. The
nominal Wilson 95% interval is **82.82–99.39%**. The original all-intent
confirmation remains **94% (282/300)**.

## Fixed policies and unsupported requests

All three margin thresholds were fixed before the original BANKING77 confirmation
and reused without tuning. Their original validation calibration targets are not
achieved external or population accuracy.

| Frozen margin threshold | Mapped class correct / answered | Mapped class coverage | Unsupported banking false accepts / 120 | Unrelated-topic false accepts / 120 |
| --- | --- | --- | --- | --- |
| 0.9581952095031738 | 26/26 | 26/29 = **89.66%** | 27/120 = **22.50%** | 23/120 = **19.17%** |
| 1.669316291809082 | 22/22 | 22/29 = **75.86%** | 11/120 = **9.17%** | 2/120 = **1.67%** |
| 2.1202011108398438 | 17/17 | 17/29 = **58.62%** | 5/120 = **4.17%** | 0/120 = **0%** |

Each policy made zero wrong answers on the mapped-class subset it answered;
the strictest answered only 17 queries and declined 12. This narrow selective
100% observation does **not** establish all-intent 100%, 99% population accuracy,
or a better model. The strictest rule also let five unsupported banking requests
through, all from `pin_change`. A large logit margin can still favor an incorrect
supported label on an unsupported query.

Unsupported banking comprised `routing`, `pin_change`, `interest_rate` and
`credit_score`; unrelated topics comprised `weather`, `play_music`, `recipe` and
`book_flight`. Each class has 30 retained queries. Without a margin gate, the
closed classifier necessarily selects a supported label for every unsupported
query. A decline means a clarification request; no follow-up dialogue or
successful resolution was measured. Do not combine these groups into one accuracy.

## Provenance and limits

- Run: `external-clinc-v1`; model: `encoder-flower-balanced-v2`, head version 500.
- Protocol/code/manifest commit, pushed before scoring: `23dba1f51422db03d29bbc378f9edb6eaea9c342`.
- External lock: `68b9f588375d58d30148eb61d799e50ab6cb446be1f4b140fb853bc72fddee46`.
- Head: `d85f8e14a62ebaa092c7a1e173e74e5c35a3c6dd59258c08252626f7d6cd7312`.
- CLINC revision: `828f8093932c8fe6ca7936c3d2e52903b1c523de`, official test split.
- 270 candidates, **269 retained** after one exact BANKING77 overlap exclusion. No outcome-based exclusions. Four retained mapped queries have token-set Jaccard >= 0.8 to BANKING77; they remain included as predeclared.
- CPU FP32, Python 3.11.15, Torch 2.8.0+cpu, Transformers 4.57.1, two Torch threads. Scoring plus model loading took **19.42 s**; zero queries truncated. Hash verification and download time are excluded.

CLINC is a separate crowdsourced collection; its source-label meaning is a proxy
for the selected BANKING77 intent. Queries were not manually relabeled. The
lexical audit is not full semantic decontamination; encoder pretraining exposure
has not been ruled out. Attribution: Stefan Larson et al., *An Evaluation Dataset
for Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP 2019,
[paper](https://aclanthology.org/D19-1131/), [dataset](https://github.com/clinc/oos-eval),
CC BY 3.0 Unported. The ignored cache retains the original README/LICENSE.

## Reproduction and verification

Read the [protocol](../../external-clinc-protocol.md), [plan](plan.json),
[aggregate result](result.json), [verification](verification.json), and
[candidate review](../../public-dataset-evaluation.md). Raw source queries,
retained rows, scores and predictions stay in ignored `assets/` and `runs/`.

Original execution sequence on the prepared Windows laptop:

```powershell
& 'C:\Users\samee\.venvs\loflwr-cpu\Scripts\python.exe' tools/evaluate_external_clinc.py prepare
# Commit and push the exact plan, protocol, evaluator and tests before inference.
& 'C:\Users\samee\.venvs\loflwr-cpu\Scripts\python.exe' tools/evaluate_external_clinc.py evaluate
```

`prepare` refuses to overwrite an existing plan or retained manifest. This
checkout already contains the immutable v1 plan. On the prepared laptop,
`evaluate` validates locked files and returns the saved result without rescoring.
An incomplete evaluation requires investigation before a retry. A clean clone
alone cannot reproduce inference: first restore/reproduce the ignored data,
model and checkpoint artifacts with the exact plan hashes. The utility does
not download model weights or retrain the selected head.

The complete WSL suite passed **169 tests, 0 failures, 0 skips**, in **18.165 s**.
Nine added fixtures cover split/label handling, duplicate rules, lock integrity,
twenty-way scoring, zero coverage and OOS denominators. All **24 original frozen
files** and **36 external-lock files** matched after scoring. Software tests are
separate from real-model accuracy.

## Presentation and next experiment

Say: “Our locked model scores 94% across twenty intents. A separate CLINC check
scores 28/29 on the one compatible intent. It also exposes unsupported banking
requests that our margin rule still answers, so scope detection is a separate
next step.”

Develop scope detection with separate development data; lock it before another
independent test. These 269 queries are now scored and cannot become untouched
again after tuning. No model, threshold or live dashboard policy changed here.
