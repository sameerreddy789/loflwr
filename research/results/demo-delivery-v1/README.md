# Presentation and collection delivery

Verified 3 November 2026. This delivery changes the presentation and evaluation
handoff around the confirmed model; it does not train or score a new model.

| Check | Observed result |
| --- | --- |
| Full WSL2 suite | **225 passed, 0 failed, 0 skipped**, 30.891 seconds |
| New collection checks | 16 synthetic checks passed separately on Windows CPU; included in the 225 |
| New preset checks | Six checks, including quorum/hash validation and navigation without model assets; included in the 225 |
| Frozen file integrity | Original 24/24, CLINC 36/36, scope 40/40; overlapping locks |
| Browser | Live pending-card-payment prediction succeeded; recovery, rejoin and missed quorum were inspected; desktop 1440×1000 and mobile 390×844 views inspected |
| UI detector | No findings on the two changed dashboard files; detector does not replace browser checks |
| Slides | Six slides; three native editable charts with embedded workbooks; finalizer/package/import checks passed; all six renders reviewed |
| Recording | Untrimmed 416.3-second silent WebM; query inference followed by labeled recorded evidence; full decode passed |
| Collection pack | Created under ignored `runs/fresh-bank20-v1`; zero human queries; readiness exits 2 with the unmet gates |

The [verification record](verification.json) carries source/log/artifact hashes.
The [deck manifest](../../../presentation/source-manifest.json) preserves exact
chart values, evidence identities and speaker notes. Raw rendering and test logs
remain under ignored `runs/`.

The dashboard offers **Live banking query**, **Recorded recovery**,
**Experiment comparisons**, and **Custom run**. Its public recorded presets
verify evidence hashes and work without model/data assets. Live inference still
requires the prepared encoder/checkpoint. Recorded recovery exposes accepted
contributor counts, equal coefficients, unknown work for absent clients, and
checkpoint retention when quorum is missed. It does not launch training.

The collection tool checks fixed labels, provenance, independently supplied
annotations, adjudication, normalized duplicates, known-source overlap and the
20-per-intent target. Its checks rely on truthful human attestations and a
complete supplied source inventory. It never creates human queries or scores
the collection. See the [workflow](../../../docs/fresh-query-collection.md).

The confirmed result remains **94% (282/300)**. The fixed clarification rule
remains **98.65% on answered queries at 74.33% coverage**. Fresh human evaluation,
actual presenter rehearsal, and three physical laptops remain pending. Only one
laptop is currently available.
