# Remaining work and completion gates

This plan tracks the work authorized after the confirmed encoder demo. Only
one physical laptop is available. The stable query model remains frozen MPNet
plus its Flower-trained linear head: **94% independent confirmation (282/300)**
and **95.5% reused validation (191/200)**. The head-only and original SmolLM2
LoRA experiments keep separate identities and results.

## Complete the feasible work on this laptop

| Work | Current evidence or implementation | Gate |
| --- | --- | --- |
| Demo readiness | [Read-only preflight](demo-preflight.md) implemented and run on 4 November; five checks passed | Python/dependencies, evidence, local port and cached functional inference; separate from OS-level offline or human rehearsal |
| Fixed-work resource comparison | [Nine serial CPU cells](../research/results/resource-frontier-v1/README.md), seeds 42/43/44, 500 optimizer steps per client in every cell | Four local steps per average cut estimated raw payload by 75%; final reused-validation accuracy fell 2 points in each seed |
| Participant-local feature preparation | `encoder-local-five-v1` and `encoder-local-three-v1` completed in isolated processes; own records only, no external network | Passed same-host parity: maximum feature difference below `1.05e-7`, zero changed predictions on 2,960 train/validation rows per layout |
| Metadata-only registration | Common plan, participant registration validation, metadata assembly, role-specific feature loaders implemented | Invalid identities/ownership/duplicates rejected; no client query or feature is needed to assemble metadata |
| Portable role configuration and app bundle | `configure_encoder_laptop.py`, `build_encoder_app.py`, and [laptop guide](encoder-laptop-rehearsal.md) | `encoder-portable-smoke-v1` configured four real prepared roles, preserved selected head hash, and built an encoder Flower FAB; fixture node IDs do not prove transport |
| Isolated Flower deployment | **Passed**: TLS, registered known nodes, role-local files/mounts, real process interruption and rejoining | [Three/five-client report](../research/results/remaining-work-v1/README.md): accepted sequences `[3,2,3,1,3]` and `[5,4,5,2,5]`; below-quorum retention and expected incoming head hashes verified |
| Offline runtime | **Passed**: fresh three-client startup with cached dependencies and runtime downloads disabled | All three contributed in a loopback-only network namespace; external connection failed. This remains separate from a physical LAN run |
| Matched repeated studies | `encoder-matrix-v1`: **33/33 completed cells**, seeds 42/43/44, 500 attempts | [All results](../research/results/encoder-matrix-v1/README.md) include selected/final metrics, variability, omission benefit of 18 points per seed, and unavailability costs of 0/0.5/0 points |
| Unsupported-topic detection | `scope-v1`, locked and pushed in `8e3f241` before scoring 2,279 new negative queries | [Completed test](../research/results/scope-v1/README.md): strictest margin false accepts 311 without scope versus 57 with scope; positive scope-only coverage remains reused validation 190/200 |
| Final checks and delivery | Tests, frozen evaluation hashes, Markdown links, result summaries, README, commit/push | State exact test count/skip count, verified remote commit, measured run/config identities, and unresolved gates |

The matrix performs sequential mathematical federation on cached features. It
does not add 33 Flower deployment experiments. Expanded 2,760-query training
and the original 400-query 90%-imbalance condition remain separately labeled.
Its weak-client omission study must report paired differences even if they are
small or negative. Equal aggregation coefficients do not ensure equal accuracy.

The scope classifier is a separate centralized development experiment. A
negative-only final test can measure false acceptance of unsupported requests;
it cannot establish fresh twenty-intent accuracy or positive-query coverage.
Do not silently install it into the already-confirmed demo. Previously scored
CLINC and BANKING77 confirmation sets remain sealed against further selection.

## Finish when two more laptops are available

1. Follow the [encoder laptop guide](encoder-laptop-rehearsal.md), provision each
   machine's own public records, and measure local preparation and memory.
2. Establish a dedicated reachable LAN, verify TLS and three known identities,
   and calibrate actual CPU work before declaring budgets/deadlines.
3. Run all three physical clients, record an actual process/network interruption,
   confirm quorum-two aggregation, miss quorum deliberately, and verify fresh
   synchronization after rejoining.
4. Start all roles from fresh processes with internet disconnected. Record
   per-machine logs, measured network traffic, timings, and whole-process memory.

The host owns two logical roles; the training server's application files should
be limited to validation and metadata even though the host also operates its
own client. Production private-data ownership, hostile clients, and physical
machine isolation are not established by the public-benchmark provisioner.

## Prepared presentation and collection handoff

The dashboard now has live query, recorded recovery, experiment comparison and
custom-run presets. The [six-slide deck and recorded fallback](../presentation/README.md)
are ready. The [collection kit](fresh-query-collection.md) creates empty blinded
packets and checks provenance, duplicate overlap and independent adjudication.
Its 400-query human gate remains unmet; no fresh queries or new accuracy result
are implied by having software ready.

## Complete the human rehearsal and evaluation

Use the [prepared demo](encoder-demo.md) and [presentation guide](judging-presentation.md)
to rehearse with the actual operator. Show the task, independent accuracy,
recorded or live federation with an explicit label, the bounded failure behavior,
and a new query. Show coverage beside any answered-query accuracy. Keep the
prepared checkpoint and policy unchanged and retain a labeled recording fallback.

For another claim of improved all-intent accuracy, collect independently authored
or otherwise suitable fresh held-out positive queries following the
[fresh evaluation guide](fresh-evaluation.md). Select the model and policy on
development data, lock them before evaluation, and report errors as well as
successes. The measured result currently supports 94%, not a promise of 99–100%.

Only after those gates should larger models, more adapter tuning, asynchronous
federation, secure aggregation, or differential privacy expand the scope. Each
extension needs a specific hypothesis, a measured cost, and its own validation.
