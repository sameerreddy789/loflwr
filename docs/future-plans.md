# Future plans after the completed local studies

Updated **4 November 2026**. The stable frozen-MPNet/federated-head model remains
**94% accurate on independent confirmation (282/300)**. New development and
scope results do not replace that score. Head-only training and the original
SmolLM2 LoRA benchmark retain separate identities.

## What moved out of the plan

- [Client-local preparation](../research/results/encoder-local-v1/README.md)
  passed in five- and three-client layouts with isolated, offline processes,
  matching features and zero changed train/validation predictions. Portable
  configuration and code-only app packaging also passed.
- [All 33 encoder study cells](../research/results/encoder-matrix-v1/README.md)
  completed across three seeds, including weighting/clipping controls, unequal
  budgets, extreme imbalance, weak-client omission and scripted unavailability.
  The omission benefit was 18 validation points per seed. These are numerical
  federation studies, not 33 Flower deployments.
- [The locked scope test](../research/results/scope-v1/README.md) reduced false
  accepts under the strictest margin from 311/2,279 to 57/2,279. The detector
  remains experimental because fresh supported-query coverage is unknown.
- [CPU profiling](../research/results/encoder-local-v1/demo-profile.json)
  measured 152 ms median and 229 ms p95 warm query latency in this observed run.
  Timing probes contain no accuracy labels.
- [Automated demo preflight](demo-preflight.md) passed all five checks on the
  prepared laptop: dependencies, evidence, local port, Python and cached model
  inference. It produces a new readiness report without changing the model.
- [Nine fixed-work runs](../research/results/resource-frontier-v1/README.md)
  measured synchronization frequency. Four local steps per average used 75%
  less estimated raw payload than one step, but final reused-validation accuracy
  fell from 95.5% to 93.5% in every seed. This is a development trade-off, not a
  change to the confirmed model or a physical-network measurement.
- [Browser rehearsal and fresh evaluator](../research/results/rehearsal-evaluator-v1/README.md)
  are complete as software checks: ten browser observations passed, and the new
  evaluator has fixture coverage for frozen inputs, publication and one-time
  scoring. The actual empty collection was refused before a lock or inference.
  No independent human data or new accuracy result was produced.
- [Isolated Flower rehearsals](../research/results/remaining-work-v1/README.md)
  passed three/five-client dropout, deliberate quorum loss and fresh rejoining.
  Fresh offline three-client startup also passed; these are same-host results.

The [remaining-work checklist](remaining-work-plan.md) records the current
runtime gates. Keep the selected head, original confirmation, CLINC test and
new scope-test locks unchanged. Use fresh run IDs for further experiments.

The [preset dashboard](encoder-demo.md), [six-slide deck](../presentation/README.md),
[short narrated backup](../presentation/narrated-demo.md),
and [blinded collection kit](fresh-query-collection.md) are implemented. The empty
pack deliberately reports zero submissions and cannot authorize scoring.

## 1. Rehearse with the actual presenter

Run the [preflight](demo-preflight.md), then follow the
[judging script](judging-presentation.md) and [encoder demo guide](encoder-demo.md).
Launch from a fresh process without internet, enter live queries, explain
recorded versus live training, show a skipped quorum and a fresh rejoin, and
keep a clearly labeled recording as a fallback. Say 94% overall confirmation
accuracy; pair 98.65% answered-query accuracy with its 74.33% coverage.

Done when the presenter can complete the sequence, recover the launch if it
fails, and answer why a head-only variant differs from LoRA. Code checks do not
replace this human rehearsal. Verify scope continuity with the organizers if
required by the submitted abstract; do not assume approval of an architecture
change or that the Round 1 scoring weights apply to finals.

## 2. Use three physical laptops

Only one is available now. Follow the [per-laptop guide](encoder-laptop-rehearsal.md)
when two more are available. Prepare each machine's own records locally, pin and
smoke-test its environment, establish the intended LAN route/TLS/known-node
identities, and calibrate actual work. Record one client disappearing, a bounded
survivor decision, below-quorum checkpoint retention, and fresh synchronization.
Finish with a fresh startup while external internet is disconnected.

Measure whole-process memory, per-machine CPU timings, network wire bytes and
completed work. Serialized payload bytes omit TLS framing and retransmissions;
summed RSS can count shared memory twice. A same-host namespace test does not
establish physical reachability or performance on a weaker machine.
Use the fixed-work study as a starting point for choosing how often to
synchronize; verify the actual latency/traffic trade-off on the intended network
before selecting a cheaper schedule. The local study excludes encoder preparation.

## 3. Confirm future accuracy and scope changes

Use the [fresh evaluation protocol](fresh-evaluation.md) and the executable
[collection kit](fresh-query-collection.md). The
[new runner](fresh-evaluation-runner.md) initially evaluates only the unchanged
confirmed head and fixed clarification rule. A combined scope comparison needs
its own declared implementation and lock before exposing a fresh collection;
do not score the baseline first and then tune a combined policy on those results.
The 300-query
confirmation, original final test, CLINC set and 2,279-query scope test have all
been scored. They are evidence, not new model-selection resources.

Gather suitable fresh supported queries, preferably independently authored,
and define an application-relevant mixture of supported and unsupported inputs.
Select the next model and combined scope/clarification policy on development
only. Lock everything before final evaluation. Report top-1 accuracy, macro F1,
worst domain, false accepts, coverage, and errors. A lower false-accept rate on
negative queries alone cannot establish better full-system accuracy.

99–100% is unachieved. New data or training can be tested, but no score is promised.
A classifier should request clarification when appropriate, with measured coverage
costs, rather than reporting selectively filtered performance as overall accuracy.

## 4. Extend only after the preceding gates

The original LoRA multi-seed program remains distinct from the completed encoder
matrix. Additional adapter experiments, heterogeneous ranks, asynchronous
federation, secure aggregation and differential privacy need separate hypotheses,
compatible runtime tests, budgets and evidence. The project does not currently
claim a new optimizer, formal privacy, malicious-client robustness or a general
convergence guarantee.
