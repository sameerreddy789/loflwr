# Class-balanced encoder head

The sequential five-client head reached **95.5% (191/200) validation accuracy**
with class-balanced cross-entropy, a longer optimization horizon and weaker
weight decay. Live CPU encoder inference exactly reproduced all cached logits
on the same 200 validation queries. This is still development-set evidence.
The actual Flower rerun, stress comparisons and new adapter trials are pending
at this publication milestone; they will receive separate completed results.

## Frozen comparison

All three configurations were written before evaluation. They use the same
MPNet/data manifest, seed 42, equal client weights, one full local batch per
client and SGD. Local objective weights are `N/(K*n_class)`, where `K` is the
number of classes present locally. With four disjoint classes on each of five
clients, the average full-batch gradient equals the twenty-class balanced
objective. A unit test verifies that relationship with unequal class counts.

| Configuration | LR | Weight decay | Attempts | Selected validation accuracy |
| --- | --- | --- | --- | --- |
| Class balance alone | 5 | 0.0004 | 300 | 90.0% |
| Weaker regularization, longer run | 5 | 1/(10*2760) | 1500 | See complete export |
| Faster, longer run | 20 | 1/(10*2760) | 1000 | **95.5%, attempt 500** |

The original unweighted LR5/300-attempt run scored 92.5%; class balancing alone
was worse. This is a combined optimization improvement, not evidence that class
weights universally help. Selection uses accuracy, then macro F1, then the
earliest evaluated checkpoint. Later attempts still run and remain reported.

The selected head hash is
`385ca9ade96264eafb0a77947847f84436b4d9c21d438d01cce378c6467b8cc2`.
Its predictions match centralized logistic C10 exactly. Worst-domain accuracy is
90%. See the [remaining-error analysis](../../encoder-error-analysis.md).

The [offline demo](../../../docs/encoder-demo.md) supports this run, actual Flower
head runs, live questions and recorded training evidence. The full WSL test
suite passed **149 tests, zero skips**. No confirmation query has been scored at
this milestone.

- [All three frozen head trials](head-trials.json)
- [Live versus cached inference](live-parity.json)
- [Error overlap](error-overlap.json)
