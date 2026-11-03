# Measured results

The [external CLINC150 report](external-clinc-v1/README.md) adds **28/29 correct
on one compatible intent** and separately measures unsupported-topic false
accepts. Its strictest frozen margin accepts 5/120 unsupported banking queries.
The source mapping and model were committed before scoring; the result does
not replace the all-twenty-intent confirmation below.

The [class-balanced head report](encoder-v2/README.md) records **94.0% independent
confirmation accuracy (282/300)** and **95.5% validation accuracy (191/200)** for
the selected actual five-client Flower head, version 500. The model and all
margin rules were locked in commit `7413e6b` before confirmation. The strictest
rule answered 223/300 queries with 220 correct: **98.65% at 74.33% coverage**.
No rule reached 99%. See the [aggregate confirmation result](encoder-v2/confirmation-result.json),
matched stress comparisons, weak-client omission and trained-model dropout/rejoin
in that report. Head-only training is distinct from the original LoRA benchmark.

The later [representation diagnostics](encoder-representation/README.md) test
adapting transformer layers on public training data. Short and longer variants
peaked at 96% reused-validation accuracy, with the larger model ending at 94.5%.
They are centralized references, not
new confirmed or Flower-trained models. The original 94% result remains intact.

The [encoder recovery report](encoder-v1/README.md) preserves the earlier 92.5%
validation result from actual five-client Flower head training, centralized
references, local process recovery, and the adapter trial with no accuracy gain.

Updated 3 November 2026. These measurements come from actual local runs on one
laptop. JSON exports retain run/configuration/model/data identities and hashes
of original evidence. Model weights, data queries and private node keys remain
outside source control. The full multi-seed study and three-physical-laptop
experiment have not been completed.

## Local optimization and transport

| Check | Measured result | Evidence |
| --- | --- | --- |
| Windows CPU B-only smoke | Two optimizer steps, eight examples; tiny smoke-batch loss 5.967963 to 5.711006; base and A hashes unchanged, B changed; local training 65.17 s, peak sampled RSS 1.024 GiB | [CPU smoke](smoke-cpu.json) |
| WSL2 CUDA B-only smoke | Same work/configuration in FP32; smoke-batch loss 5.967981 to 5.711070; frozen-state checks pass; local training 11.24 s, peak allocated GPU memory 668.1 MiB | [GPU smoke](smoke-gpu.json) |
| First real Flower LM round | Five balanced clients, two steps each, all accepted, version 1; each coefficient 0.2; full validation accuracy 31%, macro F1 0.24164, task loss 5.00248 | [First round](flower-first-round.json) |
| Reusable trainer | Cold/warm/interleaved execution produces bit-identical deltas; actual process kill, replacement and deadline checks pass; a warm two-step job took 4.31 s including job overhead | [Worker calibration](persistent-worker.json) |
| Scoring optimization | All 200 validation predictions and completion scores unchanged; 51.77 to 27.22 s; allocated GPU peak 2.084 to 1.141 GB | [Scoring parity](scoring-optimization.json) |
| Training synchronization optimization | Identical real GPU deltas, losses, norms and work counts for matched input/configuration/seed | [Training parity](training-parity.json) |

The initial five-client run used a 600-second collection limit. Its measured
collection took 106.95 seconds and the run's server/evaluation work 174.70
seconds. It preceded reusable-worker optimization. The measured B payload is
737,280 raw FP32 bytes per direction. Serialized Flower replies were about
753.8 kB per client; protobuf counts exclude HTTP/TLS framing and retransmission.

The smoke-batch loss changes are optimization checks, not held-out convergence.
The first Flower round has no matched frozen-base evaluation, so its accuracy
cannot establish improvement. CPU and GPU timings include different warm-up
costs and are not a controlled hardware speed comparison.

CUDA **allocated** memory describes live tensor allocation; **reserved** memory
includes allocator cache. The old scorer reported 9.39 GB reserved on a GPU with
6 GB dedicated memory under WSL/Windows. This is not a physical VRAM usage
measurement. The optimized scorer reported 3.70 GB reserved and releases unused
cache between evaluation and training. Each reported peak is process-local,
not the whole laptop's simultaneous total.

## Actual interruption, retention and rejoining

Run `ray-recovery-calibration-20261103` used five real logical clients, two
optimizer steps each, quorum three and a 60-second collection deadline. The
deadline was deliberately tested against cold and replacement-worker overhead.
See [the recorded run](ray-recovery.json).

| Attempt | Accepted clients | Outcome | Collection time |
| --- | --- | --- | --- |
| 1 | 0, 1, 2, 3 | Committed version 1; client 4 missed the deadline | 58.53 s |
| 2 | 0 | Client 1's process was killed after one optimizer step; quorum missed, version 1 retained | 58.16 s |
| 3 | 0, 1, 2, 3, 4 | All accepted after synchronization; committed version 2 | 54.82 s |

The attempt-1 and attempt-2 B hash is identical:
`0db7113fc4aa53bc0b896442065b53a30ef3126d4de16dd9d954dab43091101a`.
Returning clients synchronized that state before attempt 3. This verifies actual
process interruption, bounded collection, below-quorum retention, and fresh
rejoining. It also shows that 60 seconds is too tight for this cold single-worker
queue. The matched pilot therefore declares 300 seconds for every strategy.

The [offline-start check](offline-start.json) started a fresh CPU process in a
Linux network namespace with only loopback and no IPv4 routes. An external
connection failed with `Network is unreachable` before the model was imported.
Cached model/checkpoint loading and one query completed in 18.01 s. This proves
fresh offline inference on the prepared laptop; it does not establish an offline
multi-laptop deployment.

Run `tls-node-recovery-r2-20261103` exercised the Deployment Runtime with three
actual SuperNode process trees, TLS and registered node keys on this laptop.
Budgets were `[1,2,1]`, quorum two, and the collection deadline 120 seconds.
All three clients contributed in attempt 1 (88.54 s). In attempt 2 the middle
SuperNode and its children were terminated after one completed optimizer step;
the other two clients committed version 2 (78.11 s) with coefficient 0.5 each.
The restarted node synchronized version 2 and all three contributed in attempt
3 (86.52 s), committing version 3. See [TLS recovery evidence](tls-node-recovery.json).
This is a same-host network/process test, not a three-laptop measurement.

The [fresh offline Deployment Runtime check](offline-deployment.json) then started
three authenticated local nodes inside a loopback-only namespace. It committed
one B update round with two accepted contributors in 112.70 seconds; the third
client missed the 120-second collection deadline. Total fresh-start execution
was 142.18 seconds. [Round details](offline-tls.json) retain that missed work.
This demonstrates offline bootstrap and quorum-based training on the prepared
machine; it does not demonstrate all-client completion within a live deadline.

## Cheap classification reference

A fixed centralized TF-IDF/logistic-regression reference trained on the same
400-example imbalanced manifest achieved **41.5% validation accuracy**, macro
F1 **0.36003**, and worst-domain accuracy **20%**. Fit time was 0.42 s. See
[the complete reference configuration](tfidf-validation.json).

It uses raw centralized training data and inverse-class-frequency loss weights;
it is task-efficiency context rather than a federated-LoRA comparison. No
hyperparameters were chosen on this validation result. It reminds us that a
language model is not automatically the best solution for a bounded intent task.

The same fixed classifier on the balanced 800-example manifest measured **87%
validation accuracy**, macro F1 **0.868288**, and worst-domain accuracy **77.5%**.
Run `tfidf-balanced-validation-20261103` uses the same held-out queries and
settings; [its exported configuration](tfidf-balanced-validation.json) records
the changed training hash. This changes both data quantity and class coverage,
so it does not isolate balancing or prove a federated-method benefit. The
[accuracy investigation](../accuracy-diagnosis.md) connects this measurement
to the partial federated run and outstanding diagnostics.

## Interpreting the next experiments

The [99% feasibility reference](accuracy-feasibility-references.json) completed
twelve fixed method/scenario combinations on all 200 validation queries.
With 2,760 eligible training queries, MPNet nearest-example retrieval achieved
95.5% accuracy, macro F1 0.9553 and worst-domain accuracy 92.5%. The strongest
reference on the original imbalance scored 78%; on balanced data it scored 93%.
No reference reached 99% top-1 accuracy. [The comparison figure](accuracy-feasibility.png)
shows all twelve outcomes; [its provenance](accuracy-feasibility-figure-provenance.json)
records input and generator hashes. These are centralized classifiers with a frozen
sentence encoder or TF-IDF, not federated LoRA. The
[review](../accuracy-99-feasibility.md) includes all twelve results, source
checks, apparent annotation issues and uncertainty limits.

The [post-hoc distinct-label candidate diagnostic](accuracy-candidate-recall.json)
reproduces **191/200 top-1**, **198/200 top-2 recall**, and **200/200 top-3 recall**
from the same expanded-training MPNet cache. Recall above rank one means the
correct answer is among the candidates, not that a classifier selects it.
No reranker was implemented. All figures reuse validation; none is a new final
test or federated-model result.

The original seed-42 twenty-attempt test measurements are now available for
[sample weighting](pilot-sample-complete.json),
[plain equal weighting](pilot-equal-complete.json),
[equal plus clipping](pilot-main-complete.json),
[centralized B-only training](pilot-centralized-complete.json), and the
[frozen base](pilot-base-complete.json). Test accuracies are respectively
24.6%, 35.4%, 34.4%, 40.6%, and 32%. These do not show that clipping improves
quality over plain equal weighting, or that the current generative configuration
is a high-accuracy classifier. The full multi-seed matrix remains incomplete.
The resumed sample run lacks complete phase-time
history for its killed invocation; do not compare its phase total as full runtime.

The completed [scripted-dropout run](pilot-dropout-complete.json),
`pilot-r2-20261103-dropout`, committed all 20 attempts and scored **33.4% final-test
accuracy (167/500)**, macro F1 **0.28459**, and worst-domain accuracy **14%**.
Client 1 was killed after two completed optimizer steps on attempts 4, 8 and 12;
its unsuccessful replies were rejected. Each of the four survivors received
0.25 weight. On attempts 5, 9 and 13 the rejoining client synchronized the
preceding committed B, and all five contributed with 0.20 weight.
Participation was 20/17/20/20/20 accepted contributions across clients 0–4.
Accepted work totaled 592 optimizer steps, 2,368 examples and 132,853 tokens;
these counts exclude the discarded partial jobs. Collection took 31.38–59.07
seconds against the declared 300-second limit; full round times also include
aggregation and evaluation. The one-point test decrease versus the main run
supports continuation/recovery in this seed, not a general robustness or
high-accuracy claim. This was one-laptop simulation.

The [complete seven-configuration pilot](pilot-r2-20261103/README.md) includes
the client-4 omission run: **34.8%** test accuracy without it versus **34.4%**
with it, and **32% verification-domain accuracy in both**. This is a
null/unfavorable single-seed result for the weak-client utility claim. It does
not show that inclusion can never help; changing membership also changes
aggregation weights and clipping. The report preserves all configurations,
work counts, domain scores, loss curves and final-test macro F1.

### Final-checkpoint scoring diagnostics

Three additional validation-only diagnostics used final version-20 checkpoint
copies and the same fixed sum/mean/centered rules. All 200 validation queries
were scored. Original-score accuracies and B hashes match the frozen study's
recorded final validation evaluations.

| Checkpoint | Original sum | Per-token mean | Content-free centered | Corrected/harmed by centering |
| --- | ---: | ---: | ---: | ---: |
| [Sample weighting](accuracy-scoring-sample-final.json) | 22.5% | 18.5% | 52% | 65 / 6 |
| [Equal weighting](accuracy-scoring-equal-final.json) | 34.5% | 49% | 63% | 67 / 10 |
| [Equal plus clipping](accuracy-scoring-main-final.json) | 32.5% | 43.5% | 58.5% | 71 / 19 |

The correction helps, but these are exploratory reused-validation results,
not new final-test evaluations or an active scoring-contract change. They
support the [next architecture experiment](../accuracy-recovery-plan.md),
without establishing a 99% result.

Two complete validation scoring diagnostics used cached model assets and
checkpoint copies, with no additional training. The frozen base scored
31.5% / 21% / 42% under original sum / mean-token / content-free-centered sum.
The sample-weighted version-9 checkpoint scored 34.5% / 39.5% / 64.5% under
the same rules. Centering corrected 65 and harmed five original version-9
predictions; worst-domain accuracy rose from 7.5% to 45%. All 200 validation
queries were scored. See [base evidence](accuracy-scoring-base.json),
[version-9 evidence](accuracy-scoring-sample-v9.json), and the
[interpretation](../accuracy-diagnosis.md). These are exploratory validation
results, not changes to the active benchmark or final-test scores.

The first frozen study, `pilot-20261103`, was interrupted after its first
sample-weighted attempt. Two clients encountered `FileNotFoundError` while a
progress snapshot was being replaced on the Windows-mounted WSL filesystem.
The round consequently used only clients 0, 2 and 3. That is not the declared
healthy weighting comparison. [Its evidence is retained](interrupted-pilot-sample.json).
The fix makes only advisory progress reads tolerate a transient missing or
incomplete snapshot; checkpoint/update validation remains strict. The replacement
study `pilot-r2-20261103` starts from the same initial B and settings, with new
source/configuration identities. No hyperparameters were changed using test
results. The interrupted run must not be pooled into the main comparison.

The [frozen study runner](../../docs/study-runner.md) fixes manifests, common A,
initial B, settings, source hashes and attempt horizons before final-test access.
Its seed-42 pilot compares frozen base, centralized B-only training, sample
weighting, equal weighting, equal weighting plus clipping, scripted dropout,
and omission of the weakest verification client. Partial or failed runs remain
visible. A source-code or toy protocol check is not a language-model experiment.

Equal coefficients address dataset-size dominance among survivors. They do not
guarantee equal domain accuracy, useful contribution, malicious-client
robustness, secure aggregation or differential privacy. Those claims need their
own evidence. Refer to the [acceptance status](../implementation-status.md) and
[evaluation contract](../evaluation.md) before presenting results.
