# Measured results

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
