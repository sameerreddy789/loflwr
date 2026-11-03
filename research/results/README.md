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

## Cheap classification reference

A fixed centralized TF-IDF/logistic-regression reference trained on the same
400-example imbalanced manifest achieved **41.5% validation accuracy**, macro
F1 **0.36003**, and worst-domain accuracy **20%**. Fit time was 0.42 s. See
[the complete reference configuration](tfidf-validation.json).

It uses raw centralized training data and inverse-class-frequency loss weights;
it is task-efficiency context rather than a federated-LoRA comparison. No
hyperparameters were chosen on this validation result. It reminds us that a
language model is not automatically the best solution for a bounded intent task.

## Interpreting the next experiments

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
