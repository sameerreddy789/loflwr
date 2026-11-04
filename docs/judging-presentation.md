# Three-minute judging presentation

Use the [six-slide offline deck](../presentation/fedlora-edge-demo.pptx), with
speaker notes and editable evidence charts. Three minutes is a rehearsal default;
confirm the actual allowance. The selected model is frozen MPNet plus a federated
classifier head; head-only training is distinct from the original LoRA experiment.

## Prepare

```powershell
Set-Location D:\Hackathons\loflwr
.\tools\check_demo.ps1
.\tools\start_encoder_demo.ps1
```

If the preflight says `open_existing`, open its URL and skip the launch command.
See the [preflight guide](demo-preflight.md) for failures or an alternate port.
Open <http://127.0.0.1:8511>. **Live banking query** is the default, with the
fixed clarification rule enabled. Confirm version 500 and independent accuracy
94.0% (282/300). Warm one query before judging: this is preparation, not evaluation.
The sidebar now also offers **Recorded recovery** and **Experiment comparisons**;
no run paths need editing. **Custom run** is for technical inspection.

Keep the [short narrated backup](../presentation/narrated-demo.md) locally.
It combines actual query/recovery clips with evidence cards and burned captions,
and stays labeled **Recorded walkthrough**. Its voice is generated offline.
The original [silent 6:56 recording](../presentation/demo-backup.webm) remains
available for longer inspection. These recordings are separate from a live run
and from the three-minute pitch. The complete TLS rehearsals take 236 seconds
for three clients and 303 seconds for five, so neither fits this whole pitch.
The 60-second collection bound excludes startup and rejoin readiness.

Rehearse with internet disconnected on the already prepared machine. A clean
clone needs the documented dependencies and ignored model/checkpoint assets.

## Slide sequence and spoken cues

| Time | Show | Say |
| --- | --- | --- |
| 0:00–0:25 | Slide 1: problem | Can machines with unequal budgets train one useful model without a data-rich client controlling aggregation or a disappearing client stopping the run? Five logical clients each own a banking domain. |
| 0:25–0:55 | Slide 2: architecture | The original frozen-A SmolLM2 LoRA experiment performed poorly. A separately measured frozen-MPNet/head variant improves task quality. Clients share head updates, starting from one version; the server checks identity, deadline and quorum before committing. Local preparation is now tested in isolated processes. |
| 0:55–1:25 | Slide 3: confirmation | The head has 15,380 trainable parameters, or 61,520 raw FP32 bytes per direction. The locked model got 282/300 correct: 94%. The clarification rule got 220/223 answered queries correct, 98.65%, at 74.33% coverage. We have not reached 99–100%. |
| 1:25–2:00 | Slide 4: matched comparisons | Under 90% data imbalance, final validation accuracy averaged 62.5% sample-weighted, 76.5% equal and 78.17% clipped across three seeds. Including the one-step verification client gave an 18-point benefit in every seed because it owns four unique intents. These are reused-validation numerical federation studies. |
| 2:00–2:35 | Slide 5: recovery | A real same-laptop TLS run accepted 5, 4, 5, 2, 5 clients. Four survivors commit at 25% each; two miss quorum three and preserve version 3. Returning clients synchronize current state. These are recorded process events, not five physical laptops. |
| 2:35–3:00 | Slide 6, then Live banking query | Enter “My card payment is still pending.” and classify. Explain the predicted intent and authored response, or honestly show clarification if returned. Finish with the next evidence gates: three physical laptops and independent fresh queries before another accuracy claim. |

## If judges want to inspect the system

Choose **Recorded recovery**, leave **Five clients** selected, and click attempts
2, 3, 4 and 5. Point to weights, the unchanged version/hash after missed quorum,
and fresh-state return. Choose **Three clients** to inspect quorum two. Every
view says this is a recorded one-laptop experiment. The source is
`encoder-isolated-five-v8` or `encoder-isolated-three-v6`.

Choose **Experiment comparisons** to inspect the 33 completed runs, the
three-seed table and the weak-client omission result. The separate scope-detector
finding is collapsed and marked experimental. It is not deployed into the live
query classifier. Select **Live banking query** to return to the confirmed model.

## Short answers for judges

| Question | Answer |
| --- | --- |
| Did you achieve 100%? | No. The independent result is 94%, with 18 errors. Even the strictest frozen clarification rule made three errors among 223 answers. Its 98.65% result requires the 74.33% coverage figure. |
| Did you test another dataset? | Yes. After freezing the mapping and model, CLINC150 gave 28/29 correct on its one compatible intent. This is not twenty-intent accuracy. The strictest fixed margin also falsely accepted 5/120 unsupported banking queries, exposing the need for separately evaluated scope detection. |
| Why change the model? | The bounded task is intent classification. The original generative LoRA experiment performed poorly; stronger sentence representations justified a separately identified head-training variant. Training data also expanded to 2,760 queries, so old and new scores are not a matched aggregation comparison. |
| Is the final model LoRA? | The demonstrated high-accuracy model trains a classifier head on frozen MPNet features. The original SmolLM2 B-only LoRA and MPNet B-plus-head experiments remain in the repository. Further adapter training did not improve the selected validation accuracy. |
| What is new? | I am not claiming to have invented LoRA, frozen-A training, equal averaging or clipping. The work integrates update validation, deadlines, quorum commits, fresh rejoining and visible influence, then measures them with matched stress and omission experiments and a locked confirmation test. |
| Is this still aligned with the submitted proposal? | The original LoRA implementation and evidence are preserved. The MPNet classifier is a disclosed architecture variant pursuing the same weak-device federation problem. Organizer acceptance of that change has not been verified; proposal continuity needs checking against the submitted abstract and current rules. |
| Why not use one centralized classifier? | A centralized reference is useful for task quality. Federation addresses the constraint where participants own their data. The new local preparation path embeds each participant's own records offline in an isolated process; the server receives metadata and head updates. The public benchmark was provisioned by an auditor, so production data ownership and physical isolation remain unproven. |
| Does equal weighting guarantee fairness? | It gives equal coefficients to accepted survivors, not equal accuracy or guaranteed equal influence in every sense. Clipping limits update norms. Repeated absence can still create participation bias. |
| Did clipping beat averaging? | In the new three-seed 400-query study, fixed-final accuracy was 78.17% ± 0.29 points clipped versus 76.5% equal and 62.5% sample-weighted. Clipping and equal averaging both had 57.5% worst-domain accuracy. Most of the gain came from equal coefficients; three development seeds do not establish a general advantage. |
| Did weak clients actually help? | In all three matched unequal-budget runs at attempt 500, including client 4 gave 95.5% versus 77.5% without it; verification fell from 90% to 0% when omitted. It owns the only training examples of four intents. This is a specific knowledge contribution under a one-step budget; the original LoRA omission result was null/unfavorable. |
| Are these five physical devices? | No. They are five logical clients on one laptop. The process kill and rejoin are real; the three-laptop experiment remains to be run. |
| Is this private or secure against malicious clients? | Local preparation and TLS are exercised, and training replies carry head updates. Model updates and metadata can still leak information. Differential privacy, secure aggregation and malicious-client robustness are not established. |
| How cheap is the system? | The selected head transmits 61,520 raw FP32 bytes per direction, before transport overhead. That is not total memory or end-to-end traffic. The frozen encoder still costs storage, RAM and feature-computation time. |
| Do 260 tests mean 100% model accuracy? | No. The 260 software checks passed with zero skips in the prepared environment. They test behavior and contracts. Intent accuracy comes from the separate 300-query confirmation evaluation. |
| Can clients communicate less often? | In nine fixed-work development runs, averaging every four local steps cut estimated raw payload by 75%, but validation accuracy fell from 95.5% to 93.5% in every seed. These are cached-feature CPU experiments; physical network costs remain unmeasured. |
| Can it reject unsupported banking requests? | The fixed margin alone falsely accepted 311/2,279 new unsupported BANKING77 queries. Adding a separately trained, pre-locked scope detector reduced this to 57/2,279. The detector is experimental: fresh supported-query coverage is untested, so this is not 97.5% overall accuracy or a replacement for the confirmed demo. |
| How will you improve accuracy now? | Use training-only data and validation to test a small, declared next experiment, preserving adverse results. Freeze the next model and any clarification policy before scoring a new independently held-out set. The current 300 queries are now evaluation evidence; 99–100% is a target, not a promise. |
| Did the subsequent larger-model experiment solve the accuracy problem? | No. Four transformer-layer adaptation configurations peaked at 96% reused-validation accuracy. The twelve-layer ten-epoch run finished at 94.5% and has a much larger trainable state. These centralized diagnostics did not replace the confirmed Flower model. |

## Evidence to keep one click away

| Claim to show | Evidence |
| --- | --- |
| Final score, every domain, all fixed clarification policies | [Confirmation result](../research/results/encoder-v2/confirmation-result.json) |
| Model and rules fixed before confirmation | [Confirmation plan](../research/results/encoder-v2/confirmation-plan.json), [policy](../research/results/encoder-v2/final-policy.json), pre-evaluation commit `7413e6b` |
| Actual five-client run, selected checkpoint, payload | [Flower head export](../research/results/encoder-v2/flower-head.json) |
| Real interruption, survivor weights, fresh rejoin | [Trained-model recovery](../research/results/encoder-v2/trained-recovery.json) |
| New TLS recovery, missed quorum and fresh offline start | [Role-isolated Flower report](../research/results/remaining-work-v1/README.md) |
| Unequal budgets and clipping | [Heterogeneous head](../research/results/encoder-v2/heterogeneous-head.json), [clipped counterpart](../research/results/encoder-v2/heterogeneous-clipped-head.json) |
| Weak-client omission and interpretation | [Omission export](../research/results/encoder-v2/omission-head.json), [paired comparison report](../research/results/encoder-v2/README.md) |
| Presentation comparison figure | [PNG](../research/results/encoder-v2/judging-evidence.png), [source hashes and comparison conditions](../research/results/encoder-v2/judging-evidence-provenance.json) |
| 90% data imbalance, including plain equal control | [Imbalance trials](../research/results/encoder-v2/imbalance-trials.json) |
| New complete three-seed comparisons, fixed final versus selected | [33-cell report](../research/results/encoder-matrix-v1/README.md), [full measurements](../research/results/encoder-matrix-v1/summary.json) |
| Participant-local encoding, preparation costs, CPU latency | [Local preparation](../research/results/encoder-local-v1/README.md) |
| Fixed unsupported-query evaluation | [Scope test](../research/results/scope-v1/README.md), pre-inference commit `8e3f241` |
| Negative adapter result | [Federated adapter](../research/results/encoder-v2/federated-adapter.json), [centralized adapter reference](../research/results/encoder-v2/centralized-adapter.json) |
| Software tests, live query, locked-file checks | [Original handoff verification](../research/results/encoder-v2/handoff-verification.json), [latest 203-test verification](../research/results/remaining-work-v1/verification.json) |
| Original LoRA results and proposal history | [Original pilot](../research/results/pilot-r2-20261103/README.md), [abstract alignment](../research/abstract-alignment.md) |

Before presenting, confirm both demo tabs display the intended run, the model is
cached, and the evidence links open. Keep the selected checkpoint, policy and
confirmation result unchanged. Do not add new performance numbers based on a
few successful stage queries.

The user-supplied rubric describes **Round 1 abstract assessment**: problem
understanding, technical feasibility, novelty, clarity, evaluation and readiness.
Use those dimensions to make this presentation coherent. The current final-round
scoring weights and prototype-scope approval have not been independently verified;
do not present Round 1 percentages as confirmed finals criteria.
