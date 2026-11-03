# Three-minute judging presentation

Use this as a spoken cue guide for the current one-laptop prototype. Three
minutes is a rehearsal default; confirm the actual presentation allowance.
Lead with the working federation system and its measured task quality.

## Prepare the two demo views

From PowerShell on the prepared laptop:

```powershell
Set-Location D:\Hackathons\loflwr
.\tools\start_encoder_demo.ps1 -PolicyFile research/results/encoder-v2/final-policy.json
```

Open two browser tabs at <http://127.0.0.1:8511>. Keep the launcher running.

1. **Tab A — selected model:** leave **Completed run directory** as
   `runs/encoder-flower-balanced-v2` and **Margin policy file (optional)** as
   `research/results/encoder-v2/final-policy.json`. Confirm the page identifies
   selected version **500** and independent confirmation **94.0% (282/300)**.
   Submit one practice query to warm model loading, then leave **Ask a query**
   visible. This is preparation, not a new accuracy evaluation.
2. **Tab B — recorded recovery:** first clear **Margin policy file (optional)**,
   then set **Completed run directory** to
   `runs/encoder-flower-trained-recovery-v2` and press Enter. Open
   **Training evidence**, scroll to **Recorded client participation**, and set
   **Attempt to inspect** to **1**. This run starts from the trained model;
   selected version 0 here means its warm-start checkpoint.
3. Have the [results report](../research/results/encoder-v2/README.md) ready for
   comparison questions. The saved curves and process events are recorded;
   questions entered into Tab A run live local inference.
4. Keep the [updated presentation figure](../research/results/remaining-work-v1/comparison.png)
   ready. Its first two panels show the new three-seed study at fixed attempt
   500, with 400-query imbalance and 2,760-query omission-control conditions
   separated. The last panel is the separate locked unsupported-query test.
   The [full matrix report](../research/results/encoder-matrix-v1/README.md)
   explains the conditions. Tab B deliberately uses the earlier completed
   replay; the newer five-attempt TLS recovery checks have their own report.

Rehearse once with internet disconnected. A clean clone needs the documented
assets and environments; this launcher assumes the laptop is already prepared.
If the page cannot run during judging, show the saved report and identify it as
recorded evidence. Do not describe a screenshot or replay as a live training run.

The newer complete five-attempt TLS rehearsals take **236 seconds for three
clients and 303 seconds for five clients**. They do not fit inside this entire
three-minute pitch. Use the labeled replay above, or rehearse one live
dropout/rejoin segment with the actual presentation allowance. The 60-second
training collection deadline is separate from startup and rejoin readiness.

## Spoken script and actions

### 0:00–0:25 — The problem

**Show Tab A.**

“FedLoRA-Edge asks a practical question: can machines with unequal training
budgets contribute to one useful model, without a data-rich participant
controlling the aggregation or a disappearing client stopping the run?

I built a banking-intent prototype with five logical clients. Each covers a
different domain. It predicts one of twenty intents and displays an authored
support response.”

### 0:25–0:55 — The design

“The original experiment trains frozen-A LoRA adapters on SmolLM2. Its accuracy
was weak, so I also implemented a separate frozen-MPNet encoder with a federated
classifier head. That head has 15,380 trainable parameters: 61,520 raw bytes per
direction.

Clients train from the same version. The server checks replies, waits only to
a deadline, and commits when a valid-update quorum is met. Equal survivor
weights make each participant’s aggregation coefficient visible.”

### 0:55–1:30 — Useful prediction with an independent score

**In Tab A, enter “My card payment is still pending.” and click
Classify query. Show the returned intent and response, or the clarification
message if one appears.**

“This question is being classified locally now. The model completed 500 actual
Flower training attempts with all five clients. It reached 95.5% on validation.
I then locked the model and scoring rules before opening a separate confirmation
set: it got 282 of 300 correct, or 94%.

The optional clarification rule trades coverage for fewer wrong answers. Its
measured result is 98.65% among answered queries at 74.33% coverage. Overall
accuracy remains 94%.”

### 1:30–2:10 — Failure and recovery

**Switch to Tab B. Move Attempt to inspect from 1 to 2, then to 3.**

“These are recorded events from a real process-interruption test on this laptop.
Attempt one has five contributors. During attempt two, one client’s optimizer
process was killed. Four survivors committed, each with weight 0.25. In attempt
three, the returning client synchronized fresh state and all five contributed
again. Validation stayed at 95.5% throughout.

The recovery test starts from the trained checkpoint. It demonstrates bounded
recovery, not training this model from scratch in three rounds.”

### 2:10–2:40 — Why the federation matters

"In all three matched seeds, removing the one-step verification client reduced
validation accuracy from 95.5% to 77.5%. It supplies the only training examples
for four intents, so its contribution matters despite its smaller work budget.

In the separate 400-query imbalance study, final accuracy averaged 62.5% with
sample weighting, 76.5% with equal weighting, and 78.17% with clipping. Most
of the improvement came from changing the weighting. These are development
comparisons; the independent twenty-intent score remains 94%."

### 2:40–3:00 — Contribution and next evidence

"My contribution is measured, reproducible federation using established methods.
Local feature preparation now passes in isolated processes, and all 33 repeated
study runs are complete. The clients still share one laptop. Next are three
physical laptops and new accuracy improvements evaluated on fresh independent
queries."

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
| Do 203 tests mean 100% model accuracy? | No. The 203 software checks passed with zero skips in the prepared environment. They test behavior and contracts. Intent accuracy comes from the separate 300-query confirmation evaluation. |
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
