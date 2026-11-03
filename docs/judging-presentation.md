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

Rehearse once with internet disconnected. A clean clone needs the documented
assets and environments; this launcher assumes the laptop is already prepared.
If the page cannot run during judging, show the saved report and identify it as
recorded evidence. Do not describe a screenshot or replay as a live training run.

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

“In one matched experiment, removing the weakest verification client reduced
validation accuracy from 95.5% to 77.5%. That client supplied useful knowledge.

In the separate extreme-imbalance study, sample weighting reached 72%, equal
weighting 78.5%, and equal weighting with clipping 79%. Most of this observed
gain came from changing the weighting; half a point does not establish a
reliable clipping advantage.”

### 2:40–3:00 — Contribution and next evidence

“My contribution is measured, reproducible federation using established methods.
Five clients currently share one laptop; public-data features were prepared
centrally. Next are local feature preparation, three physical laptops, repeated
seeds, and accuracy improvements evaluated on a new independent test.”

## Short answers for judges

| Question | Answer |
| --- | --- |
| Did you achieve 100%? | No. The independent result is 94%, with 18 errors. Even the strictest frozen clarification rule made three errors among 223 answers. Its 98.65% result requires the 74.33% coverage figure. |
| Why change the model? | The bounded task is intent classification. The original generative LoRA experiment performed poorly; stronger sentence representations justified a separately identified head-training variant. Training data also expanded to 2,760 queries, so old and new scores are not a matched aggregation comparison. |
| Is the final model LoRA? | The demonstrated high-accuracy model trains a classifier head on frozen MPNet features. The original SmolLM2 B-only LoRA and MPNet B-plus-head experiments remain in the repository. Further adapter training did not improve the selected validation accuracy. |
| What is new? | I am not claiming to have invented LoRA, frozen-A training, equal averaging or clipping. The work integrates update validation, deadlines, quorum commits, fresh rejoining and visible influence, then measures them with matched stress and omission experiments and a locked confirmation test. |
| Is this still aligned with the submitted proposal? | The original LoRA implementation and evidence are preserved. The MPNet classifier is a disclosed architecture variant pursuing the same weak-device federation problem. Organizer acceptance of that change has not been verified; proposal continuity needs checking against the submitted abstract and current rules. |
| Why not use one centralized classifier? | A centralized reference is useful for task quality. Federation addresses the deployment constraint where each participant owns its data. This prototype exercises separate client archives and update exchange, but central public-data feature preparation means it does not yet demonstrate complete local preparation. |
| Does equal weighting guarantee fairness? | It gives equal coefficients to accepted survivors, not equal accuracy or guaranteed equal influence in every sense. Clipping limits update norms. Repeated absence can still create participation bias. |
| Did clipping beat averaging? | In the 400-example study, the best scores were 79% clipped versus 78.5% plain equal. That difference is one validation query in one seed; it is not strong evidence of superiority. Sample weighting reached 72%. |
| Did weak clients actually help? | In the matched expanded-data omission pair, the score fell from 95.5% to 77.5%, and verification accuracy from 90% to 0%, without client 4. Those four labels had no other training owner. This is one paired seed; the original LoRA omission result was null/unfavorable. |
| Are these five physical devices? | No. They are five logical clients on one laptop. The process kill and rejoin are real; the three-laptop experiment remains to be run. |
| Is this private or secure against malicious clients? | Training replies carry model updates. The current setup does not establish differential privacy, secure aggregation, or malicious-client robustness. Local preparation is a next step; model updates themselves can leak information. |
| How cheap is the system? | The selected head transmits 61,520 raw FP32 bytes per direction, before transport overhead. That is not total memory or end-to-end traffic. The frozen encoder still costs storage, RAM and feature-computation time. |
| Do 157 tests mean 100% model accuracy? | No. All 157 software tests passed with zero skips in the prepared environment. They check behavior and contracts. Intent accuracy comes from the separate 300-query confirmation evaluation. |
| How will you improve accuracy now? | Use training-only data and validation to test a small, declared next experiment, preserving adverse results. Freeze the next model and any clarification policy before scoring a new independently held-out set. The current 300 queries are now evaluation evidence; 99–100% is a target, not a promise. |

## Evidence to keep one click away

| Claim to show | Evidence |
| --- | --- |
| Final score, every domain, all fixed clarification policies | [Confirmation result](../research/results/encoder-v2/confirmation-result.json) |
| Model and rules fixed before confirmation | [Confirmation plan](../research/results/encoder-v2/confirmation-plan.json), [policy](../research/results/encoder-v2/final-policy.json), pre-evaluation commit `7413e6b` |
| Actual five-client run, selected checkpoint, payload | [Flower head export](../research/results/encoder-v2/flower-head.json) |
| Real interruption, survivor weights, fresh rejoin | [Trained-model recovery](../research/results/encoder-v2/trained-recovery.json) |
| Unequal budgets and clipping | [Heterogeneous head](../research/results/encoder-v2/heterogeneous-head.json), [clipped counterpart](../research/results/encoder-v2/heterogeneous-clipped-head.json) |
| Weak-client omission and interpretation | [Omission export](../research/results/encoder-v2/omission-head.json), [paired comparison report](../research/results/encoder-v2/README.md) |
| 90% data imbalance, including plain equal control | [Imbalance trials](../research/results/encoder-v2/imbalance-trials.json) |
| Negative adapter result | [Federated adapter](../research/results/encoder-v2/federated-adapter.json), [centralized adapter reference](../research/results/encoder-v2/centralized-adapter.json) |
| Software tests, live query, locked-file checks | [Handoff verification](../research/results/encoder-v2/handoff-verification.json) |
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
