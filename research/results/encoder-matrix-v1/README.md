# Three-seed encoder federation study

This is the fixed `encoder-matrix-v1` development study: 33 configurations,
training seeds 42/43/44, and 500 round attempts per configuration. It uses real
classifier-head optimization and sequential in-memory aggregation. It is
**not 33 Flower runtime or physical-device experiments**. The independently
confirmed demo checkpoint is unchanged.

## Matched conditions

The plan hash is
`feaa99c55fc908fe7b3236477ad0066fc70141c7624486b49d5caeddce78bd77`.
The plan freezes source/data hashes, the runtime, and actual saved initial head
tensors for each seed. Methods share those initial tensors, data, optimizer,
work limits and client-specific random streams.

| Setting | Value |
| --- | --- |
| Representation | Pinned frozen MPNet, 768-dimensional local features |
| Trainable state | 20-class linear head; 15,380 FP32 parameters / 61,520 raw bytes per direction |
| Optimizer | SGD, learning rate 20, weight decay `3.6231884057971014e-05`, class balancing, batch 1,024, local gradient norm cap 10 |
| Expanded training | 2,760 queries; domain client counts 513/615/608/616/408 |
| Extreme imbalance | Original 400-query scenario; client 1 owns 360, others 10 each |
| Equal budgets | One optimizer step per client per attempt |
| Heterogeneous budgets | `[1,4,2,2,1]` optimizer steps in domain order |
| Aggregation | Manifest-size weighted, equal, equal with median norm clipping |
| Evaluation | Reused 200-query validation at attempt 0 and every 50 attempts |
| Selection | Highest evaluated accuracy, then macro F1, then earliest attempt; fixed attempt-500 endpoint reported separately |

Each seed has nine method/data/work cells, one paired omission of client 4
under expanded/heterogeneous/equal conditions, and one paired client-1
unavailability condition under imbalanced/heterogeneous/clipped conditions.
The unavailable intervals are attempts 101â€“110, 251â€“260 and 401â€“410. They skip
that client's work; actual process loss is evaluated separately in deployment.
The quorum is three valid contributors.

The numerical runner reuses the existing `HeadTrainer` and aggregation code.
A real-optimizer fixture compares its final tensors and metrics exactly with
the original protocol-based runner. It avoids repeated growing checkpoint-file
writes during this numerical study; protocol and runtime evidence remain
separate. No confirmation or final-test queries are loaded.

## Reproduce

Use the documented Windows CPU environment for this frozen run. A new run must
use a fresh output directory and creates a new environment/source lock.

```powershell
python tools/run_encoder_matrix.py prepare --output runs/matrix-new
python tools/run_encoder_matrix.py run --output runs/matrix-new --workers 2
python tools/run_encoder_matrix.py report --output runs/matrix-new
```

The runner rejects changed inputs and incomplete cell directories rather than
silently overwriting them. A repeated invocation may reuse completed matching
cells. Investigate a failed partial cell explicitly before continuing. Selecting
the best evaluated validation checkpoint is development selection, not a fresh
test. Three seeds describe observed variability; they are insufficient to promise
generalization, significance or equal accuracy across participants.

## Completed measurements

All **33/33 cells** completed, totaling **16,500 committed round attempts**.
The [frozen plan](plan.json) and [full result export](summary.json) retain all
seed/run identities, validation histories, domain metrics, work, participation,
memory, timing and artifact hashes. No failed or unfavorable method is omitted.

Values below are percentages, **mean ± sample standard deviation** over three
training seeds. Final means use attempt 500; selected means use the predeclared
development selection rule. Zero observed SD on this fixed set is not a claim
of zero generalization uncertainty.

| Data / work / strategy | Final accuracy | Selected accuracy | Final worst domain |
| --- | --- | --- | --- |
| expanded-equal-sample | 95.50 ± 0.00 | 95.50 ± 0.00 | 90.00 ± 0.00 |
| expanded-equal-equal | 95.50 ± 0.00 | 95.50 ± 0.00 | 90.00 ± 0.00 |
| expanded-equal-equal_clipped | 95.50 ± 0.00 | 95.67 ± 0.29 | 90.00 ± 0.00 |
| expanded-heterogeneous-sample | 95.50 ± 0.00 | 95.50 ± 0.00 | 90.00 ± 0.00 |
| expanded-heterogeneous-equal | 95.50 ± 0.00 | 95.50 ± 0.00 | 90.00 ± 0.00 |
| expanded-heterogeneous-equal_clipped | 95.50 ± 0.00 | 95.50 ± 0.00 | 90.00 ± 0.00 |
| imbalanced-heterogeneous-sample | 62.50 ± 0.00 | 62.67 ± 0.29 | 45.00 ± 0.00 |
| imbalanced-heterogeneous-equal | 76.50 ± 0.00 | 77.83 ± 0.29 | 57.50 ± 0.00 |
| imbalanced-heterogeneous-equal_clipped | 78.17 ± 0.29 | 78.33 ± 0.29 | 57.50 ± 0.00 |
| expanded-heterogeneous-equal-omit4 | 77.50 ± 0.00 | 77.50 ± 0.00 | 0.00 ± 0.00 |
| imbalanced-heterogeneous-equal_clipped-unavailable1 | 78.00 ± 0.00 | 78.33 ± 0.29 | 57.50 ± 0.00 |

All expanded-data methods reached **95.5% final validation accuracy** across
all seeds despite unequal budgets. This demonstrates learning on this frozen
development setup, not another independent confirmation. The equal-budget
clipped seed-43 run briefly selected 96%; it finished at 95.5%.

In the 90%-imbalance condition, sample weighting finished at **62.5%**, equal
weighting at **76.5%**, and equal plus clipping at **78.17% ± 0.29 points**.
The weighting change accounts for most of this improvement. Clipping adds
1.67 points over plain equal at the fixed endpoint; both have the same 57.5%
worst-domain accuracy. These small-seed development results do not establish a
general clipping advantage. The earlier v2 best-checkpoint study used different
budgets/settings and must not be mixed with these fixed-final numbers.

Omitting the one-step verification client reduced final validation accuracy
from **95.5% to 77.5% in each paired seed**: a benefit of **18 percentage points**
for including it. Verification accuracy fell from 90% to 0%. This client owns
the only training examples of four intents; the experiment measures that
specific marginal knowledge contribution, not a universal benefit of weak
devices with redundant data. Surviving client random streams stay matched.

The scripted client-1 unavailability condition finished at **78%** for all seeds.
Control minus unavailable differences were **0, 0.5 and 0 percentage points**
for seeds 42/43/44. All 500 attempts still had quorum. This is predeclared skipped
work, not network/process-loss evidence; consult the separate deployment report.

Cross-entropy, macro F1, every domain, domain gap, selected/final hashes and
per-client work are in the full export. Timings were collected while other work
ran on the laptop. The summed per-cell wall times are 1.422 hours; cells ran in
two workers, so that sum is not elapsed end-to-end time or total CPU time.
