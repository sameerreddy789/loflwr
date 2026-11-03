# Inspect recorded runs

The Streamlit console reads saved run evidence and can score a query using the
cached SmolLM2 model and a selected B checkpoint. It never starts training.
No results are generated for an empty run directory.

Install the project's dashboard dependencies in the tested Python environment,
then launch from the repository root:

```powershell
python -m streamlit run fedlora_edge/dashboard.py -- --runs-dir runs
```

Open the local URL printed by Streamlit. Select a run in the sidebar. Use
**Refresh files**, or enable the five-second refresh while a run is writing.
The latest event timestamp is visible; a saved file is not proof that a worker
is currently alive. A one-laptop run represents logical clients, not multiple
physical devices.

The views provide:

- **Overview:** attempt/commit counts, the latest saved model version, declared
  configuration, and measured accepted-reply work and traffic.
- **Attempts & clients:** accepted, rejected, and absent clients; actual work;
  coefficient, norm, and clipping details; timings and memory when recorded.
- **Quality:** saved evaluations and domain scores, separated by split. Run
  comparison displays the individual configuration identities and does not
  establish that the experiments were matched.
- **Ask a query:** lazy CPU-only loading of local model/common-A assets and an
  actual selected checkpoint, followed by twenty-candidate completion scoring.
- **Evidence:** configuration, checkpoint metadata, source-file SHA-256 hashes,
  event records, and downloadable JSON/Markdown reports.

## Recorded replay

Choose **Replay recorded events** and move the event-count slider. The console
uses only the selected prefix of the real event stream. Future summary metrics,
checkpoint metadata, and consolidated round files are withheld. Query inference
is disabled in replay because the latest saved checkpoint may belong to a later
event. Replay does not produce measurements or simulate training.

## Query demo

Prepare the cached model and common adapter using the model preparation command
in the project README. The default directories are `assets/model` and
`assets/adapter`; the query panel accepts other local paths. Runtime inference
does not download missing assets. Install training dependencies to enable it.

Select a checkpoint saved in the run directory (`checkpoint*.json`) or its
`checkpoints/` directory. The **Frozen base + zero B** option evaluates the
initial adapter state. Loading is deferred until **Score query** is pressed, and
the model runs on CPU. This avoids GPU allocation by the dashboard, although CPU
and RAM are still shared with other work on the laptop.

The router always chooses one of the twenty labels. It has no rejection model
for unrelated queries. Candidate scores are summed completion log probabilities
including EOS, not calibrated confidence. The displayed support responses are
team-authored application content in
[responses.py](../fedlora_edge/responses.py), not BANKING77-provided answers.
The demo has no bank account access and performs no banking actions.

## Report from the command line

The report reader has no optional dependencies:

```powershell
python tools/report_results.py runs/my-run
python tools/report_results.py runs/my-run --output runs/reports/my-run
```

This writes `report.json` and `report.md` by default. Each report includes the
complete saved configuration, a canonical configuration SHA-256 hash, exact
source-file hashes, evaluation sources/splits, attempt details, and measurement
coverage. A missing metric remains null or **Not recorded**. Review a report
before publishing it; configuration files can contain local paths or machine
details.

## File contract

The preferred run directory contains `config.json`, `events.jsonl`,
`round_results.json`, and `checkpoint.json`. Optional `summary.json`,
`evaluations.jsonl`, `evaluation.json`, `metrics.json`, and `results.json` are
also read when they contain recognized measurements. The reader reports corrupt
records and tolerates a partially appended final JSONL line; refresh after the
writer completes it.

Events use `event`, `run_id`, `timestamp`, `attempt_id`, and `version`. Supported
event names include `attempt_started`, `reply_accepted`, `reply_rejected`,
`round_result`, and `evaluation`. An evaluation carries `metrics` with fields
such as `accuracy`, `macro_f1`, `task_loss`, `count`, and `per_domain_accuracy`,
plus an explicit `split`. Accepted replies carry `telemetry` and separately
record raw delta bytes and serialized reply bytes. Missing telemetry is not
silently replaced by zeros or theoretical payload estimates.

Flower request and reply sizes are serialized protobuf measurements. They do
not include HTTP/TLS framing or retransmissions. Process interruption, deadline,
synchronization, and rejoin events are shown as recorded observations, with a
separate column for the terminal server acceptance decision.

Reader invariants are tested with:

```powershell
python -m unittest discover -s tests -p test_results.py -v
```
