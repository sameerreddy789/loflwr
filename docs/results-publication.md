# Publish saved study measurements

`tools/publish_study.py` reads an existing frozen study and writes a new compact
report directory. It does not load a model, train, run inference, or access the
final test again. Use the original local evidence under `runs/` as the source.

Install the small reporting environment on Windows with Python 3.11:

```powershell
uv venv --python 3.11 .venv-report
uv pip install --python .venv-report/Scripts/python.exe -r requirements/reporting-windows-py311.txt
.venv-report/Scripts/python.exe tools/publish_study.py --study-dir runs/pilot-001 --output-dir runs/pilot-001-publication
```

The reporting pins were exported after a Windows Python 3.11.15 publication and
PNG/SVG rendering smoke check on 3 November 2026. This environment does not
validate Torch optimization, Flower execution, or physical-laptop recovery.
The existing laptop environment is
`C:\Users\samee\.venvs\loflwr-report\Scripts\python.exe`.

The output contains:

- `README.md`, a measured quality table, work/time/traffic table, and per-domain
  accuracy table with every declared run and its status.
- `summary.csv` and `per-domain-accuracy.csv`, retaining missing values as empty
  cells rather than zero.
- `publication.json`, the recorded study plan, plan and source-file hashes,
  compact configurations, original configuration-file hashes, checkpoint B
  hashes, measurement coverage, and reader warnings.
- `run-evidence/<key>.json`, an export from the existing evidence helper for each
  completed run. Raw queries, individual predictions, tensor states, and adapter
  schemas are omitted. Certificates and private node-key files are never read.
- `validation-task-loss.png/.svg` and `final-test-macro-f1.png/.svg`, using recorded
  measurements, accessible colors, and individual seed labels. SVG text remains
  editable. No confidence intervals or across-seed estimates are invented.

The command requires a new output directory and verifies the frozen plan hash.
It reads the standard `configs/<key>.json` and `runs/<key>/` layout beneath the
supplied study directory, so WSL evidence can be reported from Windows without
changing its original configuration paths or hashes. Recorded hashes identify
the original files, not the compact transformed JSON.

Final-test table values and bars require `completed` in `status.json`, agreement
between the frozen/runtime configurations and checkpoint identity, the declared
attempt horizon in both checkpoint and completion record, all attempt records,
and a test record with the same run, attempt, committed version, and B hash.
Missing or contradictory completion evidence becomes `incomplete_evidence`.
Unselected, failed, and unfinished runs stay visible; their test fields remain
missing. Validation points from incomplete runs remain visible with their status.
A calibration study may have completed validation runs but no eligible final test.

`PARTIAL STUDY` refers to the complete declared matrix, even if every selected
run finished. A single-seed label remains explicit. The tool does not establish
that methods had matched processed work, that a method improved quality, or that
deployment recovery succeeded. Check actual accepted work and its coverage before
making a comparison. Report unfavorable and failed runs alongside favorable ones.

Work comes from accepted federated replies or reported centralized blocks;
partial centralized work is included. Phase wall time sums all recorded child
phases and retry histories, including process startup and evaluation; it excludes
orchestrator gaps. Serialized request/reply bytes describe application payloads,
not network-wire traffic or initial A/model/data transfer. Missing telemetry is
not zero. Historical source hashes are retained even if current source files have
changed; the report lists those differences without certifying a rerun.

The interrupted `pilot-20261103` was used only to validate this reporting path.
It is superseded infrastructure evidence, not a completed method comparison.
Keep such checks in ignored `runs/` and attach this context explicitly:

```powershell
.venv-report/Scripts/python.exe tools/publish_study.py --study-dir runs/pilot-20261103 --output-dir runs/pilot-20261103-report-check --note "Interrupted infrastructure pilot, superseded by a new study identity. Reporting validation only; no completed method comparison."
```

Inspect the generated tables, figures, identities, and limitations before copying
a concrete report to a versioned research location. Never copy `assets/`, raw
checkpoints, dataset manifests containing queries, or `certificates/` into that
publication directory.
