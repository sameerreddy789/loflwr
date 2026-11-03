# Project utilities

Run these commands from the repository root with the appropriate Python environment. The [prepared-laptop guide](../docs/morning-demo.md) gives exact Windows/WSL paths; [requirements](../requirements/README.md) documents tested dependencies. Store assets in ignored `assets/`, outputs in ignored `runs/`, and private node credentials in ignored `certificates/`.

## Demo and evidence

| Tool | Purpose | Command |
| --- | --- | --- |
| [start_demo.ps1](start_demo.ps1) | Start the localhost dashboard with offline model settings | `.\tools\start_demo.ps1` |
| [report_results.py](report_results.py) | Export measured run evidence to JSON/Markdown without optional dependencies | `python tools/report_results.py runs/my-run` |
| [run_study.py](run_study.py) | Freeze/execute/resume the declared real-training study | `python tools/run_study.py --list` |

The dashboard inspects saved runs and performs checkpoint-backed CPU inference. It does not start training. Reports include configuration and source-file identities, and mark unrecorded metrics as missing. Review local paths and metadata before publishing. See the [dashboard](../docs/dashboard.md) and [study runner](../docs/study-runner.md) guides for the full workflows.

## Data, model, and runtime commands

These implementations live in the package and can run with `python -m` without installing console entry points:

```sh
python -m fedlora_edge.data --help
python -m fedlora_edge.training --help
python -m fedlora_edge.federation --help
```

| Module/command | Behavior |
| --- | --- |
| `fedlora_edge.data download`, `prepare`, `verify` | Cache the pinned BANKING77 source, freeze deterministic manifests, and verify hashes/counts/disjointness |
| `fedlora_edge.training download`, `prepare` | Cache the pinned model/tokenizer, then save common A and initial B with their identities |
| `fedlora_edge.training smoke` | Run real optimization and check frozen base/A, changing B, masking/EOS, and scoring invariants |
| `fedlora_edge.training train`, `evaluate`, `query` | Run local B-only training, completion-scored evaluation, or a single query |
| `fedlora_edge.federation simulate` | Run real Flower clients through a single Ray worker; use the WSL2 environment |
| `fedlora_edge.federation write-config` | Write the selected runtime configuration without starting Flower |
| `fedlora_edge.federation deploy-local` | Launch TLS/authenticated Flower SuperLink/SuperNodes on localhost; this is a same-host rehearsal |

The real federation CLI accepts work budgets, attempts, quorum, collection deadline, strategy, optional validation, omission, and process-interruption schedules. See `--help` for exact options. Default simulation is a one-attempt balanced five-client check with two steps per client; it is not the twenty-attempt heterogeneous-budget study. Training/evaluation use cached assets; only explicit download commands fetch them.

Use [run_study.py](run_study.py) for comparative measurements. Its plan fixes the matrix before execution, keeps incomplete runs visible, and limits final-test access to the complete declared twenty-attempt horizon. Use validation for calibration. The standalone `evaluate` command is an inspection tool and does not implement the study's access policy for you.

## Runtime probes

The following require the WSL2 Flower environment:

```sh
python tools/check_flower_simulation.py
python tools/check_flower_transport.py --run-id transport-check-new
```

[check_flower_simulation.py](check_flower_simulation.py) checks routing to five logical nodes through one Ray worker. Its output is `runs/flower-ray-routing-smoke/proof.json`; rerunning uses that same destination.

[check_flower_transport.py](check_flower_transport.py) launches real localhost TLS/authenticated Flower processes and checks a named FP32 array round trip. Use a new run ID for each invocation so its output and generated certificate directory are distinct. It records raw tensor and serialized reply bytes.

These probes do not train a language model. Neither proves physical multi-laptop deployment or live recovery. Use the real training runtime and record interruption, bounded collection, acceptance decisions, and fresh synchronization to establish those behaviors.

[classifier_reference.py](classifier_reference.py) provides a centralized TF-IDF/logistic-regression context baseline and requires scikit-learn. It does not exercise federated adapters. Use validation during calibration and keep its run identity and configuration with any published measurement.

[calibrate_persistent_worker.py](calibrate_persistent_worker.py) checks real-model cold/warm delta equivalence, interleaved client state reset, optimizer-process termination/replacement, and deadline cleanup. It uses isolated subprocesses rather than a complete Flower round. Choose a new `--run-id` when repeating it; the default identifies the original local calibration. The [deployment guide](../docs/deployment.md) separates localhost runtime checks from an untested manual LAN setup.

## Repository checks

```sh
python -m unittest discover -s tests -v
python tools/check_docs.py
```

The unit suite covers tensor state, aggregation, protocol/checkpoint contracts, data manifests, trainer fixtures, result readers, and frozen study plans. Optional Torch/Flower/Streamlit tests skip when dependencies are absent. Run the complete suite in the tested WSL2 environment and read the skip count; the dependency-free subset is useful but narrower.

[check_docs.py](check_docs.py) checks required dossier files, local file links in its configured root/research/skills/tools folders, and the four project skills' names/descriptions. It does not check external URLs, anchors, rendering, or every file under `docs/`; review new guide links separately.

## Source review

[review_sources.py](review_sources.py) uses the agent-reach Jina Reader route to collect primary-page titles, paper comments, access dates, and errors. Raw reader responses are temporary files outside the repository. [Committed metadata](../research/source-review.json) records the checks used for the dossier.

```sh
python tools/review_sources.py --date 2026-11-03 --workers 1 --labels ffa-lora fedprox
```

Use the actual access date. Reader quotas may return 429; record failures rather than turning an unavailable page into a verified claim. Metadata/abstract review does not reproduce a paper or establish model quality. See the [source register](../research/sources.md) for review depth.
