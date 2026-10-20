# Project utilities

These utilities support the current documentation/scaffolding delivery. Training, deployment, and dashboard commands remain implementation work; there are no runnable Flower applications in this delivery.

## Documentation checks

From the repository root with Python 3.11+:

```powershell
.\.venv\Scripts\python.exe tools\check_docs.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

[check_docs.py](check_docs.py) checks required dossier files, local Markdown file-link targets, and the four skills' required names/descriptions. It does not verify external URLs, anchors, or rendered appearance. The tests cover the initial tensor-state helpers, not model learning.

## Source review

[review_sources.py](review_sources.py) uses the agent-reach Jina Reader route to collect primary-page titles, paper comments, access dates, and errors. Raw reader responses are temporary files outside the repository. [The committed metadata](../research/source-review.json) records the checks used for this dossier.

For a small review batch:

```powershell
.\.venv\Scripts\python.exe tools\review_sources.py --date 2026-10-20 --workers 1 --labels ffa-lora fedprox
```

Use the actual access date in your timezone. Reader quotas may return 429; record that limit and supplement through verified publisher APIs or a later bounded retry. Do not turn an unavailable page into a verified claim. The script preserves supplemental metadata when updating a subset of sources.

Metadata/abstract review does not reproduce a paper, execute a repository, or verify model quality. See [sources.md](../research/sources.md) for review depth and source-specific limitations.

## Utilities still to implement

The later training delivery needs deterministic dataset preparation, model/common-A caching, environment locks, local training smoke checks, matched experiment runners, run-report generation, and offline deployment bundles. Their contracts are described in [system-design.md](../research/system-design.md), [evaluation.md](../research/evaluation.md), and [roadmap.md](../research/roadmap.md).

Place public model/data caches in ignored `assets/`, outputs in ignored `runs/`, and private node credentials in ignored `certificates/`. The old `tools/reference/` directory contains no successfully preserved legacy scripts in this delivery.
