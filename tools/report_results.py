"""Generate a reviewable JSON/Markdown report from real saved run evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedlora_edge.results import build_report, load_run, report_markdown  # noqa: E402 - CLI path bootstrap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Directory containing a recorded run")
    parser.add_argument("--output", type=Path, help="Output prefix (default: RUN/report)")
    args = parser.parse_args(argv)
    if not args.run.is_dir():
        parser.error(f"Run directory does not exist: {args.run}")
    run = load_run(args.run)
    if not run.sources:
        parser.error("No recognized run evidence found; a report would contain no observations.")
    report = build_report(run)
    prefix = args.output or args.run / "report"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    # Never overwrite the evidence we just inspected.
    source_paths = {(run.path / source["file"]).resolve() for source in run.sources}
    if json_path.resolve() in source_paths or markdown_path.resolve() in source_paths:
        parser.error("Output would overwrite a source file. Choose a different --output prefix.")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(report_markdown(report), encoding="utf-8")
    print(f"{run.run_id}: {run.evidence_label}")
    print(json_path.resolve())
    print(markdown_path.resolve())
    for warning in run.warnings:
        print(f"Reader warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
