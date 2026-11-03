"""Read-only Streamlit console for recorded FedLoRA-Edge runs and CPU scoring.

Launch from the repository root:
    python -m streamlit run fedlora_edge/dashboard.py -- --runs-dir runs
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import threading
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedlora_edge.results import (  # noqa: E402 - runnable source path is bootstrapped above
    RunEvidence, attempt_rows, build_report, client_rows, discover_runs, first,
    load_run, number, report_markdown, work_totals,
)


STYLE = """
<style>
:root { --edge-ink: #243447; --edge-muted: #52697c; --edge-blue: #176b87;
        --edge-canvas: #f5f7fa; --edge-line: #d4dee6; --edge-success: #28634d; }
.stApp { background: var(--edge-canvas); color: var(--edge-ink); }
[data-testid="stSidebar"] { background: #e9eff4; }
.block-container { max-width: 1480px; padding-top: 2rem; padding-bottom: 3rem; }
html, body, [class*="css"], [data-testid="stMarkdownContainer"] {
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
}
h1 { font-size: 2rem !important; font-weight: 650 !important; letter-spacing: -.025em; }
h2 { font-size: 1.4rem !important; font-weight: 650 !important; margin-top: .7rem !important; }
h3 { font-size: 1.12rem !important; font-weight: 650 !important; }
p { max-width: 75ch; }
code, [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
[data-testid="stMetricLabel"] { color: var(--edge-muted); }
[data-testid="stTabs"] button { font-weight: 600; }
button:focus-visible, a:focus-visible, input:focus-visible, textarea:focus-visible {
    outline: 3px solid #176b87 !important; outline-offset: 3px;
}
::selection { background: #c5e6ed; color: #183c4b; }
input, textarea { caret-color: #176b87; }
a { text-underline-offset: .2em; }
.run-line { color: #52697c; font-size: .88rem; overflow-wrap: anywhere; margin-bottom: 1.4rem; }
.attempt-ledger { display: flex; flex-wrap: wrap; gap: .55rem; margin: .65rem 0 1rem; }
.attempt-entry { padding: .65rem .8rem; background: #fff; border: 1px solid #d4dee6;
    border-radius: 8px; min-width: 112px; line-height: 1.6; font-variant-numeric: tabular-nums; }
.attempt-entry strong { display: block; color: #243447; font-size: .86rem; }
.attempt-entry span { display: block; font-size: .82rem; }
.attempt-entry.committed { background: #e9f3ef; border-color: #aacaba; color: #22583f; }
.attempt-entry.skipped { background: #fff3df; border-color: #dac29a; color: #755018; }
.attempt-entry.collecting { background: #e9f2f7; border-color: #a8c6d5; color: #195a75; }
.response-text { background: #fff; border: 1px solid #d4dee6; border-radius: 12px;
    padding: 1.2rem 1.35rem; color: #243447; line-height: 1.65; max-width: 75ch; }
@media (max-width: 640px) {
    .block-container { padding: 1.1rem 1rem 2rem; }
    h1 { font-size: 1.65rem !important; }
    .attempt-entry { min-width: 96px; padding: .5rem .6rem; }
}
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
</style>
"""


def _format(value: Any, *, percent: bool = False, decimals: int = 2) -> str:
    numeric = number(value)
    if numeric is None:
        return "—"
    if percent:
        return f"{numeric:.1%}"
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.{decimals}f}"


def _bytes(value: Any) -> str:
    numeric = number(value)
    if numeric is None:
        return "—"
    if numeric < 1024:
        return f"{numeric:,.0f} B"
    if numeric < 1024 ** 2:
        return f"{numeric / 1024:,.1f} KiB"
    return f"{numeric / 1024 ** 2:,.2f} MiB"


def _table(st: Any, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if rows:
        if columns:
            rows = [{key: row.get(key) for key in columns} for row in rows]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _ledger(st: Any, rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No attempts recorded. Start a run, then refresh to inspect its saved events.")
        return
    entries = []
    for row in rows[-30:]:
        status = str(row.get("Status", "Recorded"))
        style = status.lower() if status.lower() in {"committed", "skipped", "collecting"} else ""
        version = row.get("Version after")
        label = f"v{version}" if version is not None else "awaiting result"
        accepted = row.get("Accepted")
        count = f"{accepted} accepted" if accepted is not None else "collection open"
        entries.append(
            f'<div class="attempt-entry {style}"><strong>Attempt {html.escape(str(row.get("Attempt")))}</strong>'
            f'<span>{html.escape(status)} · {html.escape(label)}</span><span>{html.escape(count)}</span></div>'
        )
    st.markdown('<div class="attempt-ledger">' + "".join(entries) + "</div>", unsafe_allow_html=True)
    if len(rows) > 30:
        st.caption("Showing the latest 30 attempts. The complete ledger is in Attempts & clients.")


def _overview(st: Any, run: RunEvidence) -> None:
    rows = attempt_rows(run)
    committed = sum(row.get("Status") == "Committed" for row in rows)
    skipped = sum(row.get("Status") == "Skipped" for row in rows)
    version = first(run.checkpoint, "version")
    if version is None and run.rounds:
        version = first(run.latest_round, "version_after", "version")
    columns = st.columns(4)
    columns[0].metric("Attempts recorded", len(rows))
    columns[1].metric("Committed", committed)
    columns[2].metric("Skipped", skipped)
    columns[3].metric("Current model version", _format(version))
    st.subheader("Attempt ledger")
    _ledger(st, rows)
    st.caption("Each attempt can commit one new B checkpoint or leave the current version unchanged.")
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Run configuration")
        identity = run.config.get("identity") or run.checkpoint.get("identity") or {}
        model = first(identity, "model_id", "base_model_id", "model") if isinstance(identity, dict) else None
        _table(st, [
            {"Setting": "Strategy", "Recorded value": run.config.get("strategy")},
            {"Setting": "Seed", "Recorded value": _format(run.config.get("seed"))},
            {"Setting": "Quorum", "Recorded value": _format(run.config.get("quorum"))},
            {"Setting": "Collection deadline (s)", "Recorded value": _format(run.config.get("deadline_seconds"))},
            {"Setting": "Model", "Recorded value": model},
        ])
        st.caption("One-laptop runs describe logical clients. Physical multi-device deployment requires separate evidence.")
    with right:
        st.subheader("Latest quality measurement")
        if not run.evaluations:
            st.info("No model evaluation recorded. Protocol events do not establish learning or intent quality.")
        else:
            latest = run.latest_evaluation
            metrics = latest["metrics"]
            first_col, second_col = st.columns(2)
            first_col.metric("Macro F1", _format(metrics.get("macro_f1"), percent=True))
            second_col.metric("Accuracy", _format(metrics.get("accuracy"), percent=True))
            st.caption(f"Split: {latest['split']} · source: {latest['source']} · count: {_format(metrics.get('count'))}")
    st.subheader("Recorded work and traffic")
    totals = work_totals(run)
    columns = st.columns(5)
    columns[0].metric("Completed optimizer steps", _format(totals.get("optimizer_steps")))
    columns[1].metric("Processed tokens", _format(totals.get("processed_tokens")))
    columns[2].metric("Raw delta payload", _bytes(totals.get("raw_delta_bytes")))
    columns[3].metric("Serialized replies", _bytes(totals.get("serialized_reply_bytes")))
    columns[4].metric("Serialized requests", _bytes(totals.get("serialized_request_bytes")))
    st.caption("Work covers accepted replies or reported centralized blocks. Raw delta bytes cover accepted replies; serialized reply coverage is shown below. Unrecorded values appear as —.")
    st.caption(f"Serialized reply traffic: {totals['serialized_reply_bytes_scope']}. Flower sizes exclude HTTP/TLS framing and retransmissions.")
    with st.expander("Measurement coverage"):
        st.json(totals)


def _attempts(st: Any, run: RunEvidence) -> None:
    rows = attempt_rows(run)
    if not rows:
        st.info("No federated attempts have been recorded in this run.")
        return
    st.subheader("Attempts and checkpoint commits")
    _table(st, rows)
    selected = st.selectbox("Inspect attempt", [row["Attempt"] for row in rows], index=len(rows) - 1)
    clients = client_rows(run, selected)
    result = next((item for item in run.rounds if str(item.get("attempt_id")) == str(selected)), {})
    st.subheader("Client participation and work")
    _table(st, clients, ["Client", "Domain", "Status", "Reason", "Declared examples", "Step budget",
                         "Completed steps", "Processed examples", "Processed tokens", "Training seconds", "Queue seconds"])
    st.caption("Accepted means the reply passed server validation. A skipped attempt can still contain accepted replies below quorum.")
    with st.expander("Process and synchronization evidence"):
        _table(st, clients, ["Client", "Process event", "Process ID", "Synchronized version", "Rejoined"])
        st.caption("These are recorded process events, not a live worker health check.")
    st.subheader("Influence and clipping")
    _table(st, clients, ["Client", "Status", "Coefficient", "Delta norm", "Clip factor", "Clipped norm"])
    aggregation = result.get("aggregation") or {}
    cols = st.columns(3)
    cols[0].metric("Median clip threshold", _format(aggregation.get("clip_threshold"), decimals=6))
    cols[1].metric("Aggregate norm", _format(aggregation.get("aggregate_norm"), decimals=6))
    cols[2].metric("Collection seconds", _format(result.get("collection_seconds")))
    st.caption("Coefficients apply to accepted survivors. Missing coefficients on a skipped attempt do not mean zero influence was measured.")
    st.subheader("Memory and reply traffic")
    _table(st, clients, ["Client", "Peak RSS bytes", "Peak GPU allocated bytes", "Peak GPU reserved bytes",
                         "Raw delta bytes", "Serialized reply bytes"])
    with st.expander("Complete attempt record"):
        st.json(result or {"status": "No final attempt record yet"})


def _quality(st: Any, run: RunEvidence, run_paths: list[Path], *, replay: bool = False) -> None:
    st.subheader("Held-out quality")
    if not run.evaluations:
        st.info("No held-out evaluation exists in this run's saved evidence. Run the declared evaluator to populate this view.")
    else:
        splits = list(dict.fromkeys(item["split"] for item in run.evaluations))
        split = st.selectbox("Evaluation split", splits)
        evaluations = [item for item in run.evaluations if item["split"] == split]
        rows = []
        for index, item in enumerate(evaluations):
            metrics = item["metrics"]
            rows.append({
                "Measurement": index + 1, "Version": item.get("version"),
                "Attempt": item.get("attempt_id"), "Count": metrics.get("count"),
                "Accuracy": metrics.get("accuracy"), "Macro F1": metrics.get("macro_f1"),
                "Task loss": first(metrics, "task_loss", "validation_loss", "loss"),
                "Worst domain": metrics.get("worst_domain_accuracy"),
                "Domain gap": metrics.get("domain_gap"), "Source": item["source"],
            })
        _table(st, rows)
        if len(rows) > 1 and any(number(row.get("Accuracy")) is not None for row in rows):
            st.line_chart(rows, x="Measurement", y=["Accuracy", "Macro F1"], color=["#176b87", "#8c5790"])
            st.caption("Chronological saved measurements for this split. Missing values remain gaps.")
        latest = evaluations[-1]
        domains = latest["metrics"].get("per_domain_accuracy")
        if isinstance(domains, dict) and domains:
            st.subheader("Latest per-domain accuracy")
            domain_rows = [{"Domain": domain.replace("_", " ").capitalize(), "Accuracy": value}
                           for domain, value in domains.items()]
            _table(st, domain_rows)
            st.bar_chart(domain_rows, x="Domain", y="Accuracy", color="#176b87", horizontal=True)
        st.caption("Accuracy and macro F1 are fractions on a 0–1 scale. Model versions and attempts are separate counters.")
    if replay:
        st.caption("Switch to Latest saved state to compare complete run results.")
        return
    st.subheader("Compare recorded runs")
    st.caption("This table makes no claim that runs are matched. Check manifests, initialization, budgets, horizon, split, seed, and scoring rule in their evidence.")
    selected = st.multiselect("Runs to compare", run_paths, default=[run.path] if run.path in run_paths else [],
                              format_func=lambda path: str(path.relative_to(path.parent.parent)))
    comparison = []
    for path in selected:
        candidate = load_run(path)
        latest = candidate.latest_evaluation
        metrics = latest.get("metrics", {})
        comparison.append({
            "Run": candidate.run_id, "Strategy": candidate.config.get("strategy"),
            "Seed": candidate.config.get("seed"), "Evidence": candidate.evidence_label,
            "Split": latest.get("split"), "Count": metrics.get("count"),
            "Macro F1": metrics.get("macro_f1"), "Accuracy": metrics.get("accuracy"),
            "Task loss": first(metrics, "task_loss", "validation_loss", "loss"),
            "Config SHA-256": candidate.config_hash,
        })
    _table(st, comparison)


def _show_query_result(st: Any, scored: dict[str, Any]) -> None:
    from fedlora_edge.responses import response_for

    result = scored["result"]
    intent = result["intent"]
    st.caption(f"Last scored query: {scored['query']}")
    st.write(f"**Selected intent:** `{intent}`")
    st.markdown(f'<div class="response-text">{html.escape(response_for(intent))}</div>', unsafe_allow_html=True)
    st.caption(f"Team-authored demonstration response · CPU · {scored['elapsed']:.2f} s including any model load")
    st.caption(f"Checkpoint: {scored['checkpoint'] or 'frozen base + zero B'} · scoring rule: {result.get('scoring_rule', 'not reported')}")
    if result.get("b_hash"):
        st.caption(f"Scored B hash: {result['b_hash']}")
    if result.get("truncated_query"):
        st.warning("The query was truncated to the configured token limit before scoring.")
    scores = result.get("scores", {})
    if isinstance(scores, dict):
        with st.expander("Candidate completion scores"):
            st.caption("Summed completion log probabilities including EOS. Higher is preferred; these scores and their gaps are not calibrated confidence.")
            _table(st, [{"Intent": label, "Log probability": value}
                        for label, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)])


def _query(st: Any, run: RunEvidence) -> None:
    st.subheader("Try the twenty-intent router")
    st.write("Score a banking query against the twenty permitted intent labels, then view its team-authored support response.")
    st.caption("This closed-set demo always chooses a permitted label, including for out-of-scope text. It does not access a bank account or perform banking actions.")
    checkpoint_paths = sorted(run.path.glob("checkpoint*.json"))
    checkpoint_paths += sorted((run.path / "checkpoints").glob("*.json"))
    options: list[Path | None] = [None, *checkpoint_paths]
    checkpoint = st.selectbox("Inference checkpoint", options,
                              index=1 if checkpoint_paths else 0,
                              format_func=lambda path: "Frozen base + zero B" if path is None else str(path.relative_to(run.path)))
    checkpoint_revision = checkpoint.stat().st_mtime_ns if checkpoint else 0
    result_key = f"query-result:{run.path}:{checkpoint}:{checkpoint_revision}"
    with st.expander("Local inference settings"):
        model_dir = st.text_input("Cached model directory", str(ROOT / "assets" / "model"))
        adapter_dir = st.text_input("Common adapter directory", str(ROOT / "assets" / "adapter"))
        st.caption("Loaded only when you score a query. Inference runs on CPU and never downloads assets.")
    with st.form("query_form"):
        query = st.text_area("Banking query", placeholder="My transfer is still pending", max_chars=2000)
        submitted = st.form_submit_button("Score query", type="primary")
    if not submitted:
        if result_key in st.session_state:
            _show_query_result(st, st.session_state[result_key])
        return
    if not query.strip():
        st.warning("Enter a query before scoring.")
        return
    if not Path(model_dir).is_dir() or not Path(adapter_dir).is_dir():
        st.error("The cached model or common adapter directory is missing. Prepare both assets, then enter their local paths above.")
        return
    try:
        import time

        @st.cache_resource(max_entries=1, show_spinner=False)
        def load_engine(model_path: str, adapter_path: str, checkpoint_path: str,
                        modified_ns: int, training_json: str) -> tuple[Any, threading.Lock]:
            from fedlora_edge.training import LocalTrainer, TrainingConfig, load_b_state

            settings = json.loads(training_json)
            allowed = {field.name for field in fields(TrainingConfig)}
            config = TrainingConfig(**{key: value for key, value in settings.items() if key in allowed})
            trainer = LocalTrainer(asset_dir=model_path, adapter_dir=adapter_path,
                                   device="cpu", config=config)
            if checkpoint_path:
                trainer.set_b_state(load_b_state(checkpoint_path, expected_identity=trainer.identity))
            return trainer, threading.Lock()

        runtime = run.config.get("runtime") or {}
        runtime = runtime if isinstance(runtime, dict) else {}
        settings = run.config.get("training_config", run.config.get("training", runtime.get("training_config", {}))) or {}
        checkpoint_path = str(checkpoint.resolve()) if checkpoint else ""
        modified_ns = checkpoint.stat().st_mtime_ns if checkpoint else 0
        with st.spinner("Loading local CPU model and scoring the twenty completions…"):
            start = time.perf_counter()
            trainer, lock = load_engine(model_dir, adapter_dir, checkpoint_path, modified_ns,
                                        json.dumps(settings, sort_keys=True))
            with lock:
                result = trainer.score(query.strip())
            elapsed = time.perf_counter() - start
        st.session_state[result_key] = {
            "result": result, "elapsed": elapsed, "checkpoint": checkpoint_path, "query": query.strip(),
        }
        _show_query_result(st, st.session_state[result_key])
    except Exception as exc:
        st.error(f"CPU inference could not complete: {type(exc).__name__}: {exc}")
        st.caption("Check the cached assets, checkpoint identity, and training dependencies. Run inspection remains available.")


def _evidence(st: Any, run: RunEvidence) -> None:
    st.subheader("Evidence and reproducibility")
    st.write(f"Run ID: `{run.run_id}`")
    st.write(f"Configuration SHA-256: `{run.config_hash or 'not recorded'}`")
    st.caption("The configuration hash above is computed from the saved configuration's canonical JSON. Any producer-recorded identity remains in the source data below.")
    _table(st, run.sources)
    report = build_report(run)
    left, right = st.columns(2)
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in run.run_id)
    left.download_button("Download JSON report", json.dumps(report, indent=2, ensure_ascii=False),
                         file_name=f"{safe_name}-report.json", mime="application/json")
    right.download_button("Download Markdown report", report_markdown(report),
                          file_name=f"{safe_name}-report.md", mime="text/markdown")
    with st.expander("Full configuration", expanded=False):
        st.json(run.config)
    with st.expander("Checkpoint identity and metadata"):
        st.json(run.checkpoint or {"status": "No checkpoint metadata available in this view"})
    with st.expander("Saved summary"):
        st.json(run.summary)
    with st.expander("Recorded events"):
        st.caption("The most recent 100 records in this view. The complete JSONL stays in the run directory.")
        st.json(run.events[-100:])


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runs-dir", default=str(ROOT / "runs"))
    args, _ = parser.parse_known_args()
    import streamlit as st

    st.set_page_config(page_title="FedLoRA-Edge · Experiment console", page_icon=None,
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(STYLE, unsafe_allow_html=True)
    with st.sidebar:
        st.title("FedLoRA-Edge")
        st.caption("Shared frozen A · B-only updates")
        runs_root = st.text_input("Runs directory", value=args.runs_dir)
        try:
            run_paths = discover_runs(runs_root)
        except OSError as exc:
            st.error(f"Could not inspect the runs directory: {exc}")
            run_paths = []
        if not run_paths:
            st.info("No saved runs found in this directory.")
            st.button("Refresh files", use_container_width=True)
        else:
            root = Path(runs_root).expanduser().resolve()
            run_path = st.selectbox("Recorded run", run_paths, key="recorded_run",
                                   format_func=lambda path: str(path.relative_to(root)) or path.name)
            mode = st.radio("View", ["Latest saved state", "Replay recorded events"])
            watch = st.toggle("Refresh every 5 seconds", value=False,
                               disabled=mode == "Replay recorded events")
            st.button("Refresh files", use_container_width=True)
        st.divider()
        st.caption("Read-only experiment evidence. Training is started from the command line.")
        st.caption("SmolLM2-135M-Instruct · 20 BANKING77 intents")
    st.title("Experiment console")
    if not run_paths:
        st.write("No experiment evidence yet.")
        st.info("Start a recorded experiment or select an existing runs directory. This console will show attempts, client work, held-out quality, and checkpoint-backed query results as those files appear.")
        st.subheader("What a run provides")
        _table(st, [
            {"File": "config.json", "Evidence": "Run ID, seed, strategy, work budgets, and model/data identities"},
            {"File": "events.jsonl", "Evidence": "Actual attempts, replies, validation, and evaluations"},
            {"File": "round_results.json", "Evidence": "Commit/skip decisions, coefficients, norms, and timing"},
            {"File": "checkpoint.json", "Evidence": "Saved global B state and its identity for inference"},
        ])
        st.caption("No metrics, clients, or success states are generated for this empty view.")
        return
    initial = load_run(run_path)
    event_limit = None
    if mode == "Replay recorded events":
        st.warning("Replay of saved events. This view does not run training or simulate new measurements.")
        if initial.events:
            event_limit = st.slider("Recorded events visible", 0, len(initial.events), len(initial.events))
        else:
            st.info("This run has no event stream to replay.")
            event_limit = 0

    @st.fragment(run_every=5 if watch and mode == "Latest saved state" else None)
    def render() -> None:
        run = load_run(run_path, event_limit=event_limit)
        latest_timestamp = run.events[-1].get("timestamp", "not recorded") if run.events else "not recorded"
        label = "Watching file updates" if watch and mode == "Latest saved state" else "Recorded evidence"
        st.markdown(f'<div class="run-line">{html.escape(run.run_id)} · {html.escape(label)} · '
                    f'{html.escape(run.evidence_label)}<br>Latest visible event: {html.escape(str(latest_timestamp))}</div>',
                    unsafe_allow_html=True)
        if run.warnings:
            with st.expander(f"Reader notices ({len(run.warnings)})", expanded=True):
                for warning in run.warnings:
                    st.warning(warning)
        overview, attempts, quality, query, evidence = st.tabs(
            ["Overview", "Attempts & clients", "Quality", "Ask a query", "Evidence"]
        )
        with overview:
            _overview(st, run)
        with attempts:
            _attempts(st, run)
        with quality:
            _quality(st, run, run_paths, replay=event_limit is not None)
        with query:
            if event_limit is not None:
                st.info("Switch to Latest saved state to score a query with the current checkpoint. Replay only shows recorded events.")
            else:
                _query(st, run)
        with evidence:
            _evidence(st, run)
    render()


if __name__ == "__main__":
    main()
