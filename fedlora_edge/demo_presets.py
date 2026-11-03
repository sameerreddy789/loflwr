"""Read completed, published evidence for the presentation presets.

These views never start training or access raw evaluation queries.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("Live banking query", "Recorded recovery", "Experiment comparisons", "Custom run")
SCOPE_EXPORT_SHA256 = "71ae9de3898d61d6ad770d742210fd23dfd9d633320363c2aaa0b33bd8895fb9"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verified_export(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Published evidence changed. Restore the committed result before presenting it.")
    return json.loads(raw)


def load_recovery(layout="five", root=ROOT):
    if layout not in ("five", "three"):
        raise ValueError("Choose the three- or five-client recorded layout.")
    folder = Path(root) / "research/results/remaining-work-v1"
    name = f"{layout}-recovery.json"
    verification = read(folder / "verification.json")
    result = verified_export(folder / name, verification["runtime_exports"][name])
    if result["status"] != "completed" or len(result["rounds"]) != result["attempts"]:
        raise ValueError("Recovery evidence is incomplete.")
    previous = result["initial_head_hash"]
    version = 0
    for number, row in enumerate(result["rounds"], 1):
        accepted = row["accepted"]
        if (row["attempt_id"] != number or row["version_before"] != version
                or row["incoming_head_hash"] != previous
                or len(set(accepted)) != len(accepted)
                or not set(accepted) <= set(result["client_ids"])):
            raise ValueError("Recovery attempt identities disagree.")
        if row["status"] == "skipped":
            if (len(accepted) >= result["quorum"] or row["checkpoint_hash"] != previous
                    or row["version_after"] != version or row["coefficients"]
                    or row["below_quorum_retention_passed"] is not True):
                raise ValueError("Skipped attempt did not retain the checkpoint.")
        elif row["status"] == "committed":
            weights = row["coefficients"]
            if (len(accepted) < result["quorum"] or row["version_after"] != version + 1
                    or set(weights) != set(accepted)
                    or any(not math.isclose(v, 1/len(accepted)) for v in weights.values())
                    or row["all_accepted_clients_started_from_current_head"] is not True):
                raise ValueError("Committed attempt violates the recorded quorum/weight contract.")
        else:
            raise ValueError("Unknown recorded attempt status.")
        previous, version = row["checkpoint_hash"], row["version_after"]
    if previous != result["final_head_hash"] or version != result["committed_versions"]:
        raise ValueError("Recovery final checkpoint disagrees.")
    return result


def load_comparisons(root=ROOT):
    root = Path(root)
    verification = read(root / "research/results/remaining-work-v1/verification.json")
    matrix = verified_export(root / "research/results/encoder-matrix-v1/summary.json",
                             verification["matrix"]["summary_sha256"])
    if (matrix["status"] != "completed" or matrix["completed"] != 33
            or matrix["plan_hash"] != verification["matrix"]["plan_hash"]):
        raise ValueError("The declared three-seed matrix is incomplete or incompatible.")
    return matrix


def load_scope(root=ROOT):
    return verified_export(Path(root) / "research/results/scope-v1/negative-test-result.json",
                           SCOPE_EXPORT_SHA256)


def render_recovery(st):
    st.header("Recorded dropout and recovery")
    st.info("Recorded experiment on one laptop. This view replays measured results; it does not start training.")
    layout = st.selectbox("Recorded layout", ["Five clients", "Three clients"])
    result = load_recovery("five" if layout == "Five clients" else "three")
    names = {1: "All clients", 2: "One client drops", 3: "Client rejoins", 4: "Quorum missed", 5: "Clients return"}
    attempt = st.radio("Choose a recorded attempt", list(range(1, 6)),
                       format_func=lambda n: f"{n}. {names[n]}", horizontal=True)
    row = result["rounds"][attempt - 1]
    st.subheader(f"Attempt {attempt}: " + ("checkpoint kept" if row["status"] == "skipped" else "update committed"))
    st.write(f"**{len(row['accepted'])}/{len(result['client_ids'])} contributors** · "
             f"quorum {result['quorum']} · {row['collection_seconds']:.2f}s reply collection")
    st.write(f"Global version: **{row['version_before']} to {row['version_after']}**.")
    if row["status"] == "skipped":
        st.warning("Too few valid replies. The global checkpoint stayed unchanged.")
    elif attempt in (3, 5):
        st.success("Returning clients synchronized the current global head before contributing.")
    domains = (["Card lifecycle", "Card payments", "Transfers", "Cash withdrawals", "Verification"]
               if len(result["client_ids"]) == 5 else ["Cards", "Transfers and cash", "Verification"])
    st.dataframe([{"Client": c, "Local domain": domains[int(c)],
                   "Reply": "Accepted" if c in row["accepted"] else "Absent",
                   "Aggregation weight": f"{row['coefficients'][c]:.0%}" if c in row["coefficients"] else "No update committed",
                   "Steps in accepted reply": row["completed_optimizer_steps"].get(c)}
                  for c in result["client_ids"]], hide_index=True, width="stretch")
    st.caption("Absent replies do not imply zero work: a killed client may have completed partial training. Equal weights apply only when an update commits.")
    st.line_chart([{"Attempt": r["attempt_id"], "Accepted contributors": len(r["accepted"])}
                   for r in result["rounds"]], x="Attempt", y="Accepted contributors", color="#176b87")
    st.caption(f"Source: {result['run_id']}. Complete rehearsal: {result['wall_seconds']:.0f}s. "
               "Training collection has a 60s bound; startup and rejoin readiness are separate.")
    with st.expander("Checkpoint and transport evidence"):
        st.code(f"Incoming head: {row['incoming_head_hash']}\nSaved head: {row['checkpoint_hash']}")
        st.write("TLS and registered node identities were exercised. Client-local feature files were isolated. Physical multi-laptop testing remains pending.")
        st.download_button("Download recorded evidence", json.dumps(result, indent=2),
                           file_name=f"{result['run_id']}.json", mime="application/json")


def render_comparisons(st):
    matrix = load_comparisons()
    st.header("Measured experiment comparisons")
    st.info("Completed development experiments. The separate confirmed twenty-intent accuracy remains 94% (282/300).")
    st.write("33 completed runs across seeds 42, 43 and 44; 500 attempts per run. "
             "These comparisons use real head optimization with sequential aggregation.")
    st.subheader("When one client owns 90% of the data")
    labels = {"sample": "Sample weighted", "equal": "Equal weights", "equal_clipped": "Equal weights + clipping"}
    rows = []
    for key, label in labels.items():
        metrics = matrix["groups"][f"imbalanced-heterogeneous-{key}"]["final"]
        rows.append({"Aggregation": label, "Final validation accuracy (%)": metrics["accuracy"]["mean"]*100,
                     "Sample SD (points)": metrics["accuracy"]["sample_std"]*100,
                     "Worst domain (%)": metrics["worst_domain_accuracy"]["mean"]*100})
    st.vega_lite_chart({"data": {"values": rows}, "height": 160,
        "mark": {"type": "bar", "color": "#176b87"},
        "encoding": {
            "y": {"field": "Aggregation", "type": "nominal", "sort": list(labels.values()),
                  "axis": {"title": None, "labelLimit": 190}},
            "x": {"field": "Final validation accuracy (%)", "type": "quantitative",
                  "scale": {"domain": [0, 100]}},
            "tooltip": [{"field": "Aggregation"},
                        {"field": "Final validation accuracy (%)", "format": ".2f"}]}},
        width="stretch")
    st.dataframe(rows, hide_index=True, width="stretch", column_config={
        column: st.column_config.NumberColumn(format="%.2f") for column in rows[0] if column != "Aggregation"})
    st.caption("400 training queries, unequal work budgets, fixed attempt 500, reused 200-query validation. Most of the observed gain comes from changing the weighting.")
    st.subheader("The one-step verification client contributes useful knowledge")
    all_clients = matrix["groups"]["expanded-heterogeneous-equal"]["final"]["accuracy"]["values"]
    omitted = matrix["groups"]["expanded-heterogeneous-equal-omit4"]["final"]["accuracy"]["values"]
    st.dataframe([{"Seed": seed, "All clients (%)": a*100, "Without client 4 (%)": b*100,
                   "Benefit (percentage points)": round((a-b)*100, 2)}
                  for seed, a, b in zip((42, 43, 44), all_clients, omitted)], hide_index=True, width="stretch")
    st.caption("The control uses 2,760 training queries. Client 4 owns the only examples of four verification intents; this is a specific contribution under a one-step budget.")
    with st.expander("Experimental unsupported-query detector"):
        scope = load_scope()
        point = scope["comparisons"][-1]
        st.write(f"The fixed strictest margin falsely accepted **{point['original_margin_false_accepts']}/{scope['count']}** "
                 f"unsupported queries. Adding the separate scope detector reduced this to **{point['with_scope_false_accepts']}/{scope['count']}**.")
        st.write("This detector is not enabled in the live query view. Fresh supported-query coverage remains untested; this negative-only result is not overall accuracy.")
    st.caption(f"Matrix identity: {matrix['plan_hash']}")
