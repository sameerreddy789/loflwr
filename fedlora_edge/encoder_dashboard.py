"""Local recorded-evidence console and live offline MPNet/head query demo."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedlora_edge.dashboard import STYLE
from fedlora_edge.data import DOMAINS
from fedlora_edge.encoder_inference import EncoderPredictor, load_head_checkpoint


def load_evidence(run_dir):
    directory = Path(run_dir)
    for name in ("final_results.json", "summary.json"):
        path = directory / name
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            selected = result.get("selected", {})
            if selected.get("split") != "validation" or not selected.get("metrics"):
                raise ValueError("This console requires a completed validation-scored encoder run.")
            saved = load_head_checkpoint(directory / "selected")
            expected = selected.get("state_hash") or selected["metrics"].get("head_hash")
            if saved["b_hash"] != expected or saved["version"] != selected["version"]:
                raise ValueError("Evidence and selected checkpoint disagree.")
            return result, saved
    raise FileNotFoundError("No completed encoder evidence here. Choose a finished run directory.")


def confirmation_for(saved):
    """Read published aggregate evidence only, never confirmation queries."""
    path = ROOT / "research/results/encoder-v2/confirmation-result.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("status") == "completed"
            and result.get("split") == "previously_unscored_confirmation"
            and result.get("head_hash") == saved["b_hash"]
            and result.get("config_hash") == saved["config_hash"]):
        return result
    return None


def load_margin_threshold(policy_file, saved):
    """Bind the confirmed model's policy to its pre-evaluation lock."""
    import math
    policy = json.loads(Path(policy_file).read_text(encoding="utf-8"))
    if policy.get("head_hash") != saved["b_hash"] or policy.get("run_id") != saved["run_id"]:
        raise ValueError("Margin policy belongs to a different checkpoint or run.")
    if (policy.get("kind") != "validation_calibrated_margin_policies"
            or policy.get("split") != "reused_validation"
            or not policy.get("manifest_hash") or not policy.get("validation_records_hash")):
        raise ValueError("Margin policy lacks validation provenance.")
    plan = json.loads((ROOT / "research/results/encoder-v2/confirmation-plan.json").read_text(encoding="utf-8"))
    if saved["b_hash"] == plan["head_hash"] and policy != plan["selective_policies"]:
        raise ValueError("Margin policy differs from the frozen confirmation policy.")
    point = next(p for p in policy["policies"] if p["target_validation_accuracy"] == 1.)
    threshold = point["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid margin threshold.")
    return threshold


def main():
    import streamlit as st
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(ROOT / "runs/encoder-flower-balanced-v2"))
    parser.add_argument("--model-dir", default=str(ROOT / "assets/reference-mpnet"))
    parser.add_argument("--policy-file", default="")
    args, _ = parser.parse_known_args()
    st.set_page_config(page_title="FedLoRA-Edge | Banking demo", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("Banking support, learned across clients")
    st.write("Try the shared intent classifier, then inspect the training evidence behind it.")
    with st.sidebar:
        st.header("Experiment")
        run_dir = st.text_input("Completed run directory", args.run_dir)
        model_dir = st.text_input("Cached MPNet directory", args.model_dir)
        policy_file = st.text_input("Margin policy file (optional)", args.policy_file)
        st.button("Reload evidence")
        st.caption("Inference runs locally. The selected model uses a frozen encoder and a trained classifier head.")
    try:
        evidence, saved = load_evidence(run_dir)
        confirmation = confirmation_for(saved)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        st.error(str(exc))
        return
    run_id = saved["run_id"]
    metrics = evidence["selected"]["metrics"]
    margin_threshold = None
    if policy_file:
        try:
            margin_threshold = load_margin_threshold(policy_file, saved)
            st.sidebar.caption("Clarification mode: low-margin predictions ask for more detail. The threshold was chosen on validation; future accuracy is not guaranteed.")
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            st.error(f"Cannot use margin policy: {exc}")
            return
    st.caption(f"{run_id} · selected version {saved['version']} · recorded training, live query inference")
    if confirmation:
        st.write(f"**Independent confirmation: {confirmation['metrics']['accuracy']:.1%} accuracy** "
                 f"({confirmation['correct']}/{confirmation['metrics']['count']}).")
    query_tab, evidence_tab, scope_tab = st.tabs(["Ask a query", "Training evidence", "Supported intents"])
    with query_tab:
        st.subheader("What do you need help with?")
        st.write("English questions about card delivery, payments, transfers, cash withdrawals, or identity verification.")
        with st.form("banking-query"):
            query = st.text_area("Banking query", placeholder="My card payment is still pending.", max_chars=4000)
            submitted = st.form_submit_button("Classify query", type="primary")
        if submitted:
            if not query.strip():
                st.warning("Enter a banking query before classifying it.")
            else:
                try:
                    checkpoint_dir = Path(run_dir) / "selected"
                    signature = hashlib.sha256((checkpoint_dir / "checkpoint.json").read_bytes()).hexdigest()
                    @st.cache_resource(show_spinner=False, max_entries=2)
                    def predictor(assets, checkpoint, signature):
                        return EncoderPredictor(assets, checkpoint)
                    with st.spinner("Loading local model and classifying your query…"):
                        result = predictor(model_dir, str(checkpoint_dir), signature).query(query)
                    if result["head_hash"] != saved["b_hash"]:
                        raise ValueError("Checkpoint changed during inference. Reload evidence and try again.")
                    gap = result["candidates"][0]["Uncalibrated logit"] - result["candidates"][1]["Uncalibrated logit"]
                    needs_details = margin_threshold is not None and gap < margin_threshold
                    if needs_details:
                        st.subheader("More detail needed")
                        st.write("Please name the payment method or the exact issue, then classify the revised question. The topics below are suggestions; no support response has been selected.")
                    else:
                        st.subheader(result["intent"].replace("_", " ").capitalize())
                        st.write(result["response"])
                    st.caption(f"{result['seconds']:.2f}s inference · " + ("no automatic support response" if needs_details else "team-authored support response"))
                    st.dataframe(result["candidates"], hide_index=True, width="stretch")
                    st.caption("Logits rank the allowed intents. They are not calibrated probabilities.")
                    if result["truncated"]:
                        st.warning("The query exceeded 384 tokens; only its beginning was classified.")
                except (OSError, ValueError, RuntimeError, ImportError) as exc:
                    st.error(f"Local inference could not start: {exc}")
        st.caption("The model ranks 20 intents. Optional clarification mode declines low-margin answers; it cannot detect every unsupported request or perform account actions.")
    with evidence_tab:
        if confirmation:
            st.subheader("Locked confirmation evaluation")
            st.write(f"{confirmation['correct']} correct, {confirmation['errors']} errors across "
                     f"{confirmation['metrics']['count']} previously unscored queries. "
                     f"Macro F1: {confirmation['metrics']['macro_f1']:.4f}.")
            st.caption("The model and every threshold below were fixed before these queries were scored. No rule reached 99% on this confirmation set.")
            st.dataframe([{"Fixed margin threshold": p["threshold"],
                "Accuracy on answered": f"{p['confirmation_accuracy_on_answered']:.2%}",
                "Coverage": f"{p['confirmation_coverage']:.2%}", "Declined": p["confirmation_abstained"]}
                for p in confirmation.get("selective_results", [])], hide_index=True, width="stretch")
        st.subheader("Measured validation result")
        st.write(f"**{metrics['accuracy']:.1%} accuracy** on {metrics['count']} development queries. "
                 f"Macro F1: {metrics['macro_f1']:.4f}. Worst domain: {metrics['worst_domain_accuracy']:.1%}.")
        st.caption("The validation set was reused for model selection. This is not an untouched-test estimate or a 99% accuracy claim.")
        history = evidence.get("evaluations", evidence.get("history", []))
        rows = [{"Attempt": r["attempt_id"], "Accuracy": r["metrics"]["accuracy"],
                 "Worst domain": r["metrics"]["worst_domain_accuracy"]} for r in history]
        if rows:
            st.line_chart(rows, x="Attempt", y=["Accuracy", "Worst domain"], color=["#176b87", "#28634d"])
        st.dataframe([{"Domain": k, "Accuracy": f"{v:.1%}"} for k, v in metrics["per_domain_accuracy"].items()],
                     hide_index=True, width="stretch")
        rounds = evidence.get("rounds", [])
        if rounds:
            st.subheader("Recorded client participation")
            number = st.slider("Attempt to inspect", 1, len(rounds), len(rounds)) if len(rounds) > 1 else 1
            row = rounds[number - 1]
            coefficients = (row.get("aggregation") or {}).get("coefficients", {})
            clip_factors = (row.get("aggregation") or {}).get("clip_factors", {})
            clipped_norms = (row.get("aggregation") or {}).get("clipped_norms", {})
            clients = row.get("clients", {})
            st.write(f"{row['status'].capitalize()} · {len(row['accepted'])} accepted · "
                     f"{row.get('collection_seconds', 0):.2f}s collection")
            st.dataframe([{"Client": c, "Status": "Accepted" if c in row["accepted"] else "Absent or rejected",
                           "Weight": coefficients.get(c, 0), "Clip factor": clip_factors.get(c),
                           "Clipped delta norm": clipped_norms.get(c),
                           "Steps": clients.get(c, {}).get("optimizer_steps", 0)}
                          for c in row.get("selected", [str(i) for i in range(5)])], hide_index=True, width="stretch")
        st.write("Each head update contains 15,380 FP32 parameters: 61,520 raw bytes per direction. Transport adds overhead.")
        with st.expander("Model identity and limits"):
            st.code(f"Run: {run_id}\nHead: {saved['b_hash']}\nConfig: {saved['config_hash']}")
            st.write("Five logical clients share this laptop. Frozen features were prepared centrally from public data before partitioning. Head training is distinct from the separate LoRA experiments.")
    with scope_tab:
        for domain, intents in DOMAINS.values():
            st.subheader(domain)
            st.write(", ".join(intent.replace("_", " ") for intent in intents))


if __name__ == "__main__":
    main()
