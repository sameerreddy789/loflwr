"""Lock one fresh paired comparison of fixed margin versus fixed margin + scope.

This module does not modify or rescore the historical evaluations. No fresh
queries are supplied. The shared exposure ledger prevents reuse across runners.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import math
import os
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import evaluate_fresh_bank20 as base

collection = base.collection
INTENTS = base.INTENTS
KIND = "fresh_bank20_paired_scope_v1"
SCOPE_PLAN = "research/results/scope-v1/confirmation-plan.json"
SCOPE_LOCK = "d5944ea9ec76a14c67870421ba782425d65245e3ed35207ff83660ba3ac0c7aa"
SCOPE_MODEL = "runs/scope-v1/scope-model.json"
SCOPE_MODEL_SHA = "c9517050df871bd5abc29eec9bdb5f12ed0299f6feca86f0ac0412a4827e9bbc"
ENCODER_ID = "6135b7f45d34cbcb9271c44f446662ba0bfdbe9d735a3a9b890b8f58b3da03fb"
SCOPE_THRESHOLD = -0.4532273876866362
MIN_HUMAN_UNSUPPORTED = 200
INFERENCE = {"device": "cpu", "requested_cpu_threads": 2, "effective_cpu_threads": 1,
             "batch_size": 16, "max_length": 384}
DESIGN = {
    "baseline": "answer iff FP32 top-two margin >= 2.1202011108398438",
    "combined": "answer iff baseline answers and FP64 scope score >= -0.4532273876866362",
    "head_numerics": base.NUMERICS,
    "scope_numerics": "FP32 normalized features @ NumPy FP64 coefficients + FP64 intercept; >= FP64 threshold",
    "feature_recipe": "One shared masked-mean/L2 FP32 MPNet pass, right padding, max_length384, batch16. Historical scope used batch8; no historical result is recomputed.",
    "supported_minimum": "20 audited independent-human queries per intent; retain extras",
    "unsupported_minimum": MIN_HUMAN_UNSUPPORTED,
    "unsupported_cohort": "Reviewed independent-human unsupported queries only; exploratory protocol minimum, not a power calculation or representative sampling guarantee",
    "primary": "Paired policy counts on human supported and human unsupported cohorts separately; no pooled overall accuracy",
    "secondary": "Ambiguous and external/model-generated/unknown rows reported separately by provenance and label",
    "failure_rule": "Claim collection and every retained query once, jointly for both policies, before model construction. Failure stays spent; report may use bound saved predictions only.",
}
SOURCES = ("tools/evaluate_fresh_scope.py", "tests/test_fresh_scope_evaluation.py",
           "docs/fresh-scope-evaluation.md", SCOPE_PLAN)


def runtime():
    import numpy as np
    stream = io.StringIO()
    with redirect_stdout(stream):
        np.show_config()
    return {**base.runtime(), "numpy_build": stream.getvalue(),
            "machine": platform.machine(), "processor": platform.processor(),
            "thread_environment": {name: os.environ.get(name) for name in
                ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def model_contract():
    """Verify frozen bytes/identities; never decode or score old evaluation rows."""
    head = base.model_contract()
    scope = base.read(ROOT / SCOPE_PLAN)
    if (scope.get("lock_hash") != SCOPE_LOCK
            or base.digest({k: v for k, v in scope.items() if k != "lock_hash"}) != SCOPE_LOCK
            or scope.get("model_file") != SCOPE_MODEL or scope.get("model_sha256") != SCOPE_MODEL_SHA
            or scope.get("head_hash") != head["head_hash"]):
        raise ValueError("Frozen scope plan or head identity changed")
    for name, expected in scope["files"].items():
        if base.relative(ROOT / name) != name or base.sha(ROOT / name) != expected:
            raise ValueError(f"Previously frozen scope file changed: {name}")
    model = base.read(ROOT / SCOPE_MODEL)
    validate_scope_model(model)
    files = {**head["files"], **scope["files"], **{name: base.sha(ROOT / name) for name in SOURCES}}
    return {"head": head, "scope_file": SCOPE_MODEL, "scope_sha256": SCOPE_MODEL_SHA,
            "scope_lock": SCOPE_LOCK, "scope_threshold": SCOPE_THRESHOLD,
            "encoder_identity_hash": ENCODER_ID, "files": files}


def validate_scope_model(model):
    coef = model.get("coef")
    if (not isinstance(coef, list) or len(coef) != 768
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in coef)
            or type(model.get("intercept")) not in (int, float) or not math.isfinite(model["intercept"])
            or model.get("threshold") != SCOPE_THRESHOLD
            or model.get("encoder_identity_hash") != ENCODER_ID):
        raise ValueError("Scope coefficients, numeric values, threshold or encoder identity changed")


def cohort_accounting(records, *, require_ready=True):
    retained = [r for r in records if r["disposition"] != "excluded"]
    unsupported = [r for r in retained if r["disposition"] == "secondary"
                   and r["provenance"] == "independent_human" and r["final_label"] == "unsupported"]
    if require_ready and len(unsupported) < MIN_HUMAN_UNSUPPORTED:
        raise ValueError(f"Combined scope cohort not ready: need {MIN_HUMAN_UNSUPPORTED} reviewed independent-human unsupported queries; found {len(unsupported)}")
    counts = Counter((r["provenance"], r["final_label"] if r["final_label"] in ("unsupported", "ambiguous")
                      else "supported") for r in retained)
    return {"retained": len(retained), "human_unsupported": len(unsupported),
            "human_supported": sum(r["disposition"] == "supported" for r in retained),
            "retained_mixture": [{"provenance": key[0], "label_group": key[1], "count": count,
                                  "fraction_of_retained": count / len(retained)}
                                 for key, count in sorted(counts.items())],
            "interpretation": "Observed collection mix only; no production prior or representative-population claim"}


def committed_files(model):
    return {name: value for name, value in model["files"].items()
            if name.startswith(("tools/", "tests/", "fedlora_edge/", "docs/", "research/"))}


def prepare(pack, output, public):
    pack, output = base.private(pack), base.private(output)
    public = Path(public).resolve()
    if not public.is_relative_to((ROOT / "research/results").resolve()):
        raise ValueError("Public hash-only plans belong under research/results/")
    if output.exists() or public.exists():
        raise FileExistsError("Use new private/public directories; never overwrite an evaluation")
    before = base.pack_hashes(pack)
    readiness = collection.ready_check(pack)
    if readiness["status"] != "ready_for_lock" or not readiness["collection_ready"]:
        raise ValueError("Fresh collection is not ready: " + ", ".join(readiness["errors"]))
    records = base.joined_records(pack, readiness)
    cohorts = cohort_accounting(records)
    if base.pack_hashes(pack) != before:
        raise ValueError("Collection changed during preparation")
    model, environment = model_contract(), runtime()
    output.mkdir(parents=True)
    public.mkdir(parents=True)
    base.write_new(output / "records.json", records)
    base.write_new(output / "readiness.json", readiness)
    files = {**before, **model["files"], base.relative(output / "records.json"): base.sha(output / "records.json"),
             base.relative(output / "readiness.json"): base.sha(output / "readiness.json")}
    plan = {"kind": KIND, "created_utc": datetime.now(timezone.utc).isoformat(),
            "pack": base.relative(pack), "output": base.relative(output), "public": base.relative(public),
            "collection_sha256": readiness["collection_content_sha256"],
            "readiness_sha256": base.digest(readiness), "counts": readiness["counts"], "cohorts": cohorts,
            "records_sha256": base.digest(records), "record_commitments": [base.digest(r) for r in records],
            "labels": list(INTENTS), "design": DESIGN, "inference": INFERENCE,
            "model": model, "files": files, "committed_files": committed_files(model), "runtime": environment}
    plan["lock_hash"] = base.digest(plan)
    base.write_new(output / "plan.json", plan)
    base.write_new(public / "plan.json", plan)
    return plan


def verify(output, *, check_runtime=True):
    output = base.private(output)
    plan = base.read(output / "plan.json")
    if (plan.get("kind") != KIND or plan.get("labels") != list(INTENTS)
            or plan.get("design") != DESIGN or plan.get("inference") != INFERENCE
            or plan.get("output") != base.relative(output)
            or base.digest({k: v for k, v in plan.items() if k != "lock_hash"}) != plan.get("lock_hash")):
        raise ValueError("Paired evaluation lock or fixed design changed")
    public = (ROOT / plan["public"]).resolve()
    if not public.is_relative_to((ROOT / "research/results").resolve()) or base.read(public / "plan.json") != plan:
        raise ValueError("Public/private evaluation plans differ")
    pack = base.private(ROOT / plan["pack"])
    model = model_contract()
    files = {**base.pack_hashes(pack), **model["files"],
             base.relative(output / "records.json"): base.sha(output / "records.json"),
             base.relative(output / "readiness.json"): base.sha(output / "readiness.json")}
    if plan["model"] != model or plan["committed_files"] != committed_files(model) or plan["files"] != files:
        raise ValueError("Required model, source or input contract changed")
    if check_runtime and runtime() != plan["runtime"]:
        raise ValueError("Scoring environment differs from prepared environment")
    for name, value in plan["files"].items():
        if base.relative(ROOT / name) != name or base.sha(ROOT / name) != value:
            raise ValueError(f"Locked file changed: {name}")
    records, readiness = base.read(output / "records.json"), collection.ready_check(pack)
    if (not readiness["collection_ready"] or base.digest(readiness) != plan["readiness_sha256"]
            or readiness != base.read(output / "readiness.json")
            or records != base.joined_records(pack, readiness) or plan["counts"] != readiness["counts"]
            or plan["collection_sha256"] != readiness["collection_content_sha256"]
            or base.digest(records) != plan["records_sha256"]
            or [base.digest(r) for r in records] != plan["record_commitments"]
            or cohort_accounting(records) != plan["cohorts"]):
        raise ValueError("Frozen snapshot/cohorts differ from audited collection")
    return plan, records


def attest_publication(output, commit):
    plan, _ = verify(output)
    remote = base.git("remote", "get-url", "origin").decode().strip()
    if remote not in {"https://github.com/sameerreddy789/loflwr", "https://github.com/sameerreddy789/loflwr.git",
                      "git@github.com:sameerreddy789/loflwr.git"}:
        raise ValueError("Publication must use the declared project GitHub origin")
    base.verify_commit(plan, commit)
    if base.git("ls-remote", "origin", "refs/heads/main").decode().strip().split() != [commit, "refs/heads/main"]:
        raise ValueError("Publication commit is not the currently observed remote main tip")
    receipt = {"kind": "observed_github_publication_v1", "lock_hash": plan["lock_hash"],
               "commit": commit, "origin": remote, "ref": "refs/heads/main",
               "observed_utc": datetime.now(timezone.utc).isoformat(), "files": base.publication_files(plan)}
    receipt["receipt_hash"] = base.digest(receipt)
    base.write_new(Path(output) / "publication.json", receipt)
    return receipt


def scores_from_features(features, scope_model):
    import numpy as np
    validate_scope_model(scope_model)
    X = np.asarray(features)
    if (X.dtype != np.float32 or X.ndim != 2 or X.shape[1] != 768 or not len(X)
            or not np.isfinite(X).all() or not np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5)):
        raise ValueError("Expected finite normalized FP32 MPNet features")
    result = X @ np.asarray(scope_model["coef"], dtype=np.float64) + scope_model["intercept"]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite scope scores")
    return result


def paired_scores(predictor, queries, scope_model):
    """Expose one feature pass to both fixed models, preserving head inference."""
    from fedlora_edge.encoder_local_data import encode_batches
    if predictor.device != "cpu" or predictor.torch.get_num_threads() != INFERENCE["effective_cpu_threads"]:
        raise ValueError("Expected unchanged CPU predictor with one effective Torch thread")
    with predictor.lock:
        features, truncated = encode_batches(predictor.encoder, predictor.tokenizer, queries,
                                             batch_size=INFERENCE["batch_size"])
        logits = predictor.head.predict_scores(features)
        scope_scores = scores_from_features(features, scope_model)
    return logits, scope_scores, truncated


def predict(plan, rows):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from fedlora_edge.encoder_inference import EncoderPredictor
    head = plan["model"]["head"]
    predictor = EncoderPredictor(ROOT / head["asset_dir"], ROOT / head["checkpoint_dir"],
                                 device="cpu", cpu_threads=INFERENCE["requested_cpu_threads"])
    if predictor.saved["b_hash"] != head["head_hash"]:
        raise ValueError("Loaded head differs from locked model")
    logits, scope, truncated = paired_scores(predictor, [r["query"] for r in rows],
                                             base.read(ROOT / plan["model"]["scope_file"]))
    return logits.tolist(), scope.tolist(), int(truncated)


def predictions_from_scores(records, logits, scope_scores):
    baseline = base.predictions_from_scores(records, logits)
    if len(scope_scores) != len(baseline):
        raise ValueError("Scope scores differ from retained row count")
    predictions = []
    for row, score in zip(baseline, scope_scores):
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("Expected finite FP64 scope scores")
        predictions.append({**row, "scope_score": score, "scope_passed": score >= SCOPE_THRESHOLD,
                            "combined_answered": row["answered"] and score >= SCOPE_THRESHOLD})
    return predictions


def policy_metrics(rows, key, *, supported):
    selected = [{**r, "answered": r[key]} for r in rows]
    if supported:
        return base.supported_metrics(selected)
    n, answers = len(selected), sum(r["answered"] for r in selected)
    return {"count": n, "answered": answers, "declined": n - answers,
            "answer_rate": answers / n if n else None, "nominal_wilson_95_answer_rate": base.wilson(answers, n)}


def paired_metrics(rows, *, supported=False, unsupported=False):
    removed = [r for r in rows if r["answered"] and not r["combined_answered"]]
    baseline = policy_metrics(rows, "answered", supported=supported)
    combined = policy_metrics(rows, "combined_answered", supported=supported)
    changed = {"count": len(rows), "answers_removed": len(removed), "answers_added": 0,
               "unchanged_decisions": len(rows) - len(removed)}
    if supported:
        lost = sum(r["prediction"] == r["final_label"] for r in removed)
        changed.update(correct_answers_lost=lost, wrong_answers_removed=len(removed) - lost,
                       correct_answers_lost_per_supported_query=lost / len(rows) if rows else None,
                       wrong_answers_removed_per_supported_query=(len(removed) - lost) / len(rows) if rows else None,
                       coverage_change=combined["selective"]["coverage"] - baseline["selective"]["coverage"] if rows else None)
    elif unsupported:
        changed.update(false_accepts_removed=len(removed),
                       false_accept_rate_change=-len(removed) / len(rows) if rows else None)
    return {"baseline_margin": baseline, "combined_margin_scope": combined, "paired": changed}


def summarize(records, predictions):
    retained = [r for r in records if r["disposition"] != "excluded"]
    required = {"query_id", "prediction", "margin", "answered", "scope_score", "scope_passed", "combined_answered"}
    if any(not isinstance(p, dict) or set(p) != required for p in predictions):
        raise ValueError("Predictions must contain only the fixed scored fields, never labels or metadata")
    required = {"query_id", "prediction", "margin", "answered", "scope_score", "scope_passed", "combined_answered"}
    if any(not isinstance(p, dict) or set(p) != required for p in predictions):
        raise ValueError("Unexpected fields in saved paired prediction")
    # The old validator checks row alignment, finite margins and baseline decisions.
    baseline_report = base.summarize(records, [{k: p[k] for k in ("query_id", "prediction", "margin", "answered")}
                                               for p in predictions])
    rows = []
    for record, prediction in zip(retained, predictions):
        score = prediction.get("scope_score")
        if (type(score) not in (int, float) or not math.isfinite(score)
                or type(prediction.get("scope_passed")) is not bool
                or type(prediction.get("combined_answered")) is not bool
                or prediction["scope_passed"] != (score >= SCOPE_THRESHOLD)
                or prediction["combined_answered"] != (prediction["answered"] and prediction["scope_passed"])):
            raise ValueError("Invalid frozen scope decision")
        rows.append({**record, **prediction})
    supported = [r for r in rows if r["disposition"] == "supported"]
    unsupported = [r for r in rows if r["provenance"] == "independent_human" and r["final_label"] == "unsupported"]
    secondary = {}
    for provenance in sorted({r["provenance"] for r in rows if r["disposition"] == "secondary"}):
        group = [r for r in rows if r["provenance"] == provenance and r["disposition"] == "secondary"]
        secondary[provenance] = {
            "supported": paired_metrics([r for r in group if r["final_label"] in INTENTS], supported=True),
            "unsupported": paired_metrics([r for r in group if r["final_label"] == "unsupported"], unsupported=True),
            "ambiguous": paired_metrics([r for r in group if r["final_label"] == "ambiguous"])}
    negative_writers = Counter(r["author_id"] for r in unsupported)
    domain_policies = {}
    for domain in sorted(set(base.DOMAINS.values())):
        group = [r for r in supported if base.DOMAINS[r["final_label"]] == domain]
        paired = paired_metrics(group, supported=True)
        domain_policies[domain] = {"count": len(group),
            "baseline_margin": paired["baseline_margin"]["selective"],
            "combined_margin_scope": paired["combined_margin_scope"]["selective"],
            "paired": paired["paired"]}
    return {"primary_human_supported": paired_metrics(supported, supported=True),
            "primary_human_supported_by_domain": domain_policies,
            "primary_human_unsupported": paired_metrics(unsupported, unsupported=True),
            "secondary_by_provenance": secondary, "cohorts": cohort_accounting(records, require_ready=False),
            "accounting": baseline_report["accounting"],
            "primary_writer_diversity": baseline_report["primary_writer_diversity"],
            "unsupported_writer_diversity": {"count": len(negative_writers),
                "largest_share": max(negative_writers.values()) / len(unsupported) if unsupported else None,
                "query_counts_descending": sorted(negative_writers.values(), reverse=True)},
            "limits": [*baseline_report["limits"],
                "The exploratory unsupported minimum is not a statistical power calculation.",
                "The observed positive/negative mix is not a deployment prior; no pooled overall accuracy is reported.",
                "Both policies share the identical fixed top-1 head. Scope only removes answers; it cannot improve top-1 accuracy.",
                "Human unsupported rows repeat in the provenance breakdown for accounting, not as additional observations."]}


def score(output):
    output = base.private(output)
    plan, records = verify(output)
    receipt = base.verify_receipt(output, plan)
    publication = base.live_publication(plan, receipt)
    base.claim_inference(output, plan, records, publication)
    try:
        rows = [r for r in records if r["disposition"] != "excluded"]
        logits, scope_scores, truncated = predict(plan, rows)
        predictions = predictions_from_scores(records, logits, scope_scores)
        verify(output)
        saved = {"lock_hash": plan["lock_hash"], "records_sha256": plan["records_sha256"],
                 "publication_receipt_hash": receipt["receipt_hash"], "predictions": predictions,
                 "truncated_queries": truncated}
        saved["prediction_hash"] = base.digest(saved)
        base.write_new(output / "predictions.json", saved)
        return report(output)
    except Exception as error:
        base.write_new(output / "failure.json", {"lock_hash": plan["lock_hash"], "error_type": type(error).__name__,
                       "status": "Both policies claimed; never rescore; preserve/disclose failure"})
        raise


def report(output):
    output = base.private(output)
    plan, records = verify(output)
    receipt = base.verify_receipt(output, plan)
    started, saved = base.read(output / "started.json"), base.read(output / "predictions.json")
    if (started["lock_hash"] != plan["lock_hash"] or saved["lock_hash"] != plan["lock_hash"]
            or saved["records_sha256"] != plan["records_sha256"]
            or saved["publication_receipt_hash"] != receipt["receipt_hash"]
            or base.digest({k: v for k, v in saved.items() if k != "prediction_hash"}) != saved["prediction_hash"]):
        raise ValueError("Saved paired predictions differ from locked once-only inference")
    result = {"kind": KIND, "lock_hash": plan["lock_hash"], "publication_commit": receipt["commit"],
              "prediction_hash": saved["prediction_hash"], "head_hash": plan["model"]["head"]["head_hash"],
              "scope_sha256": plan["model"]["scope_sha256"], "design": DESIGN,
              "truncated_queries": saved["truncated_queries"], **summarize(records, saved["predictions"])}
    result["result_hash"] = base.digest(result)
    target = ROOT / plan["public"] / "result.json"
    if target.exists():
        if base.read(target) != result:
            raise ValueError("Existing public result differs; never replace scored outcomes")
    else:
        base.write_new(target, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare_cli = commands.add_parser("prepare")
    for name in ("pack", "output", "public"):
        prepare_cli.add_argument(f"--{name}", type=Path, required=True)
    attest = commands.add_parser("attest-publication")
    attest.add_argument("--output", type=Path, required=True)
    attest.add_argument("--commit", required=True)
    for action in ("score", "report"):
        commands.add_parser(action).add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = prepare(args.pack, args.output, args.public)
            status = "locked_locally_not_yet_published"
        elif args.action == "attest-publication":
            result = attest_publication(args.output, args.commit)
            status = "publication_observed"
        else:
            result = score(args.output) if args.action == "score" else report(args.output)
            status = "reported"
        import json
        print(json.dumps({"status": status, "lock_hash": result["lock_hash"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"Blocked: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
