"""Lock and score a genuinely fresh collection once; no queries are supplied here.

The first version evaluates the unchanged confirmed MPNet head and strict margin.
Scope detection is deferred. Raw data and predictions remain under ignored runs/.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PureWindowsPath, PurePosixPath
import platform
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import fresh_query_collection as collection

INTENTS = collection.INTENTS
DOMAINS = {row[0]: row[1] for row in collection.ONTOLOGY}
KIND = "fresh_bank20_confirmed_head_v1"
ORIGINAL = "research/results/encoder-v2/confirmation-plan.json"
ORIGINAL_LOCK = "81fedc0777a0c98f02d071b9dd3adaef862c52c538a165ccc75d2b4f01187681"
POLICY = "research/results/encoder-v2/final-policy.json"
THRESHOLD = 2.1202011108398438
NUMERICS = "Round logits and top-two subtraction to IEEE754 binary32; lowest-index ties; margin >= threshold"
PACK_FILES = ("collection-plan.json", "ontology.json", "overlap-sources.json", "queries.csv",
              "participants.csv", "labels-a.csv", "labels-b.csv", "adjudications.csv",
              "INSTRUCTIONS.md", "evaluation-handoff.json")
SOURCES = ("tools/evaluate_fresh_bank20.py", "tests/test_fresh_evaluation.py",
           "tests/test_fresh_query_collection.py", "fedlora_edge/responses.py",
           "tools/fresh_query_collection.py", "docs/fresh-query-collection.md",
           "docs/fresh-evaluation-runner.md", ORIGINAL, POLICY)
PACKAGES = ("torch", "transformers", "peft", "tokenizers", "numpy", "safetensors")
digest = collection.digest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def relative(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Freeze inputs inside this repository; external paths are not portable")
    return path.relative_to(ROOT.resolve()).as_posix()


def private(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to((ROOT / name).resolve()) for name in ("runs", "assets")):
        raise ValueError("Private collection/run output must remain in ignored runs/ or assets/")
    return path


def runtime():
    return {"python": platform.python_version(), "platform": sys.platform,
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES}}


def model_contract():
    """Read identities and verify bytes without accessing old evaluation queries."""
    original = read(ROOT / ORIGINAL)
    if (original["lock_hash"] != ORIGINAL_LOCK
            or digest({k: v for k, v in original.items() if k != "lock_hash"}) != ORIGINAL_LOCK):
        raise ValueError("Original confirmation lock changed")
    cls = PureWindowsPath if PureWindowsPath(original["run_dir"]).drive else PurePosixPath
    old_root = cls(original["run_dir"]).parents[1]
    files = {cls(name).relative_to(old_root).as_posix(): value for name, value in original["locked_files"].items()}
    if any(sha(ROOT / name) != value for name, value in files.items()):
        raise ValueError("A previously frozen model, source or configuration changed")
    policy = read(ROOT / POLICY)
    if policy != original["selective_policies"]:
        raise ValueError("Clarification policy differs from its pre-confirmation lock")
    selected = next(p for p in policy["policies"] if p["target_validation_accuracy"] == 1.0)
    if selected["threshold"] != THRESHOLD:
        raise ValueError("Unexpected frozen clarification threshold")
    files.update({name: sha(ROOT / name) for name in SOURCES})
    return {"head_hash": original["head_hash"], "config_hash": original["config_hash"],
            "original_lock": ORIGINAL_LOCK, "run_id": original["run_id"], "version": 500,
            "asset_dir": "assets/reference-mpnet", "checkpoint_dir": "runs/encoder-flower-balanced-v2/selected",
            "scoring": "Unchanged EncoderPredictor.scores; masked mean + L2, FP32 CPU, max length384, batch16, threads2",
            "tie_rule": "Lowest index in the fixed ordered label mapping", "files": files}


def pack_hashes(pack):
    paths = [pack / name for name in PACK_FILES]
    paths.extend((pack / row["path"]).resolve() for row in read(pack / "overlap-sources.json") if row.get("path"))
    return {relative(path): sha(path) for path in paths}


def joined_records(pack, report):
    raw = {r["query_id"]: r for r in collection.read_rows(pack / "queries.csv", collection.QUERY_FIELDS)}
    adjudicated = {r["query_id"]: r for r in collection.read_rows(pack / "adjudications.csv", collection.ADJUDICATION_FIELDS)}
    records = []
    for row in sorted(report["records"], key=lambda r: r["query_id"]):
        q = raw[row["query_id"]]
        record = {**row, "provenance": q["provenance"], "provenance_detail": q["provenance_detail"],
                  "collection_batch": q["collection_batch"], "collected_at": q["collected_at"],
                  "adjudication": adjudicated[row["query_id"]]}
        # Privacy exclusions stay accounted for without copying excluded text.
        if row["disposition"] != "excluded":
            record["query"] = q["query"]
        records.append(record)
    return records


def prepare(pack, output, public):
    pack, output = private(pack), private(output)
    public = Path(public).resolve()
    if not public.is_relative_to((ROOT / "research/results").resolve()):
        raise ValueError("Public hash-only plans belong under research/results/")
    if output.exists() or public.exists():
        raise FileExistsError("Use new private and public directories; no lock may be overwritten")
    before = pack_hashes(pack)
    readiness = collection.ready_check(pack)
    if readiness["status"] != "ready_for_lock" or not readiness["collection_ready"]:
        raise ValueError("Fresh collection is not ready: " + ", ".join(readiness["errors"]))
    records = joined_records(pack, readiness)
    if pack_hashes(pack) != before:
        raise ValueError("Collection inputs changed during preparation")
    model = model_contract()
    environment = runtime()
    output.mkdir(parents=True)
    public.mkdir(parents=True)
    write_new(output / "records.json", records)
    write_new(output / "readiness.json", readiness)
    files = {**before, **model["files"], relative(output / "records.json"): sha(output / "records.json"),
             relative(output / "readiness.json"): sha(output / "readiness.json")}
    committed = {name: value for name, value in model["files"].items()
                 if name.startswith(("tools/", "tests/", "fedlora_edge/", "docs/", "research/"))}
    plan = {"kind": KIND, "created_utc": datetime.now(timezone.utc).isoformat(),
            "pack": relative(pack), "output": relative(output), "public": relative(public),
            "collection_sha256": readiness["collection_content_sha256"],
            "readiness_sha256": digest(readiness), "counts": readiness["counts"],
            "records_sha256": digest(records), "record_commitments": [digest(r) for r in records],
            "labels": list(INTENTS), "threshold": THRESHOLD, "acceptance": "margin >= threshold",
            "numerics": NUMERICS,
            "primary": "disposition supported: independently authored human queries with one fixed intent",
            "secondary": "Separate results by provenance and supported/unsupported/ambiguous label; never pooled into primary",
            "exclusions": "Fixed before inference; never scored; all counts and reasons retained privately",
            "model": model, "files": files, "committed_files": committed, "runtime": environment,
            "inference": {"device": "cpu", "cpu_threads": 2, "batch_size": 16},
            "scope_detector": "disabled; no fresh combined-scope claim",
            "failure_rule": "Claim collection and every retained query before model construction. No repeated inference after failure. Report-only may rebuild from bound saved predictions.",
            "limitations": "Human provenance/independence and sampling quality remain curator attestations. Nominal Wilson intervals ignore writer clustering. No future 100% guarantee."}
    plan["lock_hash"] = digest(plan)
    write_new(output / "plan.json", plan)
    write_new(public / "plan.json", plan)
    return plan


def verify(output, *, check_runtime=True):
    output = private(output)
    plan = read(output / "plan.json")
    if (plan.get("kind") != KIND or plan.get("labels") != list(INTENTS) or plan.get("threshold") != THRESHOLD
            or plan.get("numerics") != NUMERICS
            or plan.get("output") != relative(output)
            or digest({k: v for k, v in plan.items() if k != "lock_hash"}) != plan.get("lock_hash")):
        raise ValueError("Evaluation lock or fixed design changed")
    if read(ROOT / plan["public"] / "plan.json") != plan:
        raise ValueError("Public and private evaluation plans differ")
    model = model_contract()
    committed = {name: value for name, value in model["files"].items()
                 if name.startswith(("tools/", "tests/", "fedlora_edge/", "docs/", "research/"))}
    expected_files = {**pack_hashes(ROOT / plan["pack"]), **model["files"],
                      relative(output / "records.json"): sha(output / "records.json"),
                      relative(output / "readiness.json"): sha(output / "readiness.json")}
    if (plan["model"] != model or plan["committed_files"] != committed or plan["files"] != expected_files
            or plan["inference"] != {"device": "cpu", "cpu_threads": 2, "batch_size": 16}
            or plan["scope_detector"] != "disabled; no fresh combined-scope claim"):
        raise ValueError("Required model, input, source or inference contract changed")
    if check_runtime and runtime() != plan["runtime"]:
        raise ValueError("Scoring environment differs from the prepared environment")
    for name, value in plan["files"].items():
        path = ROOT / name
        if relative(path) != name or sha(path) != value:
            raise ValueError(f"Locked input changed: {name}")
    records = read(output / "records.json")
    if digest(records) != plan["records_sha256"] or [digest(r) for r in records] != plan["record_commitments"]:
        raise ValueError("Private frozen records changed")
    readiness = collection.ready_check(ROOT / plan["pack"])
    if not readiness["collection_ready"] or digest(readiness) != plan["readiness_sha256"]:
        raise ValueError("Collection no longer passes its frozen readiness gate")
    if (readiness != read(output / "readiness.json") or records != joined_records(ROOT / plan["pack"], readiness)
            or plan["counts"] != readiness["counts"]
            or plan["collection_sha256"] != readiness["collection_content_sha256"]):
        raise ValueError("Frozen records/counts differ from the audited collection")
    return plan, records


def git(*arguments):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", *arguments],
                                   cwd=ROOT, stderr=subprocess.PIPE, timeout=45)


def publication_files(plan):
    return {**plan["committed_files"], f"{plan['public']}/plan.json": sha(ROOT / plan["public"] / "plan.json")}


def verify_commit(plan, commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use the full 40-character publication commit")
    for name, value in publication_files(plan).items():
        if hashlib.sha256(git("show", f"{commit}:{name}")).hexdigest() != value:
            raise ValueError(f"Committed plan/source bytes differ: {name}")


def attest_publication(output, commit):
    plan, _ = verify(output)
    remote = git("remote", "get-url", "origin").decode().strip()
    if remote not in {"https://github.com/sameerreddy789/loflwr", "https://github.com/sameerreddy789/loflwr.git",
                      "git@github.com:sameerreddy789/loflwr.git"}:
        raise ValueError("Publication must use the declared project GitHub origin")
    verify_commit(plan, commit)
    observed = git("ls-remote", "origin", "refs/heads/main").decode().strip().split()
    if observed != [commit, "refs/heads/main"]:
        raise ValueError("Publication commit is not the currently observed remote main tip")
    receipt = {"kind": "observed_github_publication_v1", "lock_hash": plan["lock_hash"],
               "commit": commit, "origin": remote, "ref": "refs/heads/main",
               "observed_utc": datetime.now(timezone.utc).isoformat(), "files": publication_files(plan)}
    receipt["receipt_hash"] = digest(receipt)
    write_new(Path(output) / "publication.json", receipt)
    return receipt


def verify_receipt(output, plan):
    receipt = read(Path(output) / "publication.json")
    if (receipt.get("kind") != "observed_github_publication_v1" or receipt.get("lock_hash") != plan["lock_hash"]
            or receipt.get("ref") != "refs/heads/main" or receipt.get("files") != publication_files(plan)
            or digest({k: v for k, v in receipt.items() if k != "receipt_hash"}) != receipt.get("receipt_hash")):
        raise ValueError("Publication receipt is missing or changed")
    verify_commit(plan, receipt["commit"])
    return receipt


def live_publication(plan, receipt):
    """A self-hashed local receipt alone cannot prove a commit was published."""
    remote = git("remote", "get-url", "origin").decode().strip()
    if remote != receipt["origin"] or remote not in {
            "https://github.com/sameerreddy789/loflwr", "https://github.com/sameerreddy789/loflwr.git",
            "git@github.com:sameerreddy789/loflwr.git"}:
        raise ValueError("Project publication origin changed after attestation")
    observed = git("ls-remote", "origin", "refs/heads/main").decode().strip().split()
    if (len(observed) != 2 or observed[1] != "refs/heads/main"
            or not re.fullmatch(r"[0-9a-f]{40}", observed[0])):
        raise ValueError("Cannot verify the currently published main commit")
    tip = observed[0]
    # Both objects must be present locally. Fetching Git history is an explicit
    # preparation step; this command never installs code or dependencies.
    git("merge-base", "--is-ancestor", receipt["commit"], tip)
    verify_commit(plan, tip)
    return {"origin": remote, "ref": "refs/heads/main", "remote_tip": tip,
            "publication_commit": receipt["commit"], "observed_utc": datetime.now(timezone.utc).isoformat()}


def claim_inference(output, plan, records, publication=None):
    """Shared per-query claims prevent a new output path from rescoring a set."""
    ledger = ROOT / "runs/fresh-evaluation-exposures"
    ledger.mkdir(parents=True, exist_ok=True)
    guard = ledger / ".claim-lock"
    guard.mkdir()  # Atomic exclusion; a crashed owner must be investigated manually.
    try:
        paths = [ledger / f"collection-{plan['collection_sha256']}.json"]
        paths.extend(ledger / f"query-{r['normalized_query_sha256']}.json"
                     for r in records if r["disposition"] != "excluded")
        if (Path(output) / "started.json").exists() or any(p.exists() for p in paths):
            raise FileExistsError("This collection or a retained query was already claimed; inference must not repeat")
        marker = {"lock_hash": plan["lock_hash"], "collection_sha256": plan["collection_sha256"],
                  "started_utc": datetime.now(timezone.utc).isoformat(), "output": relative(output),
                  "live_publication": publication,
                  "status": "inference_claimed; completion or failure must be reported"}
        for path in paths:
            write_new(path, marker)
        write_new(Path(output) / "started.json", marker)
        return marker
    finally:
        guard.rmdir()


def float32(value):
    """Keep the frozen FP32 evaluator's rounding without importing a model."""
    try:
        result = struct.unpack("!f", struct.pack("!f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("Score or margin is outside finite FP32 range") from error
    if not math.isfinite(result):
        raise ValueError("Score or margin is outside finite FP32 range")
    return result


def predictions_from_scores(records, scores):
    retained = [r for r in records if r["disposition"] != "excluded"]
    if len(scores) != len(retained):
        raise ValueError("Score rows differ from the retained frozen rows")
    result = []
    for record, row in zip(retained, scores):
        if len(row) != len(INTENTS) or any(type(v) not in (int, float) or not math.isfinite(v) for v in row):
            raise ValueError("Expected exactly twenty finite numeric scores per retained row")
        row = [float32(value) for value in row]
        order = sorted(range(len(row)), key=lambda i: (-row[i], i))
        margin = float32(row[order[0]] - row[order[1]])
        result.append({"query_id": record["query_id"], "prediction": INTENTS[order[0]],
                       "margin": margin, "answered": margin >= THRESHOLD})
    return result


def wilson(correct, count):
    if not count:
        return None
    z = 1.959963984540054
    p, denominator = correct / count, 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - radius, center + radius]


def supported_metrics(rows):
    n = len(rows)
    correct = sum(r["prediction"] == r["final_label"] for r in rows)
    answered = [r for r in rows if r["answered"]]
    answered_correct = sum(r["prediction"] == r["final_label"] for r in answered)
    per_intent = {}
    for label in INTENTS:
        tp = sum(r["prediction"] == label and r["final_label"] == label for r in rows)
        support = sum(r["final_label"] == label for r in rows)
        predicted = sum(r["prediction"] == label for r in rows)
        per_intent[label] = {"count": support, "correct": tp, "accuracy": tp / support if support else None,
                             "nominal_wilson_95": wilson(tp, support),
                             "f1": 2 * tp / (support + predicted) if support + predicted else 0.0}
    domains = {}
    for domain in sorted(set(DOMAINS.values())):
        subset = [r for r in rows if DOMAINS[r["final_label"]] == domain]
        matched = sum(r["prediction"] == r["final_label"] for r in subset)
        domains[domain] = {"count": len(subset), "correct": matched,
                           "accuracy": matched / len(subset) if subset else None,
                           "nominal_wilson_95": wilson(matched, len(subset))}
    available = [r["accuracy"] for r in domains.values() if r["count"]]
    return {"count": n, "correct": correct, "accuracy": correct / n if n else None,
            "nominal_wilson_95": wilson(correct, n), "macro_f1_all_20": sum(r["f1"] for r in per_intent.values()) / 20 if n else None,
            "per_intent": per_intent, "per_domain": domains,
            "worst_domain_accuracy": min(available) if available else None,
            "domain_gap": max(available) - min(available) if available else None,
            "selective": {"answered": len(answered), "correct_answered": answered_correct,
                "wrong_answered": len(answered) - answered_correct, "declined": n - len(answered),
                "coverage": len(answered) / n if n else None,
                "accuracy_on_answered": answered_correct / len(answered) if answered else None,
                "nominal_wilson_95_on_answered": wilson(answered_correct, len(answered)),
                "correct_answers_per_supported_query": answered_correct / n if n else None}}


def summarize(records, predictions):
    retained = [r for r in records if r["disposition"] != "excluded"]
    if len(predictions) != len(retained) or [p["query_id"] for p in predictions] != [r["query_id"] for r in retained]:
        raise ValueError("Prediction IDs/order do not match every retained row")
    rows = []
    for r, p in zip(retained, predictions):
        if (p["prediction"] not in INTENTS or type(p["margin"]) not in (int, float)
                or not math.isfinite(p["margin"]) or p["margin"] < 0 or type(p["answered"]) is not bool
                or p["answered"] != (p["margin"] >= THRESHOLD)):
            raise ValueError("Invalid prediction or frozen margin decision")
        rows.append({**r, **p})
    primary = [r for r in rows if r["disposition"] == "supported"]
    secondary = {}
    for provenance in sorted({r["provenance"] for r in rows if r["disposition"] == "secondary"}):
        group = [r for r in rows if r["disposition"] == "secondary" and r["provenance"] == provenance]
        entry = {"supported_labels": supported_metrics([r for r in group if r["final_label"] in INTENTS])}
        for label in ("unsupported", "ambiguous"):
            subset = [r for r in group if r["final_label"] == label]
            answered = sum(r["answered"] for r in subset)
            entry[label] = {"count": len(subset), "answered": answered, "declined": len(subset) - answered,
                            "answer_rate": answered / len(subset) if subset else None,
                            "nominal_wilson_95_answer_rate": wilson(answered, len(subset))}
        secondary[provenance] = entry
    writers = Counter(r["author_id"] for r in primary)
    return {"primary_human_supported": supported_metrics(primary), "secondary_by_provenance": secondary,
            "accounting": {"submitted": len(records), "scored": len(rows),
                "unique_source_ids": len({r["source_id"] for r in records}),
                "dispositions": dict(Counter(r["disposition"] for r in records)),
                "provenance": dict(Counter(r["provenance"] for r in records)),
                "excluded_reasons": dict(Counter(r["adjudication"]["exclusion_reason"] for r in records if r["disposition"] == "excluded"))},
            "primary_writer_diversity": {"count": len(writers), "largest_share": max(writers.values()) / len(primary) if primary else None,
                "query_counts_descending": sorted(writers.values(), reverse=True)},
            "limits": ["Nominal binomial intervals ignore writer clustering and sampling bias.",
                       "Unsupported answer rate is false acceptance; ambiguous answers are reported separately.",
                       "Declines do not count as correct intent answers. Secondary rows do not enter primary human accuracy.",
                       "Human provenance/independence require external review. This result cannot establish perfect future accuracy."]}


def predict(plan, rows):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from fedlora_edge.encoder_inference import EncoderPredictor
    model = EncoderPredictor(ROOT / plan["model"]["asset_dir"], ROOT / plan["model"]["checkpoint_dir"], cpu_threads=2)
    if model.saved["b_hash"] != plan["model"]["head_hash"]:
        raise ValueError("Loaded head differs from locked model")
    scores, truncated = model.scores([r["query"] for r in rows], batch_size=16)
    return scores.tolist(), int(truncated)


def score(output):
    output = private(output)
    plan, records = verify(output)
    receipt = verify_receipt(output, plan)
    publication = live_publication(plan, receipt)
    claim_inference(output, plan, records, publication)
    try:
        rows = [r for r in records if r["disposition"] != "excluded"]
        scores, truncated = predict(plan, rows)
        predictions = predictions_from_scores(records, scores)
        # Refuse publication if any input changed while inference ran.
        verify(output)
        envelope = {"lock_hash": plan["lock_hash"], "records_sha256": plan["records_sha256"],
                    "publication_receipt_hash": receipt["receipt_hash"], "predictions": predictions,
                    "truncated_queries": truncated}
        envelope["prediction_hash"] = digest(envelope)
        write_new(output / "predictions.json", envelope)
        return report(output)
    except Exception as error:
        write_new(output / "failure.json", {"lock_hash": plan["lock_hash"], "error_type": type(error).__name__,
                  "status": "claimed; do not rescore; investigate and disclose this attempt"})
        raise


def report(output):
    output = private(output)
    plan, records = verify(output)
    receipt = verify_receipt(output, plan)
    started = read(output / "started.json")
    saved = read(output / "predictions.json")
    if (started["lock_hash"] != plan["lock_hash"] or saved["lock_hash"] != plan["lock_hash"]
            or saved["records_sha256"] != plan["records_sha256"]
            or saved["publication_receipt_hash"] != receipt["receipt_hash"]
            or digest({k: v for k, v in saved.items() if k != "prediction_hash"}) != saved["prediction_hash"]):
        raise ValueError("Saved predictions do not match the once-only locked inference")
    result = {"kind": KIND, "lock_hash": plan["lock_hash"], "publication_commit": receipt["commit"],
              "prediction_hash": saved["prediction_hash"], "model_head_hash": plan["model"]["head_hash"],
              "threshold": THRESHOLD, "truncated_queries": saved["truncated_queries"],
              **summarize(records, saved["predictions"])}
    result["result_hash"] = digest(result)
    target = ROOT / plan["public"] / "result.json"
    if target.exists():
        if read(target) != result:
            raise ValueError("Existing public result differs; never replace a scored outcome")
    else:
        write_new(target, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("--pack", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--public", required=True, type=Path)
    attest = commands.add_parser("attest-publication")
    attest.add_argument("--output", required=True, type=Path)
    attest.add_argument("--commit", required=True)
    for action in ("score", "report"):
        command = commands.add_parser(action)
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = prepare(args.pack, args.output, args.public)
            print(json.dumps({"status": "locked_locally_not_yet_published", "lock_hash": result["lock_hash"]}))
        elif args.action == "attest-publication":
            result = attest_publication(args.output, args.commit)
            print(json.dumps({"status": "publication_observed", "commit": result["commit"]}))
        else:
            result = score(args.output) if args.action == "score" else report(args.output)
            print(json.dumps({"status": "reported", "lock_hash": result["lock_hash"], "result_hash": result["result_hash"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"Blocked: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
