"""Audit full participant preparation and one-round head compatibility on reused data.

This single-host public-benchmark audit does not establish data isolation,
privacy, Flower transport, convergence, new accuracy, or physical deployment.
Every run uses a fresh ignored directory and retains failures without retries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedlora_edge.data import content_hash
from fedlora_edge.encoder_bucketing import FEATURE_SETTINGS, assemble, build_plan, validate_plan
from fedlora_edge.encoder_data import load_encoder_manifest
from fedlora_edge.encoder_local_data import load_own_records
from tools import evaluate_fresh_bank20 as base
from tools.run_encoder_matrix import HEAD

KIND = "bucketed_full_local_preparation_compatibility_v1"
DATASET = "assets/encoder-bank20-v1"
COUNTS = {"0": 513, "1": 615, "2": 608, "3": 616, "4": 408, "server": 200}
TOKENIZER_PROBES = ("A short synthetic query.", "word " * 410,
                    "Different punctuation: abc?!", "many " * 382)
THREAD_ENV = {key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
SOURCES = (
    "tools/check_bucketed_integration.py", "tests/test_bucketed_integration.py",
    "tools/prepare_encoder_bucketed.py", "fedlora_edge/encoder_bucketing.py",
    "tools/run_encoder_matrix.py", "fedlora_edge/encoder_local_data.py",
    "fedlora_edge/encoder_data.py", "fedlora_edge/encoder_head.py",
    "fedlora_edge/aggregation.py", "fedlora_edge/state.py", "fedlora_edge/training.py",
)
DESIGN = {
    "participants": COUNTS, "layout": "five", "total_rows": 2960,
    "preparation": FEATURE_SETTINGS, "participant_timeout_seconds": 900,
    "feature_absolute_tolerance": 1e-5, "fixed_head_top1_changes_allowed": 0,
    "fixed_head_strict_margin_changes_allowed": 0,
    "tokenizer_probes": {"count": len(TOKENIZER_PROBES), "probe_hash": content_hash(TOKENIZER_PROBES),
                         "scope": "Authored tokenization probes with no expected labels or model inference."},
    "head_smoke": {"initial_seed": 42, "attempt_id": 1, "optimizer_steps_per_client": 1,
                   "strategy": "equal", "state_absolute_tolerance": 1e-5, "config": HEAD},
    "scope": "Full single-host public-benchmark compatibility audit using existing training and reused validation; no new accuracy or Flower experiment.",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def output_paths(output, public=None):
    output = Path(output).resolve()
    private_root = (ROOT / "runs").resolve()
    if not output.is_relative_to(private_root) or output == private_root:
        raise ValueError("Integration output must be a fresh directory inside ignored runs/.")
    public = None if public is None else Path(public).resolve()
    if public is not None:
        public_root = (ROOT / "research/results").resolve()
        if not public.is_relative_to(public_root) or public.suffix != ".json":
            raise ValueError("Optional public summary must be a JSON file under research/results/.")
    if output.exists() or (public is not None and public.exists()):
        raise FileExistsError("Use new outputs; preserve completed or failed integration attempts.")
    return output, public


def runtime():
    return {**base.runtime(), "os": platform.platform(), "executable": sys.executable,
            "psutil": importlib.metadata.version("psutil"),
            "participant_environment": {**THREAD_ENV, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                        "TOKENIZERS_PARALLELISM": "false"}}


def source_contract(model):
    """Validate all approved bundles and hash their caches without any test reads."""
    source = ROOT / DATASET
    manifest = load_encoder_manifest(source)
    bucketed = build_plan(manifest, "five")
    files = {**model["files"], **{name: base.sha(ROOT / name) for name in SOURCES},
             f"{DATASET}/manifest.json": base.sha(source / "manifest.json")}
    rows, seen = {}, set()
    for role, expected_count in COUNTS.items():
        entry = manifest["validation"] if role == "server" else manifest["clients"][role]
        if entry["count"] != expected_count:
            raise ValueError("Declared complete participant count changed.")
        records_path = source / entry["records_file"]
        records = load_own_records(bucketed["source_plan"], role, [records_path])
        ids = [row["source_id"] for row in records]
        if any(source_id in seen for source_id in ids):
            raise ValueError("Participants have overlapping source IDs.")
        seen.update(ids)
        rows[role] = {"count": len(records), "source_ids_hash": content_hash(ids),
                      "records_hash": entry["records_hash"], "records_file": base.relative(records_path),
                      "cache_file": f"{DATASET}/{entry['file']}"}
        for field in ("records_file", "file"):
            name = f"{DATASET}/{entry[field]}"
            files[name] = base.sha(ROOT / name)
        if files[rows[role]["cache_file"]] != entry["sha256"]:
            raise ValueError("A reference feature cache differs from its source manifest.")
    if len(seen) != DESIGN["total_rows"]:
        raise ValueError("Expected all 2,960 unique approved training/validation rows.")
    return {"manifest_hash": manifest["manifest_hash"], "participants": rows,
            "bucketed_plan": bucketed, "files": files}


def prepare_run(output, public=None):
    output, public = output_paths(output, public)
    model = base.model_contract()
    source = source_contract(model)
    plan = {"kind": KIND, "created_utc": utc(), "output": base.relative(output),
            "public": None if public is None else base.relative(public), "dataset": DATASET,
            "model": model, "source": source, "design": DESIGN, "runtime": runtime()}
    plan["lock_hash"] = content_hash(plan)
    output.mkdir(parents=True, exist_ok=False)
    # Both plans precede participant model construction and any generated features.
    base.write_new(output / "plan.json", plan)
    base.write_new(output / "bucketed-plan.json", source["bucketed_plan"])
    return plan


def verify_plan(plan, output):
    output = Path(output).resolve()
    if (plan.get("kind") != KIND or plan.get("dataset") != DATASET or plan.get("design") != DESIGN
            or plan.get("output") != base.relative(output)
            or content_hash({k: v for k, v in plan.items() if k != "lock_hash"}) != plan.get("lock_hash")
            or base.read(output / "plan.json") != plan or runtime() != plan["runtime"]):
        raise ValueError("Integration plan or runtime changed.")
    validate_plan(plan["source"]["bucketed_plan"])
    if base.read(output / "bucketed-plan.json") != plan["source"]["bucketed_plan"]:
        raise ValueError("Participant preparation plan changed.")
    model = base.model_contract()
    if plan.get("model") != model or plan["source"] != source_contract(model):
        raise ValueError("Required model/source contract or participant commitments changed.")
    for name, digest in plan["source"]["files"].items():
        if base.relative(ROOT / name) != name or base.sha(ROOT / name) != digest:
            raise ValueError(f"Frozen integration input changed: {name}")
    return plan


def participant_command(plan, output, role):
    if role not in COUNTS:
        raise ValueError("Unknown participant.")
    return [sys.executable, str(ROOT / "tools/prepare_encoder_bucketed.py"), "prepare",
            "--plan", str(Path(output) / "bucketed-plan.json"), "--participant", role,
            "--records", str(ROOT / plan["source"]["participants"][role]["records_file"]),
            "--asset-dir", str(ROOT / plan["model"]["asset_dir"]),
            "--output", str(Path(output) / "participants" / role / "data")]


def tokenizer_probe(plan, output):
    """Compare real pinned tokenizer special-aware truncation, with no encoder."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / plan["model"]["asset_dir"]), local_files_only=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    raw = tokenizer(list(TOKENIZER_PROBES), add_special_tokens=False, padding=False,
                    truncation=False, return_attention_mask=False)["input_ids"]
    prepared = [tokenizer.prepare_for_model(ids, add_special_tokens=True, truncation=True,
                max_length=384, return_attention_mask=True, padding=False) for ids in raw]
    canonical = tokenizer(list(TOKENIZER_PROBES), padding=False, truncation=True, max_length=384)
    specials = tokenizer.num_special_tokens_to_add(pair=False)
    lengths = [len(ids) + specials for ids in raw]
    passed = ([row["input_ids"] for row in prepared] == canonical["input_ids"]
              and [row["attention_mask"] for row in prepared] == canonical["attention_mask"]
              and any(length > 384 for length in lengths) and 384 in lengths)
    result = {"probe_hash": content_hash(TOKENIZER_PROBES), "count": len(TOKENIZER_PROBES),
              "raw_token_lengths": list(map(len, raw)), "special_tokens_per_query": specials,
              "lengths_with_special_tokens": lengths,
              "prepared_lengths": [len(row["input_ids"]) for row in prepared],
              "truncated_probe_count": sum(length > 384 for length in lengths),
              "canonical_ids_and_masks_exact": passed, "passed": passed,
              "scope": "Four authored tokenization probes; no expected labels and no encoder forward pass."}
    base.write_new(Path(output) / "tokenizer-probe.json", result)
    if not passed:
        raise ValueError("Pinned tokenizer IDs, masks, truncation or boundary-probe checks failed.")
    return result


def stop_process_tree(process):
    """Bound timeout cleanup, including a Windows virtualenv launcher child."""
    if process.poll() is not None:
        return  # A completed worker's PID may already have been reused.
    import psutil
    try:
        parent = psutil.Process(process.pid)
        targets = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        targets = []
    for target in targets:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, survivors = psutil.wait_procs(targets, timeout=3)
    for target in survivors:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(survivors, timeout=3)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()


def prepare_participant(plan, output, role):
    verify_plan(plan, output)
    folder = Path(output) / "participants" / role
    folder.mkdir(parents=True, exist_ok=False)
    command = participant_command(plan, output, role)
    base.write_new(folder / "started.json", {"lock_hash": plan["lock_hash"], "participant": role,
                   "started_utc": utc(), "command": command,
                   "scope": "Separate process on the same host; no filesystem or network isolation claim."})
    started = time.perf_counter()
    process = None
    process_wall_seconds = None
    try:
        with (folder / "process.log").open("xb") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, **plan["runtime"]["participant_environment"]},
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            code = process.wait(timeout=DESIGN["participant_timeout_seconds"])
            process_wall_seconds = time.perf_counter() - started
            if code:
                raise RuntimeError(f"Participant {role} exited with status {code}.")
        verification_started = time.perf_counter()
        registration = base.read(folder / "data/registration.json")
        preparation = base.read(folder / "data/preparation.json")
        if (registration["participant"] != role or preparation["participant"] != role
                or preparation["count"] != COUNTS[role]
                or preparation["telemetry"]["feature_settings"] != FEATURE_SETTINGS):
            raise ValueError("Participant output differs from its declared identity or recipe.")
        verify_plan(plan, output)
        verification_seconds = time.perf_counter() - verification_started
        report = {"participant": role, "status": "completed", "process_wall_seconds": process_wall_seconds,
                  "postprocess_verification_seconds": verification_seconds,
                  "total_role_wall_seconds": time.perf_counter() - started,
                  "process_timing_scope": "Child launch through process.wait completion; includes child imports, asset verification, model load, encoding, and output writes; excludes parent verification hashes.",
                  "registration_hash": registration["registration_hash"], "preparation": preparation}
        base.write_new(folder / "completed.json", report)
        return registration, report
    except BaseException as error:
        failure_observed_seconds = time.perf_counter() - started
        if process is not None:
            stop_process_tree(process)
        base.write_new(folder / "failure.json", {"participant": role, "status": "failed", "utc": utc(),
                       "lock_hash": plan["lock_hash"], "error_type": type(error).__name__,
                       "error": str(error), "process_wall_seconds": process_wall_seconds,
                       "failure_observed_seconds": failure_observed_seconds,
                       "total_role_wall_seconds_including_cleanup": time.perf_counter() - started})
        raise


def assemble_auditor_dataset(plan, output, registrations):
    """Explicitly collect public benchmark outputs for same-host auditing."""
    manifest = assemble(plan["source"]["bucketed_plan"], registrations)
    dataset = Path(output) / "auditor-dataset"
    dataset.mkdir(exist_ok=False)
    for role in COUNTS:
        entry = manifest["validation"] if role == "server" else manifest["clients"][role]
        participant = Path(output) / "participants" / role / "data"
        for field in ("file", "records_file"):
            destination = dataset / entry[field]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError("Auditor copy cannot replace a prior output.")
            shutil.copyfile(participant / entry[field], destination)
    base.write_new(dataset / "manifest.json", manifest)
    base.write_new(dataset / "audit-scope.json", {
        "scope": "One auditor copied all public benchmark features and records into this dataset on the same host.",
        "claims_not_established": ["private ownership", "filesystem isolation", "independent physical devices"],
        "manifest_hash": manifest["manifest_hash"], "plan_hash": plan["lock_hash"],
    })
    return dataset, manifest


def score_changes(original, candidate, ids):
    """Use the already fixed FP32 top-two margin rule without accuracy labels."""
    import numpy as np
    rows = [{"query_id": str(source_id), "disposition": "supported"} for source_id in ids]
    old = base.predictions_from_scores(rows, original.tolist())
    new = base.predictions_from_scores(rows, candidate.tolist())
    return {"max_logit_absolute_difference": float(np.abs(original - candidate).max()),
            "top1_changes": sum(a["prediction"] != b["prediction"] for a, b in zip(old, new)),
            "strict_margin_answer_changes": sum(a["answered"] != b["answered"] for a, b in zip(old, new))}


def feature_audit(plan, dataset):
    import numpy as np
    from fedlora_edge.data import INTENTS
    from fedlora_edge.encoder_data import load_encoder_records, load_feature_split
    from fedlora_edge.encoder_head import HeadConfig, HeadTrainer
    from fedlora_edge.encoder_inference import load_head_checkpoint

    saved = load_head_checkpoint(ROOT / plan["model"]["checkpoint_dir"])
    if saved["b_hash"] != plan["model"]["head_hash"]:
        raise ValueError("Confirmed head identity changed.")
    head = HeadTrainer(seed=42, config=HeadConfig(cpu_threads=1))
    head.set_state(saved["state"])
    original, candidate, parity = {}, {}, {}
    for role in COUNTS:
        client = None if role == "server" else role
        original[role] = load_feature_split(ROOT / DATASET, client)
        candidate[role] = load_feature_split(dataset, client)
        old_x, old_y, old_ids = original[role]
        new_x, new_y, new_ids = candidate[role]
        new_records = load_encoder_records(dataset, client)
        if (not np.array_equal(old_ids, new_ids) or not np.array_equal(old_y, new_y)
                or len(old_y) != COUNTS[role]
                or new_ids.tolist() != [row["source_id"] for row in new_records]
                or new_y.tolist() != [INTENTS.index(row["label"]) for row in new_records]
                or content_hash(old_ids.tolist()) != plan["source"]["participants"][role]["source_ids_hash"]):
            raise ValueError(f"Participant {role} source IDs, row order or labels differ.")
        difference = float(np.abs(new_x - old_x).max())
        changes = score_changes(head.predict_scores(old_x), head.predict_scores(new_x), old_ids)
        parity[role] = {"count": len(old_y), "source_ids_and_labels_exact": True,
                        "source_ids_hash": content_hash(old_ids.tolist()),
                        "max_feature_absolute_difference": difference, **changes,
                        "passed": difference <= DESIGN["feature_absolute_tolerance"]
                                  and changes["top1_changes"] == changes["strict_margin_answer_changes"] == 0}
    return original, candidate, parity


def state_difference(left, right):
    from fedlora_edge.state import finite_state, state_schema
    if state_schema(left) != state_schema(right) or not finite_state(left) or not finite_state(right):
        raise ValueError("State comparison requires matching finite schemas.")
    return max(abs(a - b) for key in left for a, b in zip(left[key].values, right[key].values))


def head_smoke(original, candidate, output):
    import numpy as np
    from fedlora_edge.aggregation import aggregate
    from fedlora_edge.encoder_head import HeadConfig, HeadTrainer
    from fedlora_edge.state import hash_state, state_nbytes, state_schema, state_to_dict, subtract_states
    from fedlora_edge.training import derive_client_seed

    output = Path(output) / "head-smoke"
    output.mkdir(exist_ok=False)
    initial = HeadTrainer(seed=42, config=HeadConfig(**HEAD)).get_state()
    initial_hash = hash_state(initial)
    if initial_hash != hash_state(HeadTrainer(seed=42, config=HeadConfig(**HEAD)).get_state()):
        raise ValueError("Seed 42 did not reproduce identical initialization tensors.")
    base.write_new(output / "initial-state.json", {"state": state_to_dict(initial), "state_hash": initial_hash})
    states, updates, arms = {}, {}, {}
    for arm, splits in (("original", original), ("bucketed", candidate)):
        updates[arm], client_work = {}, {}
        for role in list(COUNTS)[:-1]:
            trainer = HeadTrainer(seed=42, config=HeadConfig(**HEAD))
            trainer.set_state(initial)
            X, y, _ = splits[role]
            seed = derive_client_seed(42, 1, role)
            telemetry = trainer.train(X, y, 1, seed)
            state = trainer.get_state()
            updates[arm][role] = subtract_states(state, initial)
            changed = hash_state(state) != initial_hash
            if (telemetry["optimizer_steps"] != 1 or telemetry["examples_processed"] != COUNTS[role]
                    or telemetry["state_hash_before"] != initial_hash
                    or not telemetry["optimizer_state_reset"] or not changed):
                raise ValueError("One-step head smoke did not complete the declared fresh local optimization.")
            client_work[role] = {"seed": seed, "optimizer_steps": telemetry["optimizer_steps"],
                                 "examples_processed": telemetry["examples_processed"], "head_changed": changed,
                                 "incoming_state_hash": telemetry["state_hash_before"],
                                 "updated_state_hash": hash_state(state), "delta_hash": hash_state(updates[arm][role]),
                                 "raw_delta_bytes": state_nbytes(updates[arm][role])}
            base.write_new(output / f"{arm}-client-{role}.json",
                           {"telemetry": telemetry, "delta": state_to_dict(updates[arm][role])})
        result = aggregate(initial, updates[arm], strategy="equal", sample_sizes={role: COUNTS[role] for role in updates[arm]})
        states[arm] = result.state
        if result.telemetry["coefficients"] != {role: 0.2 for role in updates[arm]}:
            raise ValueError("Expected equal weights for all five clients.")
        arms[arm] = {"clients": client_work, "aggregate_state_hash": hash_state(result.state),
                     "aggregate_changed": hash_state(result.state) != initial_hash,
                     "aggregation": result.telemetry}
        base.write_new(output / f"{arm}-aggregate.json", {"state": state_to_dict(result.state), "state_hash": hash_state(result.state)})
    old_head = HeadTrainer(seed=42, config=HeadConfig(**HEAD))
    new_head = HeadTrainer(seed=42, config=HeadConfig(**HEAD))
    old_head.set_state(states["original"])
    new_head.set_state(states["bucketed"])
    original_val, candidate_val = original["server"][0], candidate["server"][0]
    old_scores = old_head.predict_scores(original_val)
    new_scores = new_head.predict_scores(candidate_val)
    common_scores = new_head.predict_scores(original_val)
    changes = int((old_scores.argmax(1) != new_scores.argmax(1)).sum())
    common_changes = int((old_scores.argmax(1) != common_scores.argmax(1)).sum())
    delta_differences = {role: state_difference(updates["original"][role], updates["bucketed"][role])
                         for role in list(COUNTS)[:-1]}
    aggregate_difference = state_difference(states["original"], states["bucketed"])
    state_tolerance = DESIGN["head_smoke"]["state_absolute_tolerance"]
    result = {"kind": "one_round_mathematical_head_compatibility", "initial_state_hash": initial_hash,
              "initial_seed": 42, "config": HEAD, "state_schema": state_schema(initial),
              "raw_state_bytes": state_nbytes(initial), "arms": arms,
              "max_client_delta_absolute_difference": delta_differences,
              "max_aggregate_state_absolute_difference": aggregate_difference,
              "state_absolute_tolerance": state_tolerance,
              "validation_prediction_changes": changes, "common_validation_prediction_changes": common_changes,
              "max_validation_logit_absolute_difference": float(np.abs(old_scores - new_scores).max()),
              "passed": all(arm["aggregate_changed"] for arm in arms.values()) and changes == common_changes == 0
                        and max(delta_differences.values()) <= state_tolerance and aggregate_difference <= state_tolerance,
              "scope": "One actual SGD step per client and equal tensor aggregation; no Flower transport, convergence, or new accuracy measured."}
    base.write_new(output / "result.json", result)
    return result


def publish_summary(plan, output, result):
    result = {**result, "result_hash": content_hash(result)}
    base.write_new(Path(output) / "result.json", result)
    if plan["public"] is not None:
        public = ROOT / plan["public"]
        public.parent.mkdir(parents=True, exist_ok=True)
        base.write_new(public, result)
    return result


def run(output, public=None):
    plan = prepare_run(output, public)
    output = ROOT / plan["output"]
    base.write_new(output / "started.json", {"lock_hash": plan["lock_hash"], "started_utc": utc()})
    started = time.perf_counter()
    reports, registrations = {}, []
    stage = "authored_tokenizer_parity"
    try:
        probes = tokenizer_probe(plan, output)
        stage = "participant_preparation"
        for role in COUNTS:
            print(f"Preparing participant {role}: {COUNTS[role]} approved rows", flush=True)
            registration, reports[role] = prepare_participant(plan, output, role)
            registrations.append(registration)
        stage = "metadata_assembly_and_feature_audit"
        dataset, manifest = assemble_auditor_dataset(plan, output, registrations)
        original, candidate, parity = feature_audit(plan, dataset)
        stage = "one_round_head_compatibility"
        smoke = head_smoke(original, candidate, output)
        stage = "final_identity_verification"
        verify_plan(plan, output)
        passed = all(row["passed"] for row in parity.values()) and smoke["passed"]
        summary = {"kind": KIND, "status": "completed" if passed else "failed_compatibility",
                   "run_id": output.name, "created_utc": utc(), "lock_hash": plan["lock_hash"],
                   "source_manifest_hash": plan["source"]["manifest_hash"], "manifest_hash": manifest["manifest_hash"],
                   "bucketed_plan_hash": plan["source"]["bucketed_plan"]["plan_hash"],
                   "recipe_hash": plan["source"]["bucketed_plan"]["recipe_hash"],
                   "confirmed_head_hash": plan["model"]["head_hash"], "rows_checked": sum(COUNTS.values()),
                   "tokenizer_probes": probes, "feature_parity": parity, "head_smoke": smoke,
                   "preparation": {role: {"count": COUNTS[role], "process_wall_seconds": reports[role]["process_wall_seconds"],
                       "postprocess_verification_seconds": reports[role]["postprocess_verification_seconds"],
                       "total_role_wall_seconds": reports[role]["total_role_wall_seconds"],
                       "process_timing_scope": reports[role]["process_timing_scope"],
                       "registration_hash": reports[role]["registration_hash"],
                       "encoding_seconds_including_asset_verification_and_model_loading": reports[role]["preparation"]["telemetry"]["encoding_seconds_including_asset_verification_and_model_loading"],
                       "work": reports[role]["preparation"]["telemetry"]["work"],
                       "memory": reports[role]["preparation"]["telemetry"]["memory"]} for role in COUNTS},
                   "wall_seconds": time.perf_counter() - started, "source_hashes": {name: plan["source"]["files"][name] for name in SOURCES},
                   "reproducibility": {"dataset": plan["dataset"],
                       "model_identity": {key: value for key, value in plan["model"].items() if key != "files"},
                       "input_hashes": plan["source"]["files"],
                       "participant_source_commitments": plan["source"]["participants"],
                       "bucketed_plan": plan["source"]["bucketed_plan"],
                       "lock_created_utc": plan["created_utc"], "private_run_relative_path": plan["output"],
                       "public_summary_relative_path": plan["public"]},
                   "runtime": plan["runtime"], "design": DESIGN,
                   "limits": ["Existing public training and reused validation only; no old test, confirmation, CLINC, or scope inference.",
                              "Serial fresh participant processes on one host, without filesystem or network isolation; offline model loading is configured.",
                              "Auditor dataset explicitly contains every participant's public benchmark features and records.",
                              "Fixed-head changes and one-round compatibility are not accuracy, convergence, Flower runtime, privacy, or physical-laptop evidence.",
                              "Token slots and attention cells are arithmetic work proxies; raw state bytes are not network traffic."]}
        return publish_summary(plan, output, summary)
    except BaseException as error:
        failure = {"kind": KIND, "status": "failed", "run_id": output.name, "lock_hash": plan["lock_hash"],
                   "stage": stage, "failed_utc": utc(), "error_type": type(error).__name__, "error": str(error),
                   "completed_participants": list(reports), "wall_seconds": time.perf_counter() - started,
                   "scope": DESIGN["scope"]}
        base.write_new(output / "failure.json", failure)
        if plan["public"] is not None and not (ROOT / plan["public"]).exists():
            public_path = ROOT / plan["public"]
            public_path.parent.mkdir(parents=True, exist_ok=True)
            base.write_new(public_path, failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public", type=Path)
    args = parser.parse_args()
    result = run(args.output, args.public)
    print(json.dumps({"status": result["status"], "rows_checked": result["rows_checked"],
                      "lock_hash": result["lock_hash"], "manifest_hash": result["manifest_hash"]}), flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
