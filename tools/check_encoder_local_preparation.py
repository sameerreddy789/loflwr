"""Measure real local encoding in isolated, networkless Linux participant processes.

The parent is a public-benchmark auditor/provisioner. Participant namespaces hide
the entire original repository and mount only code, pinned model, own records,
and own output. No client reads another participant's data during preparation.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fedlora_edge.data import content_hash, write_json
from fedlora_edge.encoder_data import load_encoder_manifest, load_feature_split, sha256
from fedlora_edge.encoder_local_data import build_local_plan, prepare_local_features, assemble_local_manifest


def read(path):
    return json.loads(Path(path).read_text())


def sandbox_command(role, project, model, python, participant):
    return ["bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-net",
        "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--tmpfs", str(project), "--ro-bind", str(project / "fedlora_edge"), "/tmp/app/fedlora_edge",
        "--ro-bind", str(project / "tools"), "/tmp/app/tools", "--ro-bind", str(model), "/tmp/model",
        "--ro-bind", str(role / "input"), "/tmp/input", "--bind", str(role / "output"), "/tmp/output",
        "--dir", "/tmp/home", "--setenv", "HOME", "/tmp/home", "--setenv", "HF_HUB_OFFLINE", "1",
        "--setenv", "TRANSFORMERS_OFFLINE", "1", "--setenv", "TOKENIZERS_PARALLELISM", "false",
        "--setenv", "PYTHONPATH", "/tmp/app", "--chdir", "/tmp/app", "--", python,
        "/tmp/app/tools/check_encoder_local_preparation.py", "worker", "--participant", participant,
        "--hidden-project", str(project)]


def worker(participant, hidden_project):
    plan = read("/tmp/input/plan.json")
    files = sorted(Path("/tmp/input").glob("records_*.json"))
    forbidden = [Path(hidden_project) / "assets/encoder-bank20-v1/manifest.json",
                 Path(hidden_project) / "assets/encoder-bank20-v1/confirmation.json",
                 Path(hidden_project) / "runs/encoder-flower-balanced-v2/selected/checkpoint.json"]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("Original repository data remains visible in the namespace")
    if set(path.name for path in Path("/tmp/input").iterdir()) != {"plan.json", *(p.name for p in files)}:
        raise ValueError("Unexpected input files in participant package")
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=0.5):
            raise RuntimeError("Unexpected network connectivity")
    except OSError as error:
        network_error = str(error)
    registration = prepare_local_features(plan, files, "/tmp/output/data", "/tmp/model", participant)
    write_json("/tmp/output/isolation.json", {"participant": participant, "own_input_files": [p.name for p in files],
        "hidden_paths_absent": [str(p) for p in forbidden], "network_error": network_error,
        "pid_namespace": True, "original_repository_masked": True, "registration_hash": registration["registration_hash"],
        "scope": "Linux mount/PID/network namespaces on one physical laptop, not independent physical devices"})


def run(output, layout, source, model):
    if sys.platform != "linux" or not shutil.which("bwrap"):
        raise RuntimeError("Run in the tested Ubuntu WSL environment with bubblewrap")
    if output.exists():
        raise FileExistsError("Choose a new local preparation run")
    source = source.resolve()
    manifest = load_encoder_manifest(source)
    plan = build_local_plan(manifest, layout)
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    started = time.perf_counter()
    registrations, reports = [], {}
    for participant, descriptor in plan["participants"].items():
        role = output / "participants" / participant
        (role / "input").mkdir(parents=True)
        (role / "output").mkdir()
        write_json(role / "input/plan.json", plan)
        for i, part in enumerate(descriptor["parts"]):
            entry = manifest["validation"] if part["original_client"] is None else manifest["clients"][part["original_client"]]
            shutil.copyfile(source / entry["records_file"], role / "input" / f"records_{i}.json")
        command = sandbox_command(role.resolve(), ROOT, model.resolve(), sys.executable, participant)
        write_json(role / "command.json", command)
        print(f"Preparing {layout} participant {participant}: own records only, no network", flush=True)
        with (role / "process.log").open("wb") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900)
        registrations.append(read(role / "output/data/registration.json"))
        reports[participant] = read(role / "output/data/preparation.json")
    assembled = assemble_local_manifest(plan, registrations)
    for participant in plan["participants"]:
        write_json(output / "participants" / participant / "output/data/manifest.json", assembled)
    write_json(output / "manifest.json", assembled)
    # Public audit only: compare new independently computed features to the fixed
    # reference, with source-ID alignment and no test or confirmation file access.
    import numpy as np
    from fedlora_edge.encoder_inference import load_head_checkpoint
    from fedlora_edge.encoder_head import HeadTrainer
    saved = load_head_checkpoint(ROOT / "runs/encoder-flower-balanced-v2/selected")
    trainer = HeadTrainer()
    trainer.set_state(saved["state"])
    parity = {}
    for participant, descriptor in plan["participants"].items():
        data = output / "participants" / participant / "output/data"
        X, y, ids = load_feature_split(data, None if participant == "server" else participant)
        pieces = [load_feature_split(source, part["original_client"]) for part in descriptor["parts"]]
        old_x, old_y, old_ids = [np.concatenate([piece[i] for piece in pieces]) for i in range(3)]
        if not np.array_equal(ids, old_ids) or not np.array_equal(y, old_y):
            raise ValueError("Reference identity/order or labels differ")
        delta = float(np.abs(X-old_x).max())
        changed = int((trainer.predict_scores(X).argmax(1) != trainer.predict_scores(old_x).argmax(1)).sum())
        if delta > 1e-5 or changed:
            raise ValueError(f"Predeclared feature/prediction parity gate failed: {participant}: {delta}, {changed}")
        parity[participant] = {"count": len(y), "maximum_absolute_feature_difference": delta,
                               "changed_head_predictions": changed, "absolute_tolerance": 1e-5}
    result = {"status": "completed", "run_id": output.name, "layout": layout,
        "plan_hash": plan["plan_hash"], "manifest_hash": assembled["manifest_hash"],
        "source_manifest_hash": manifest["manifest_hash"], "reference_head_hash": saved["b_hash"],
        "wall_seconds": time.perf_counter()-started, "parity": parity, "preparation": reports,
        "registrations": registrations, "source_hashes": {p: sha256(ROOT / p) for p in (
            "fedlora_edge/encoder_local_data.py", "tools/check_encoder_local_preparation.py")},
        "limits": ["Public records provisioned by an auditor before isolation; production data ownership not established.",
                   "One physical laptop. Registration size is metadata, not training payload or a privacy guarantee.",
                   "Only training/validation features compared; old confirmation/external tests not scored."]}
    write_json(output / "result.json", result)
    print(json.dumps({"status": "completed", "layout": layout, "parity": parity}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "worker"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layout", choices=["five", "three"], default="five")
    parser.add_argument("--source", type=Path, default=ROOT / "assets/encoder-bank20-v1")
    parser.add_argument("--model", type=Path, default=ROOT / "assets/reference-mpnet")
    parser.add_argument("--participant")
    parser.add_argument("--hidden-project")
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.participant, args.hidden_project)
    elif args.output is None:
        parser.error("--output is required")
    else:
        run(args.output.resolve(), args.layout, args.source, args.model)


if __name__ == "__main__":
    main()
