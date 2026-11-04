"""Read-only readiness check for the confirmed local banking demonstration.

The isolated inference probe uses an authored functional input, never evaluation
queries. No installation, downloads, training, listener termination or TLS setup.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIRMATION_LOCK = "81fedc0777a0c98f02d071b9dd3adaef862c52c538a165ccc75d2b4f01187681"
PROBE_TEXT = "My card payment is still pending."
PACKAGES = ("torch", "transformers", "peft", "tokenizers", "safetensors", "numpy",
            "streamlit", "pandas", "altair", "psutil")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_check(root=ROOT):
    pins_file = root / "requirements" / ("cpu-windows-py311.txt" if os.name == "nt" else "cuda-wsl-py311.txt")
    pins = dict(line.split("==", 1) for line in pins_file.read_text().splitlines() if "==" in line)
    versions, missing, drift = {}, [], {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
            if versions[name] != pins.get(name):
                drift[name] = {"installed": versions[name], "tested": pins.get(name)}
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    return {"versions": versions, "missing": missing, "drift": drift, "pins": pins_file.name}


def dashboard_config_matches(arguments, cwd, root=ROOT, *, run_dir=None, model_dir=None, policy_file=None):
    """Match effective app paths, including omitted defaults and --flag=value."""
    defaults = {
        "--run-dir": root / "runs/encoder-flower-balanced-v2",
        "--model-dir": root / "assets/reference-mpnet",
        "--policy-file": root / "research/results/encoder-v2/final-policy.json",
    }
    actual = dict(defaults)
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        key, separator, value = argument.partition("=")
        if key not in defaults and key != "--" and key.startswith("--") and any(
                option.startswith(key) for option in defaults):
            # The dashboard accepts argparse abbreviations. Require canonical
            # flags here so an uninspected custom path cannot be marked ready.
            return False
        if key in defaults:
            if not separator:
                index += 1
                if index >= len(arguments) or arguments[index].startswith("--"):
                    return False
                value = arguments[index]
            if not value and key != "--policy-file":
                return False
            actual[key] = Path(cwd) / value if value else defaults[key]
        index += 1
    expected = dict(zip(defaults, (run_dir, model_dir, policy_file)))
    return all(Path(actual[key]).resolve() == (root / (expected[key] or defaults[key])).resolve()
               for key in defaults)


def listener_is_demo(port, root=ROOT, *, run_dir=None, model_dir=None, policy_file=None):
    """A healthy arbitrary Streamlit server is not enough to identify this app."""
    try:
        import psutil
        target = (root / "fedlora_edge/encoder_dashboard.py").resolve()
        for connection in psutil.net_connections(kind="tcp"):
            if (connection.status != psutil.CONN_LISTEN or not connection.laddr
                    or connection.laddr.port != port or not connection.pid
                    or connection.laddr.ip not in {"127.0.0.1", "0.0.0.0", "::", "::1"}):
                continue
            process = psutil.Process(connection.pid)
            command = process.cmdline()
            if "streamlit" not in command or "run" not in command:
                continue
            index = command.index("run") + 1
            if (index < len(command) and (Path(process.cwd()) / command[index]).resolve() == target
                    and dashboard_config_matches(command[index + 1:], process.cwd(), root,
                        run_dir=run_dir, model_dir=model_dir, policy_file=policy_file)):
                return True
    except (ImportError, OSError, ValueError):
        return False
    except Exception as error:
        # Process inspection can race with exit or be denied by the OS.
        if error.__class__.__module__.startswith("psutil"):
            return False
        raise
    return False


def check_port(port, root=ROOT, *, run_dir=None, model_dir=None, policy_file=None):
    if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ValueError("Use a port from 1024 through 65535.")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        if os.name == "nt":
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            listener.bind(("127.0.0.1", port))
            return {"state": "available", "ok": True, "action": "launch"}
        except OSError:
            pass
    if listener_is_demo(port, root, run_dir=run_dir, model_dir=model_dir, policy_file=policy_file):
        try:
            # Bypass machine proxy settings; this request stays on loopback.
            with build_opener(ProxyHandler({})).open(f"http://127.0.0.1:{port}/_stcore/health", timeout=3) as response:
                if response.status == 200 and response.read(32).strip() == b"ok":
                    return {"state": "this_demo_running", "ok": True, "action": "open_existing"}
        except OSError:
            pass
    return {"state": "occupied_or_unavailable", "ok": False,
            "action": "Use another port or inspect its listener. Nothing was stopped."}


def evidence_check(root=ROOT):
    from fedlora_edge.data import content_hash
    from fedlora_edge.demo_presets import load_comparisons, load_recovery, load_scope
    plan = read(root / "research/results/encoder-v2/confirmation-plan.json")
    if (plan["lock_hash"] != CONFIRMATION_LOCK
            or content_hash({k: v for k, v in plan.items() if k != "lock_hash"}) != CONFIRMATION_LOCK):
        raise ValueError("The original confirmation plan differs from its published lock.")
    delivery = read(root / "research/results/demo-delivery-v1/verification.json")
    if sha(root / "presentation/fedlora-edge-demo.pptx") != delivery["slides"]["pptx_sha256"]:
        raise ValueError("Judging slides differ from their verified delivery. Review the replacement before presenting.")
    if sha(root / "presentation/demo-backup.webm") != delivery["recording"]["sha256"]:
        raise ValueError("Original backup recording differs from its verified delivery.")
    return {"confirmation_lock": CONFIRMATION_LOCK,
            "recovery_runs": [load_recovery(layout, root)["run_id"] for layout in ("five", "three")],
            "matrix_cells": load_comparisons(root)["completed"], "scope_queries": load_scope(root)["count"],
            "presentation_files_verified": 2}


def inference_probe(run_dir, model_dir, policy_file):
    from fedlora_edge.encoder_dashboard import load_evidence, load_margin_threshold
    from fedlora_edge.encoder_inference import EncoderPredictor
    import psutil
    evidence, saved = load_evidence(run_dir)
    plan = read(ROOT / "research/results/encoder-v2/confirmation-plan.json")
    if (saved["b_hash"] != plan["head_hash"] or saved["config_hash"] != plan["config_hash"]
            or saved["version"] != 500 or saved["run_id"] != "encoder-flower-balanced-v2"):
        raise ValueError("The query preset requires the independently confirmed checkpoint, version 500.")
    threshold = load_margin_threshold(policy_file, saved)
    start = time.perf_counter()
    predictor = EncoderPredictor(model_dir, Path(run_dir) / "selected", cpu_threads=2)
    loaded = time.perf_counter() - start
    prediction = predictor.query(PROBE_TEXT)
    if prediction["head_hash"] != saved["b_hash"]:
        raise ValueError("Checkpoint changed during the probe.")
    return {"run_id": saved["run_id"], "head_hash": saved["b_hash"], "version": saved["version"],
            "threshold": threshold, "predicted_intent": prediction["intent"],
            "model_load_seconds": loaded, "query_seconds": prediction["seconds"],
            "sampled_process_rss_bytes": psutil.Process().memory_info().rss,
            "asset_hash": predictor.asset_identity["asset_hash"],
            "probe_purpose": "Authored functional input; not a new accuracy or coverage evaluation"}


def run_probe(run_dir, model_dir, policy_file, timeout):
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false", PYTHONIOENCODING="utf-8")
    command = [sys.executable, str(ROOT / "tools/demo_preflight.py"), "--worker", "--run-dir", str(run_dir),
               "--model-dir", str(model_dir), "--policy-file", str(policy_file)]
    try:
        child = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Inference probe exceeded {timeout:g} seconds and was stopped."}
    try:
        result = json.loads(child.stdout.strip().splitlines()[-1])
        if not isinstance(result, dict) or not isinstance(result.get("ok"), bool):
            raise ValueError("Invalid probe envelope")
        if child.returncode != 0:
            result["ok"] = False
        return result
    except (ValueError, IndexError):
        return {"ok": False, "error": "The prepared Python environment could not complete its inference probe."}


def run_checks(run_dir, model_dir, policy_file, port=8511, timeout=120):
    checks = []
    def record(name, ok, detail, fix=None, warning=False):
        checks.append({"check": name, "status": "warning" if ok and warning else "pass" if ok else "fail",
                       "detail": detail, "fix": fix if not ok else None})
    record("python", sys.version_info >= (3, 11), {"version": platform.python_version()},
           "Run with the prepared Python 3.11+ environment in requirements/README.md.")
    dependencies = dependency_check()
    record("dependencies", not dependencies["missing"], dependencies,
           "Select the prepared CPU environment; install the tested pins before going offline.",
           warning=bool(dependencies["drift"]))
    try:
        record("published_evidence", True, evidence_check())
    except (OSError, ValueError, KeyError, TypeError) as error:
        record("published_evidence", False, str(error), "Restore the committed public evidence and presentation files.")
    port_result = check_port(port, run_dir=run_dir, model_dir=model_dir, policy_file=policy_file)
    record("local_port", port_result["ok"], port_result, port_result["action"])
    if all(row["status"] != "fail" for row in checks if row["check"] in {"python", "dependencies", "published_evidence"}):
        result = run_probe(run_dir, model_dir, policy_file, timeout)
        record("offline_model_and_inference", result["ok"], result,
               "Use the cached pinned MPNet files and confirmed selected checkpoint; see docs/demo-preflight.md.")
    else:
        record("offline_model_and_inference", False, "Prerequisites failed; probe was not started.",
               "Fix the failed prerequisite checks and rerun.")
    ready = all(row["status"] != "fail" for row in checks)
    return {"kind": "local_demo_preflight_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "ready_with_warnings" if ready and any(r["status"] == "warning" for r in checks)
                      else "ready" if ready else "blocked",
            "ready": ready, "checks": checks, "url": f"http://127.0.0.1:{port}/",
            "next_action": port_result["action"] if ready else "Resolve the failed checks before presenting.",
            "limits": ["Readiness is a point-in-time check; port or files can change after it returns.",
                       "No physical-laptop, fresh-query evaluation, network isolation or live TLS test occurs here.",
                       "The inference probe forces local cached loading. OS-level internet disconnection is a separate rehearsal.",
                       "Observed probe time and sampled RSS are not an isolated performance benchmark."]}


def save_report(path, report):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Preflight reports must stay inside ignored runs/.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/encoder-flower-balanced-v2")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "assets/reference-mpnet")
    parser.add_argument("--policy-file", type=Path, default=ROOT / "research/results/encoder-v2/final-policy.json")
    parser.add_argument("--port", type=int, default=8511)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        try:
            result = {"ok": True, "measurement": inference_probe(args.run_dir, args.model_dir, args.policy_file)}
        except Exception as error:
            result = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:500]}"}
        print(json.dumps(result, allow_nan=False))
        return 0 if result["ok"] else 2
    if not 10 <= args.timeout <= 300 or not 1024 <= args.port <= 65535:
        parser.error("Use timeout 10–300 seconds and port 1024–65535.")
    report = run_checks(args.run_dir, args.model_dir, args.policy_file, args.port, args.timeout)
    output = args.output or ROOT / "runs" / f"demo-preflight-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}.json"
    try:
        save_report(output, report)
    except (OSError, ValueError) as error:
        print(f"Cannot save readiness report: {error}", file=sys.stderr)
        return 2
    for row in report["checks"]:
        print(f"{row['status'].upper():7} {row['check']}" + (f" — {row['fix']}" if row['fix'] else ""))
    print(f"{report['status']}: {report['next_action']}\nReport: {output}")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
