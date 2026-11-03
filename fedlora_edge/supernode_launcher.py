"""Restart a registered Flower node across its bounded heartbeat lease.

Flower 1.40 rejects activation while an abruptly stopped instance's lease is
still ONLINE (HTTP 412, code 2004), and its HTTP transport does not retry that
response. Keep the same key and node registration, retry only that response,
and fail after a fixed startup window. No installed Flower files are patched.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time


def is_lease_conflict(returncode: int, output: str) -> bool:
    # Flower logs its diagnostic code 305 but flwr_exit maps errors to OS
    # status 1. Require both diagnostic markers, not a generic nonzero exit.
    return (returncode == 1 and "Exit Code: 305" in output
            and "[code: 2004] Failed to activate SuperNode." in output)


def _run_once(command: list[str], startup_seconds: float = 75) -> tuple[int, str]:
    """Drain logs without waiting for an orphan SuperExec to close stdout.

    A new POSIX process group belongs only to this launched node. Flower may
    start SuperExec concurrently with an activation failure; reap that group
    after the node exits so its inherited stdout and retry loop cannot linger.
    """
    if os.name != "posix":
        raise RuntimeError("Use the tested Linux/WSL runtime for this launcher")
    tail: deque[str] = deque(maxlen=100)
    active = threading.Event()
    deadline = time.monotonic() + startup_seconds
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, start_new_session=True) as process:
        assert process.stdout is not None

        def drain() -> None:
            for line in process.stdout:
                tail.append(line)
                print(line, end="", flush=True)
                # Flower 1.40 start_client_internal logs this only after its
                # Fleet connection yields an activated, authenticated node ID.
                if re.search(r"SuperNode ID: [0-9]+", line):
                    active.set()

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        try:
            while process.poll() is None and not active.is_set() and time.monotonic() < deadline:
                active.wait(timeout=min(.05, max(0, deadline-time.monotonic())))
            timed_out = process.poll() is None and not active.is_set()
            if timed_out:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            returncode = process.wait()
        finally:
            # Also runs when Ctrl-C interrupts a healthy node's long wait.
            # The child has its own session, so the terminal signal alone
            # does not stop it or its SuperExec descendants.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            reader.join(timeout=5)
            if reader.is_alive():
                raise RuntimeError("Node stdout remained open after process-group cleanup")
    if timed_out:
        print(json.dumps({"event": "supernode_startup_timeout", "startup_seconds": startup_seconds}), flush=True)
    return (124 if timed_out else returncode), "".join(tail)


def run(command: list[str], activation_wait_seconds: float = 75) -> int:
    if not command or not math.isfinite(activation_wait_seconds) or activation_wait_seconds <= 0:
        raise ValueError("Require a command and a positive finite activation window")
    started = time.monotonic()
    retries = 0
    returncode = 124
    while True:
        remaining = activation_wait_seconds - (time.monotonic() - started)
        if remaining <= 0:
            print(json.dumps({"event": "activation_lease_retry_exhausted", "retries": retries,
                              "timeout_seconds": activation_wait_seconds}), flush=True)
            return returncode
        returncode, output = _run_once(command, remaining)
        if not is_lease_conflict(returncode, output):
            return returncode
        remaining = activation_wait_seconds - (time.monotonic() - started)
        if remaining <= 0:
            print(json.dumps({"event": "activation_lease_retry_exhausted", "retries": retries,
                              "timeout_seconds": activation_wait_seconds}), flush=True)
            return returncode
        retries += 1
        delay = min(2.0, remaining)
        print(json.dumps({"event": "activation_lease_retry", "retry": retries,
                          "delay_seconds": delay, "remaining_seconds": remaining}), flush=True)
        time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-wait-seconds", type=float, default=75)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    sys.exit(run(command, args.activation_wait_seconds))


if __name__ == "__main__":
    main()
