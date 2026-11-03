# Isolated deployment and recovery evidence

Measured on 3 November 2026. The three-client recovery, five-client recovery,
and fresh offline-start gates passed on **one physical laptop**. These are real
Flower Deployment Runtime processes with TLS and registered node keys. They do
not establish recovery across three physical laptops.

This report concerns the frozen-MPNet classifier-head variant: 15,380 trainable
parameters and a 61,520-byte raw FP32 head in each direction. It is not LoRA.
Each rehearsal starts from the previously confirmed head without changing its
parameters during migration to the local-data identity. Subsequent recovery
updates are separate checkpoints; they do not replace the confirmed demo model.

## Completed gates

| Run ID | Accepted contributors by attempt | Quorum | End-to-end seconds | Largest collection, seconds | Evidence |
|---|---|---:|---:|---:|---|
| `encoder-isolated-three-v6` | 3, 2, 3, 1, 3 | 2 | 236.23 | 60.001 | [Three-client export](three-recovery.json) |
| `encoder-isolated-five-v8` | 5, 4, 5, 2, 5 | 3 | 303.35 | 60.001 | [Five-client export](five-recovery.json) |
| `encoder-isolated-offline-three-v1` | 3 | 2 | 53.78 | 20.013 | [Offline export](offline.json) |

The exports contain result/package SHA-256 hashes, packaged source hashes,
configuration and manifest identities, round versions, completed steps,
aggregation coefficients, synchronized-head hashes, isolation checks and timing.
Full logs remain in the corresponding ignored `runs/<run-id>/` directories.
Exports contain no raw queries, feature arrays or private key bytes.

For both recovery runs:

- Attempt 2 kills client 1's whole SuperNode process tree after one of its two
  optimizer steps. Survivors commit with equal coefficients: 1/2 for the
  three-client layout and 1/4 for the five-client layout.
- Attempt 3 accepts every client again. The returning client's request version
  and incoming head hash match the current committed state.
- Attempt 4 kills enough clients to miss quorum. Other killed clients have a
  one-step budget and are stopped after that step but before their reply.
  This attempt retains the previous checkpoint and version.
- Attempt 5 accepts every client again. The three-client run records three
  verified rejoin events across attempts 3 and 5; the five-client run records four.
- Every accepted client's reported pre-training head hash was checked against
  the corresponding current global checkpoint, including surviving clients.

Each role has a separate mount/PID namespace. The original repository data paths
are hidden, a client sees only its own feature archive, and the server sees only
validation features. Each node mounts its own private key and the CA certificate.
This checks access to the prepared public benchmark packages; it does not prove
ownership of data in a production installation or provide differential privacy.

The offline gate starts fresh SuperLink, SuperNode and application processes in
a new network namespace containing only `lo`. A connection to `1.1.1.1:443` fails
with `Network is unreachable`. The one-round run then completes using cached
dependencies and locally prepared features. Feature extraction is a separate
gate recorded under `runs/encoder-local-three-v1/` and
`runs/encoder-local-five-v1/`; this offline run trains the cached-feature head.

## Timing, memory and communication

Training reply collection and message TTL remain 60 seconds. Readiness checks
have a separate 90-second bound; the activation launcher has a 75-second startup
budget. The overall rehearsal supervisor is bounded at 900 seconds. These are
ordinary operating-system deadlines, not hard real-time guarantees.

Measured rejoin-readiness waits were 15.52 and 7.91 seconds for three clients,
and 41.12 and 19.97 seconds for five clients. Startup, readiness, evaluation and
inter-attempt pauses are included in the end-to-end time above.

| Run | Server peak RSS, MiB | Largest client-role peak RSS, MiB | Serialized training requests, bytes | Serialized training replies, bytes |
|---|---:|---:|---:|---:|
| Three-client recovery | 915.55 | 984.24 | 938,212 | 774,399 |
| Five-client recovery | 925.53 | 1,038.93 | 1,563,655 | 1,354,157 |
| Fresh offline start | 908.26 | 1,024.18 | 187,667 | 193,572 |

RSS is the sampled sum within each role's process tree; shared pages may be
counted more than once. Peaks across roles need not occur simultaneously. These
numbers exclude the earlier encoder feature-preparation stage. Communication
figures are serialized Flower message/protobuf sizes, **not packet-captured wire
totals**: TLS/HTTP framing, authentication, readiness/control traffic and one-time
assets are excluded.

All evaluated checkpoints retained 95.5% accuracy on the reused 200-query
validation set. This rehearsal is a recovery check, not a new accuracy study or
fresh confirmation result. The existing 94% confirmation remains separate.

## Why the earlier recovery attempts failed

Installed Flower 1.40 source establishes the lifecycle behavior:

1. `common/constant.py` sets a 30-second heartbeat interval and patience of two.
   `server/superlink/linkstate/in_memory_linkstate.py::activate_node` rejects
   activation while a registered node is still ONLINE. A killed process can
   therefore leave a lease lasting up to 60 seconds after its last heartbeat.
2. `supernode/fleet_http_connection.py` reports activation failure as diagnostic
   305 / API error 2004. `supercore/exit/exit.py::flwr_exit` maps error diagnostics
   to **OS exit status 1**, rather than returning diagnostic 305 as the OS status.
3. A spawned SuperExec can retain the failed node's stdout pipe. A launcher that
   waits for pipe EOF before checking the child can stall instead of restarting.

The [bounded launcher](../../../fedlora_edge/supernode_launcher.py) retries only
the matching activation failure while preserving the registered key and node
ID. It runs each node in its own POSIX process group, cleans up that group on
failure or Ctrl-C, and bounds startup until Flower logs an activated SuperNode
ID. A successfully activated node can continue beyond the startup deadline.
The [deployment regression tests](../../../tests/test_encoder_deployment.py)
cover lease retries, a hung startup, inherited open stdout and Ctrl-C cleanup.

The following unsuccessful runs are retained. Names in this table have the
prefix `encoder-isolated-` under `runs/`.

| Diagnostic runs | Observed outcome and correction |
|---|---|
| `five-v1` | Startup could not find `flower-superexec`; the isolated PATH now includes the tested virtual environment. |
| `five-v2`, `five-v3`; `three-v1` | Counts were 5,4,4,2,3 and 3,2,3,1,1 respectively. Waiting for identity alone did not restart a process that had exited on activation failure. |
| `three-v2`, `five-v4` | Aborted while correcting the matcher to the actual redacted client error text. |
| `three-v3`, `five-v5` | Aborted after reproducing an inherited stdout pipe held by SuperExec; process-group cleanup was added. |
| `three-v4`, `five-v6` | Concurrent deployments plus other work missed the fixed collection deadline: the five-client initial attempt received one contributor. Subsequent gates ran sequentially with the same 60-second deadline. Resource contention is plausible; an OOM event was not established. |
| `three-v5` | Aborted after finding the diagnostic-code/OS-status mismatch. The final matcher uses the verified OS status and both diagnostic markers. |

## Reproduction and remaining work

Use [the laptop rehearsal guide](../../../docs/encoder-laptop-rehearsal.md) and
the [isolated deployment runner](../../../tools/run_encoder_deployment.py).
Prepare a new run directory for every execution; completed and failed evidence
is not overwritten. Run the recovery layouts sequentially on this machine.

Still required: an actual three-laptop LAN test, host-specific memory/step-budget
calibration, unplugging/rejoining a physical client, and rehearsal of the judging
presentation. The full five-attempt recovery runs take four to five minutes;
a three-minute pitch should show completed results and one dropout/rejoin segment
or a clearly labelled replay. None of these deployment results establishes
99–100% intent accuracy, secure aggregation, or protection against malicious nodes.
