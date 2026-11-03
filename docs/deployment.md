# Prepare a manual three-laptop rehearsal

**Status: untested across physical laptops.** These instructions are based on the current project code and the installed Flower 1.40.0 CLI. They are a preparation recipe, not a record of successful LAN deployment. Use the [one-laptop morning guide](morning-demo.md) for the already-prepared demo and the [results register](../research/results/README.md) for measured evidence.

`python -m fedlora_edge.federation deploy-local --layout three` remains a localhost rehearsal. It creates localhost-only certificates and launches all processes on the current host. Changing `--layout` groups the five dataset domains into three clients; it does not distribute those processes to remote machines.

## Topology and prerequisites

| Laptop | Logical client | Training domains | Other work |
| --- | --- | --- | --- |
| Host | `0` | Card lifecycle and card payments | SuperLink, ServerApp, validation, dashboard |
| CPU laptop 1 | `1` | Transfers and cash withdrawals | SuperNode and local training |
| CPU laptop 2 | `2` | Account verification | SuperNode and local training |

Use three Ubuntu/Linux or WSL2 environments with a compatible, tested Flower 1.40.0 training stack. The existing Windows CPU pin file does not install Flower. The [WSL2 environment](../requirements/README.md) is the available tested reference; compatibility and whole-process memory must still be checked on the other two laptops. CPU training uses FP32. Start with two steps per client and a 60-second collection deadline as calibration values, then freeze budgets/deadline after measuring each laptop.

Use a dedicated LAN/hotspot. Choose a stable IPv4 address or DNS name that reaches the host's SuperLink port from the other laptops. The example below uses a chosen IPv4 address. WSL2 NAT does not automatically expose a Linux listener to the Windows LAN: use an appropriate mirrored-network configuration or a deliberately configured port forward. Limit the host firewall rule to the dedicated network and expected clients. This guide does not configure Windows routing or firewall rules.

Prepare these items before disconnecting the internet:

- The same source revision, model/tokenizer revision, actual common-A tensors, and initial B on every laptop. Check adapter identities; do not regenerate A independently.
- Assigned frozen training manifests on each client. The current ServerApp also reads **all public benchmark training manifests** to compute expected counts and hashes. This bootstrap assumes centrally prepared public BANKING77 data; it is not yet a private-data registration protocol.
- Validation manifests on the host. Keep final test access in the frozen-study workflow.
- A common absolute configuration directory that exists separately on each machine, owned by that machine's user. The example uses `/srv/fedlora-live`; the JSON contents contain local paths and can differ between hosts. There is no shared filesystem requirement.
- Synchronized system clocks. The protocol includes absolute deadline timestamps, so meaningful clock skew affects acceptance.

All shell examples below are **Bash inside Linux**, from the repository root with the prepared Python environment activated. Set the same new `RUN_ID` on all machines. Never reuse an existing run ID or certificate directory for a fresh experiment.

```bash
export RUN_ID='three-laptop-rehearsal-001'
export PROJECT="$PWD"
export LIVE_DIR="/srv/fedlora-live/$RUN_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export FLWR_TELEMETRY_ENABLED=0
export FLWR_DISABLE_UPDATE_CHECK=1
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
```

An administrator must first create `/srv/fedlora-live` with ownership/permissions appropriate to the local user. Then `mkdir "$LIVE_DIR"` creates a new run directory and fails if that identity already exists.

## Prepare the local runtime configuration

On every laptop, from its repository checkout:

```bash
mkdir "$LIVE_DIR"
python -m fedlora_edge.federation write-config \
  --run-id "$RUN_ID" --layout three --scenario balanced \
  --strategy equal_clipped --budgets 2,2,2 --quorum 2 \
  --attempts 3 --deadline-seconds 60 --device cpu \
  --validation-manifest assets/banking77/manifests/validation.json
cp "runs/$RUN_ID/runtime.json" "$LIVE_DIR/runtime.json"
python - "$LIVE_DIR/runtime.json" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
c = json.loads(p.read_text())
c.update(runtime='manual_three_laptop_deployment', persistent_worker=False,
         serial_training=False, rediscover_each_attempt=True)
p.write_text(json.dumps(c, indent=2) + '\n')
PY
```

The generated `run_dir`, model, adapter, and manifest paths point to that laptop's checkout. Each client accesses only its assigned manifest bundle, while the host ServerApp accesses all bundles. The `config-path` passed to Flower must be the same absolute string on every host, so each process resolves its own local copy of `runtime.json`.

`persistent_worker=False` is required here: the optimized persistent process is currently restricted to the single-actor Ray runtime. `serial_training=False` avoids the same-host training lock; each laptop owns its own compute. Preserve matching run/model/adapter identities, manifests, budgets, and deadline across the three configurations. Set `evaluation_device` to `cuda` only in the host JSON if host CUDA evaluation has been checked; the default here is CPU.

## Prepare TLS and node identities

On the host, set the actual LAN-reachable IPv4 address before issuing the certificate. The CA and server private keys stay on the host. Only the CA certificate goes to the clients.

```bash
export FEDLORA_HOST='192.168.137.1'
umask 077
mkdir "certificates/$RUN_ID"
openssl req -x509 -newkey rsa:3072 -nodes -sha256 -days 30 \
  -keyout "certificates/$RUN_ID/ca.key" \
  -out "certificates/$RUN_ID/ca.crt" -subj '/CN=FedLoRA rehearsal CA' \
  -addext 'basicConstraints=critical,CA:TRUE' \
  -addext 'keyUsage=critical,keyCertSign,cRLSign'
openssl req -new -newkey rsa:3072 -nodes \
  -keyout "certificates/$RUN_ID/server.key" \
  -out "certificates/$RUN_ID/server.csr" -subj '/CN=FedLoRA rehearsal host'
printf 'subjectAltName=IP:%s,IP:127.0.0.1,DNS:localhost\nbasicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n' \
  "$FEDLORA_HOST" > "certificates/$RUN_ID/server.ext"
openssl x509 -req -sha256 -days 30 \
  -in "certificates/$RUN_ID/server.csr" \
  -CA "certificates/$RUN_ID/ca.crt" -CAkey "certificates/$RUN_ID/ca.key" \
  -CAcreateserial -extfile "certificates/$RUN_ID/server.ext" \
  -out "certificates/$RUN_ID/server.crt"
```

On each laptop, create its own node key using its assigned `CLIENT_ID` (`0`, `1`, or `2`). For clients, create `certificates/$RUN_ID` first. Keep the private key on the originating laptop; transfer only its `.pub` file to the host for registration.

```bash
export CLIENT_ID=0
umask 077
ssh-keygen -t ecdsa -b 384 -N '' -f "certificates/$RUN_ID/node_$CLIENT_ID.key"
```

Distribute the CA certificate through the same trusted preparation channel and compare its SHA-256 fingerprint on every laptop with `sha256sum "certificates/$RUN_ID/ca.crt"`. Set `FEDLORA_HOST` to the same address on the clients. If the address changes, issue a certificate with the new SAN; do not disable certificate verification to bypass a mismatch.

## Start the host and register known nodes

On the host, configure the local Flower CLI profile. Use absolute certificate paths so registration and submission can resolve them regardless of working directory:

```bash
export FLWR_HOME="$PROJECT/runs/$RUN_ID/flower-home"
mkdir "$FLWR_HOME"
cat > "$FLWR_HOME/config.toml" <<EOF
[superlink]
default = "live"

[superlink.live]
address = "$FEDLORA_HOST:19092"
root-certificates = "$PROJECT/certificates/$RUN_ID/ca.crt"
EOF
flower-superlink --host 0.0.0.0 --port 19092 \
  --ssl-ca-certfile "certificates/$RUN_ID/ca.crt" \
  --ssl-certfile "certificates/$RUN_ID/server.crt" \
  --ssl-keyfile "certificates/$RUN_ID/server.key" \
  --enable-supernode-auth --disable-runtime-dependency-installation \
  --database ':flwr-in-memory:'
```

Keep this terminal running. This binds the Flower API to network interfaces, so the dedicated-network/firewall prerequisite matters. This setup assumes trusted participants; node authentication does not establish adversarial robustness or a production administration boundary.

In another host terminal with the same environment variables, register the three public keys after copying the client public keys into the host's certificate directory:

```bash
for id in 0 1 2; do
  flwr supernode register "certificates/$RUN_ID/node_$id.key.pub" live \
    --name "client-$id" --format json > "runs/$RUN_ID/register_$id.json"
done
python - "$LIVE_DIR/runtime.json" "runs/$RUN_ID" <<'PY'
import json
import sys
from pathlib import Path
p, records = Path(sys.argv[1]), Path(sys.argv[2])
c = json.loads(p.read_text())
c['node_map'] = {str(i): int(json.loads((records / f'register_{i}.json').read_text())['node-id']) for i in range(3)}
p.write_text(json.dumps(c, indent=2) + '\n')
PY
```

Only the host configuration needs `node_map`. Check that all three registrations succeeded before proceeding. This recipe uses the tested in-memory Flower state backend. Restarting SuperLink discards its registrations and is not covered by the node-rejoin check; re-registering changes must be reflected in the host map before a new run.

## Start each client and submit the run

On each laptop, with its own `CLIENT_ID`, local private key, CA certificate, and runtime JSON ready:

```bash
flower-supernode --superlink "$FEDLORA_HOST:19092" \
  --root-certificates "certificates/$RUN_ID/ca.crt" \
  --auth-supernode-private-key "certificates/$RUN_ID/node_$CLIENT_ID.key" \
  --host 127.0.0.1 --port 19093 \
  --node-config "partition-id=$CLIENT_ID device=\"cpu\""
```

The node's local Runtime API stays on loopback. Different machines can use the same local port. On the host only, change `device="cpu"` to `device="cuda"` if host GPU training has been calibrated. Do not use `--insecure` or opt into runtime dependency installation.

In the host submission terminal, create the run override and launch:

```bash
printf 'config-path = "%s/runtime.json"\n' "$LIVE_DIR" > "runs/$RUN_ID/run-overrides.toml"
flwr run "$PROJECT" live --run-config "runs/$RUN_ID/run-overrides.toml" --stream
```

The app uses the same ClientApp training function as simulation. Inspect host `events.jsonl`, `round_results.json`, and `checkpoint.json`; inspect local client logs for work and synchronization. Local client event files are not automatically merged into the host dashboard. Preserve them with machine/run identity when collecting the rehearsal evidence.

## Acceptance checks

First complete an all-client calibration. Confirm the actual work, quorum-two decisions, per-client coefficients, raw/serialized bytes, timing, and memory. Then use a **new declared run** for an interruption: stop one SuperNode during observed training, let the server reach its deadline, and restart the same node with the same private key. Check whether the survivors committed, and verify the returning client's next synchronization names the current global B/version. Rejecting the old partial work is part of the check. If fewer than two valid replies arrive, verify that the attempt is skipped and the committed checkpoint remains unchanged.

Repeat a fresh start with internet disconnected, and record that model/data/dependency downloads were unnecessary. Do not equate one warm process, an isolated worker kill, or localhost TLS with this physical-laptop acceptance gate. The [implementation status](../research/implementation-status.md) remains the authority on what has actually been demonstrated.
