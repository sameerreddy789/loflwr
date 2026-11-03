# Rehearse the encoder system on separate laptops

This guide prepares the **frozen MPNet plus federated classifier head** variant.
The original SmolLM2 LoRA deployment has a [separate guide](deployment.md).
Head-only training does not train LoRA A or B. Its 15,380 FP32 parameters occupy
61,520 raw bytes per direction, before serialization and transport overhead.

**Physical three-laptop status: pending.** Only one laptop is available. The
three-role preparation and configuration checks described below ran on that
laptop. They do not establish LAN reachability, performance on two weaker
machines, or physical network-loss recovery. Use the [encoder demo](encoder-demo.md)
for the prepared query interface and its independently confirmed 94% result.

## 1. Prepare the machines and declare the run

| Machine | Participant | Own training data | Additional role |
| --- | --- | --- | --- |
| Host | `0` | Lifecycle and card payments: 1,128 queries | SuperLink, ServerApp, 200 validation queries, dashboard |
| Laptop 1 | `1` | Transfers and cash withdrawals: 1,224 queries | SuperNode |
| Laptop 2 | `2` | Verification: 408 queries | SuperNode |

These counts belong to the expanded public BANKING77 training dataset. They are
distinct from the original 800-query balanced and 400-query imbalance experiments.
The host runs two application roles, each with its own dataset directory and
configuration. A shared OS account on the host is not an enforced security boundary.

Use Python 3.11 and the [tested Ubuntu/WSL environment](../requirements/README.md).
The Windows CPU environment does not include Flower. The Ubuntu environment's
CUDA PyTorch wheel has also executed these CPU-only paths successfully; that
does not certify a different OS or processor. Install dependencies while online,
run the relevant tests, and record the environment on each new machine before
disconnecting internet. Do not install packages automatically during a Flower run.

Choose a dedicated LAN/hotspot and a stable reachable host address. WSL2 NAT may
need a deliberately configured port forward or mirrored networking. Verify the
route and restrict the Windows firewall to the intended network; these tools do
not alter routing or firewall rules. Keep clocks synchronized for reply deadlines.

The remaining commands are **Bash inside Ubuntu/WSL**, from the repository root
with the prepared virtual environment activated. Use the same new `RUN_ID` on
all machines; each `PROJECT` is that machine's own checkout.

```bash
export RUN_ID='encoder-laptops-001'
export PROJECT="$PWD"
export LIVE="$PROJECT/runs/$RUN_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export FLWR_TELEMETRY_ENABLED=0
export FLWR_DISABLE_UPDATE_CHECK=1
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
mkdir -p "$PROJECT/runs"
mkdir "$LIVE"
```

Prepare the pinned `assets/reference-mpnet` directory on every machine before
going offline, including its hidden `.cache/huggingface/download` revision
metadata. The preparation loader verifies local model/tokenizer files against
the common plan and uses `local_files_only=True`. The model revision is
`e8c3b32edf5434bc2275fc9bab85f82640a19130`.

## 2. Prepare features locally and exchange metadata

The public-benchmark provisioner first creates a common plan using metadata:

```bash
python tools/prepare_encoder_local.py plan \
  --source-dataset assets/encoder-bank20-v1 --layout three \
  --output "$LIVE/plan.json"
```

Give each machine this plan and only its assigned original-training record
bundles. The source bundles are `assets/encoder-bank20-v1/clients/client_0.json`
through `client_4.json`; group original clients `[0,1]`, `[2,3]`, `[4]` according
to the table. The validation role gets only `validation.json`. This provisioning
step uses a public benchmark; it does not demonstrate ownership of private data.
No test or confirmation file is needed for preparation or training.

On each client, put its approved bundles in its own `assets/local-records/`
directory. Set `CLIENT_ID` to `0`, `1`, or `2`. For example, laptop 1 prepares
its two bundles with:

```bash
export CLIENT_ID=1
python tools/prepare_encoder_local.py prepare \
  --plan "$LIVE/plan.json" --participant "$CLIENT_ID" \
  --records assets/local-records/client_2.json assets/local-records/client_3.json \
  --asset-dir assets/reference-mpnet --output "$LIVE/client-data"
```

Client 0 passes `client_0.json client_1.json`; client 2 passes only
`client_4.json`. On the host, prepare validation separately:

```bash
python tools/prepare_encoder_local.py prepare \
  --plan "$LIVE/plan.json" --participant server \
  --records assets/local-records/validation.json \
  --asset-dir assets/reference-mpnet --output "$LIVE/server-data"
```

The prepared directory contains the participant's feature archive, its records,
`registration.json`, and `preparation.json`. The encoder uses FP32 CPU, batch 8,
one CPU thread, a 384-token maximum, masked mean pooling, and L2 normalization.

Transfer **only `registration.json`** from each client to the coordinator, renamed
`registrations/0.json`, `1.json`, and `2.json`. Copy the server's registration to
`registrations/server.json`. Registration contains hashes, sample/label counts,
and relative filenames; it contains no query text or feature vectors. Assemble:

```bash
python tools/prepare_encoder_local.py assemble \
  --plan "$LIVE/plan.json" \
  --registrations "$LIVE/registrations/0.json" "$LIVE/registrations/1.json" \
    "$LIVE/registrations/2.json" "$LIVE/registrations/server.json" \
  --output "$LIVE/manifest.json"
```

Distribute this identical `manifest.json` to each prepared dataset directory.
Each directory retains only that role's feature/record files. The manifest may
describe other participants, but the training loader opens only its own archive;
the server opens only validation. Hashes and metadata can themselves reveal
information, so metadata-only registration is not a formal privacy guarantee.

The same-host parity checks `encoder-local-five-v1` and
`encoder-local-three-v1` ran each participant in a fresh Linux mount, PID, and
network namespace. Other source files were hidden and external connectivity
failed. Both layouts matched reference features within `1.05e-7` (declared
tolerance `1e-5`) with **zero changed head predictions across 2,960 train/validation
rows per layout**. This checks preprocessing parity, not new held-out accuracy.
Three-role preparation took 411.85 seconds overall; individual encoding processes
peaked at approximately 1.19 GB RSS on the available laptop. Re-measure on the
actual target machines.

## 3. Build the encoder app and configure each role

On the host, build a new code-only application directory:

```bash
python tools/build_encoder_app.py --output "$LIVE/app"
```

This changes the staged Flower entry points to
`fedlora_edge.encoder_deployment_app:server_app` and `:client_app`. Submit this
directory. The repository root still points to the original LoRA application.
The package contains Python source, project metadata, README, and license;
model weights, feature archives, runs, and credentials are excluded.

Configure each client after copying `manifest.json` into its own `client-data`:

```bash
python tools/configure_encoder_laptop.py \
  --dataset "$LIVE/client-data" --output "$LIVE/client/config.json" \
  --role client --client-id "$CLIENT_ID" --run-id "$RUN_ID" --attempts 5
```

The tool checks only the selected role's feature archive. Its output contains
local paths and a common model/data/optimizer identity. It refuses to overwrite
a configuration and rejects server-only checkpoint/node-map inputs for a client.
Leave client keys out of these JSON files.

The initial rehearsal settings are CPU head training, SGD with learning rate
0.1, batch 1,024, class balancing, one step for clients 0/2 and two for client 1,
equal survivor weights with median norm clipping, quorum 2, and a 60-second
reply collection deadline. This is a warm-start recovery configuration, not the
500-attempt accuracy study. Calibrate these budgets on each actual laptop before
freezing a new rehearsal; initialization/preparation time is a separate cost.

## 4. Start TLS, register nodes, and configure the server

Follow the **TLS and node identity** and **host/registration** sections of the
[deployment guide](deployment.md) with the same `RUN_ID`. Issue the host
certificate for its actual LAN address, compare the CA fingerprint, and generate
each client's node private key on that client. Transfer only public node keys
to the host; the CA private key and server private key stay on the host.

Start SuperLink in a dedicated host terminal. Before its command, set the
server's local configuration path, even though registration precedes creating
that file:

```bash
export FEDLORA_ENCODER_CONFIG="$LIVE/server/config.json"
export FLWR_HOME="$LIVE/flower-home"
flower-superlink --host 0.0.0.0 --port 19092 \
  --ssl-ca-certfile "certificates/$RUN_ID/ca.crt" \
  --ssl-certfile "certificates/$RUN_ID/server.crt" \
  --ssl-keyfile "certificates/$RUN_ID/server.key" \
  --enable-supernode-auth --disable-runtime-dependency-installation \
  --database ':flwr-in-memory:'
```

Use another host terminal with the prepared `live` Flower CLI profile to register
the public keys, preserving the JSON replies:

```bash
for id in 0 1 2; do
  flwr supernode register "certificates/$RUN_ID/node_$id.key.pub" live \
    --name "client-$id" --format json > "$LIVE/register_$id.json"
done
python - "$LIVE" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
mapping = {str(i): int(json.loads((root / f'register_{i}.json').read_text())['node-id'])
           for i in range(3)}
(root / 'node-map.json').write_text(json.dumps(mapping, indent=2) + '\n')
PY
python tools/configure_encoder_laptop.py \
  --dataset "$LIVE/server-data" --output "$LIVE/server/config.json" \
  --role server --node-map "$LIVE/node-map.json" --run-id "$RUN_ID" --attempts 5 \
  --initial-checkpoint runs/encoder-flower-balanced-v2/selected
```

Transfer the existing selected checkpoint to the host beforehand; it is not
included in a clean Git clone. The tool verifies its pinned encoder/head
contract, then copies the exact parameters under the new local-data/run identity.
The initial head hash must remain
`d85f8e14a62ebaa092c7a1e173e74e5c35a3c6dd59258c08252626f7d6cd7312`.
Rebinding the contract is not training or another independent confirmation.
Omitting `--initial-checkpoint` starts a newly initialized head and cannot use
the selected demo's accuracy claim.

The configuration and migration smoke `encoder-portable-smoke-v1` passed on all
four prepared roles, preserving that head hash. It used fixture node IDs, so it
is not transport evidence. Flower 1.40.0 also successfully built the staged
application into a FAB. Four focused automated tests cover role-specific reads,
invalid/duplicate node maps, incompatible migration, and package contents.

## 5. Start the clients and submit

On each laptop use a **client terminal**, with its own `CLIENT_ID`, local keys,
and configuration. On the host this must be separate from the SuperLink terminal
so the server and client inherit different `FEDLORA_ENCODER_CONFIG` values.

```bash
export FEDLORA_ENCODER_CONFIG="$LIVE/client/config.json"
python -m fedlora_edge.supernode_launcher --activation-wait-seconds 75 -- \
  flower-supernode --superlink "$FEDLORA_HOST:19092" \
  --root-certificates "certificates/$RUN_ID/ca.crt" \
  --auth-supernode-private-key "certificates/$RUN_ID/node_$CLIENT_ID.key" \
  --host 127.0.0.1 --port 19093 \
  --node-config "partition-id=$CLIENT_ID device=\"cpu\""
```

On the host submission terminal, with the `live` CLI profile:

```bash
flwr run "$LIVE/app" live --stream
```

Each spawned role resolves its own environment variable. No shared absolute
configuration path is required across laptops. Keep source revisions, manifest,
run ID, head settings, budgets, quorum, and deadlines identical. Keep runtime
dependency installation disabled. Copying the whole public source dataset to
every machine would defeat the intended locality demonstration.

## 6. Record recovery, offline startup, and the remaining gates

First run an all-client calibration. Save per-machine environment/hardware,
preparation time, whole-process memory, completed work, reply timing, accepted
coefficients, and raw/serialized payload sizes. Do not equate serialized message
bytes with measured network traffic or summed process RSS with unique physical RAM.

Use a separately declared run for dropout. Stop a SuperNode after observing
local training, wait for the bounded collection decision, and restart using its
same private key. The server must accept two survivors with coefficient 0.5,
reject old partial work, and synchronize current state before the returning
client trains. The application includes a separate bounded 90-second identity
readiness window; this is additional to the 60-second training reply window.
Flower 1.40 may retain an abruptly stopped node's ONLINE heartbeat lease for up
to 60 seconds after its last heartbeat. A restart during that lease fails with
activation error 2004; waiting only for the training attempt to end is insufficient.
The launcher retries only that activation failure using the same key/registration,
for a 75-second startup window. Stop the old instance before restarting: a live
duplicate can produce the same error. Preserve logs and inspect node state if
the window is exhausted; do not create a second identity or report failed rejoining
as success.
If fewer than two replies arrive, the checkpoint must remain unchanged.

Inspect `server/run/round_results.json` and the global checkpoint. Each client
has its own `client/run/events.jsonl`; collect copies explicitly with machine
and run identity. Do not claim local client logs were automatically received by
the server. SuperLink's in-memory state loses registrations on restart: create
a fresh run, register again, and rebuild the complete node map.

Finally, stop all application processes, disconnect external internet while
preserving the LAN, and perform a fresh complete startup. A warm cache or a
loopback-only namespace does not establish this physical offline gate. Preserve
the physical evidence separately from same-host rehearsal results.

Data locality and TLS reduce unnecessary exposure; they do not provide
differential privacy, secure aggregation, or robustness against malicious nodes.
Keep this prototype on the public benchmark. The result of another recovery
run is not a new accuracy score for its updated checkpoint.
