"""Exercise actual Flower TLS/authentication/routing without claiming LM training.

This is an infrastructure gate. The separate federation CLI performs real model
training and process-interruption rehearsals using the production ClientApp.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedlora_edge.checkpoint import atomic_json
from fedlora_edge.federation import deploy_local


APP_SOURCE = '''import json
from pathlib import Path
import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp
from fedlora_edge.flower_app import _collect, serialized_message_bytes

client_app = ClientApp()
server_app = ServerApp()

@client_app.query("identity")
def identity(message, context):
    return Message(RecordDict({"identity": ConfigRecord({
        "client_id": str(context.node_config["partition-id"])
    }), "echo": message.content["arrays"]}), reply_to=message)

@server_app.main()
def main(grid, context):
    config = json.loads(Path(context.run_config["config-path"]).read_text())
    payload = np.arange(184320, dtype=np.float32)
    messages = [Message(RecordDict({"arrays": ArrayRecord({"protocol_probe": Array(payload)})}),
                        dst_node_id=int(node), message_type="query.identity", ttl=90)
                for node in config["node_map"].values()]
    replies = []
    for reply, received in _collect(grid, messages, timeout=90):
        if reply.has_error():
            raise RuntimeError(str(reply.error))
        if not np.array_equal(reply.content["echo"]["protocol_probe"].numpy(), payload):
            raise RuntimeError("Named FP32 array round trip changed the payload")
        replies.append({"node_id": reply.metadata.src_node_id,
                        "client_id": reply.content["identity"]["client_id"],
                        "serialized_reply_bytes": serialized_message_bytes(reply)})
    if {r["client_id"] for r in replies} != set(config["node_map"]):
        raise RuntimeError("Not all registered clients replied")
    result = {"run_id": config["run_id"], "kind": "transport_protocol_check",
              "language_model_experiment": False, "tls": True,
              "registered_node_authentication": True, "replies": replies,
              "raw_tensor_bytes": payload.nbytes, "named_array_echo_verified": True}
    Path(config["run_dir"], "final_results.json").write_text(json.dumps(result, indent=2))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="flower-transport-api-smoke")
    parser.add_argument("--port", type=int, default=19492)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "runs" / args.run_id
    app = output / "probe_app"
    app.mkdir(parents=True, exist_ok=True)
    (app / "probe.py").write_text(APP_SOURCE)
    (app / "pyproject.toml").write_text('''[project]
name = "fedlora-transport-probe"
version = "0.1.0"
description = "TLS routing check, not language-model training"
license = "Apache-2.0"
dependencies = []
[tool.flwr.app]
publisher = "fedlora-edge"
[tool.flwr.app.components]
serverapp = "probe:server_app"
clientapp = "probe:client_app"
[tool.flwr.app.config]
config-path = ""
''')
    config = {"run_id": args.run_id, "run_dir": str(output), "client_manifests":
              {str(client): [] for client in range(5)}, "attempts": 1, "deadline_seconds": 60,
              "runtime": "transport_protocol_check", "language_model_experiment": False}
    atomic_json(output / "runtime.json", config)
    deploy_local(config, project_dir=str(app), port=args.port,
                 certificate_dir=str(root / "certificates" / args.run_id))
    print((output / "final_results.json").read_text())


if __name__ == "__main__":
    main()
