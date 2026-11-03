"""Five-node, one-worker Ray/Flower routing check; not a model experiment."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["FLWR_TELEMETRY_ENABLED"] = "0"
os.environ["FLWR_DISABLE_UPDATE_CHECK"] = "1"

from flwr.app import Message, RecordDict
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation

from fedlora_edge.checkpoint import atomic_json
from fedlora_edge.flower_app import _collect, make_client_app
from fedlora_edge.federation import initialize_ray


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "runs" / "flower-ray-routing-smoke"
    server = ServerApp()

    @server.main()
    def probe(grid, context):
        started = time.monotonic()
        nodes = list(grid.get_node_ids())
        while len(nodes) < 5 and time.monotonic() - started < 60:
            time.sleep(0.1)
            nodes = list(grid.get_node_ids())
        messages = [Message(RecordDict(), dst_node_id=node, message_type="query.identity", ttl=90)
                    for node in nodes]
        clients = []
        for reply, _ in _collect(grid, messages, timeout=90):
            if reply.has_error():
                raise RuntimeError(str(reply.error))
            clients.append(reply.content["metadata"]["client_id"])
        if sorted(clients) != ["0", "1", "2", "3", "4"]:
            raise RuntimeError(f"Missing clients: {clients}")
        atomic_json(output / "proof.json", {
            "kind": "flower_ray_routing_check", "language_model_experiment": False,
            "logical_clients": sorted(clients), "workers": 1,
            "elapsed_seconds": time.monotonic() - started,
        })

    backend = {"init_args": {"num_cpus": 1, "num_gpus": 0, "include_dashboard": False},
               "client_resources": {"num_cpus": 1, "num_gpus": 0}}
    initialize_ray(backend)
    run_simulation(server_app=server, client_app=make_client_app(), num_supernodes=5,
                   backend_config=backend)
    print(json.loads((output / "proof.json").read_text()))


if __name__ == "__main__":
    main()
