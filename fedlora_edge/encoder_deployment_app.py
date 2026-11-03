"""Flower entry points for independently packaged encoder-head participants.

Every host sets FEDLORA_ENCODER_CONFIG to its own local config. This avoids sending
server filesystem paths to clients in the shared Flower run configuration.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp


def local_config(role):
    path = os.environ.get("FEDLORA_ENCODER_CONFIG")
    if not path:
        raise ValueError("Set FEDLORA_ENCODER_CONFIG to this host's local configuration")
    config = json.loads(Path(path).read_text())
    if config.get("role") != role:
        raise ValueError("Local deployment package has the wrong role")
    data = Path(config["dataset_dir"])
    from .encoder_data import load_encoder_manifest
    manifest = load_encoder_manifest(data)
    allowed = {"manifest.json", "registration.json", "preparation.json"}
    client = None if role == "server" else config["local_client_id"]
    entry = manifest["validation"] if client is None else manifest["clients"][client]
    allowed |= {entry["file"], entry["records_file"]}
    present = {p.relative_to(data).as_posix() for p in data.rglob("*") if p.is_file()}
    if not present <= allowed:
        raise ValueError("Dataset package contains another participant's files")
    if config.get("require_filesystem_isolation"):
        forbidden = config["forbidden_host_paths"]
        if os.environ.get("FEDLORA_ISOLATED") != "1" or any(Path(p).exists() for p in forbidden):
            raise RuntimeError("Original host dataset is visible; isolation gate failed")
        from .checkpoint import atomic_json
        atomic_json(Path(config["run_dir"]).parent / "isolation.json", {
            "role": role, "client_id": client, "dataset_files": sorted(present),
            "forbidden_paths_absent": forbidden, "mechanism": "Linux mount namespace via bubblewrap"})
    return config


client_app = ClientApp()
server_app = ServerApp()


@client_app.query("identity")
def identity(message, context):
    from .encoder_flower import make_client_app
    return make_client_app(local_config("client"))(message, context)


@client_app.train()
def train(message, context):
    from .encoder_flower import make_client_app
    return make_client_app(local_config("client"))(message, context)


@server_app.main()
def main(grid, context):
    from .encoder_flower import make_server_app
    make_server_app(local_config("server"))(grid, context)
