"""Write one host's encoder configuration after local metadata registration.

This opens only the selected role's feature archive. Generate and exchange node
public keys separately. The config includes no keys, queries or feature arrays.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fedlora_edge.data import write_json
from fedlora_edge.encoder_data import load_encoder_manifest,load_feature_split,sha256
from fedlora_edge.encoder_head import HeadConfig
from fedlora_edge.state import TensorData


def configure(dataset,output,role,client_id=None,node_map=None,initial_checkpoint=None,attempts=5,run_id=None):
    from tools.run_encoder_study import make_identity
    from fedlora_edge.encoder_flower import validate_config
    dataset,output=Path(dataset).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError("Choose a new config output")
    manifest=load_encoder_manifest(dataset)
    clients=sorted(manifest["clients"])
    if role not in ("server","client") or (role=="client" and client_id not in clients):
        raise ValueError("Invalid local role/client ID")
    if role=="client" and (node_map or initial_checkpoint):
        raise ValueError("Client configuration must not open server-only inputs")
    load_feature_split(dataset,None if role=="server" else client_id)
    head=asdict(HeadConfig(learning_rate=.1,batch_size=1024,class_balance=True))
    schema={"head.weight":TensorData((20,768),(0.0,)*15360),"head.bias":TensorData((20,),(0.0,)*20)}
    identity=make_identity(manifest,schema,head)
    config={"run_id":run_id or output.parent.name,"role":role,"dataset_dir":str(dataset),"run_dir":str(output.parent/"run"),
        "identity":identity,"manifest_hash":manifest["manifest_hash"],"head_config":head,"attempts":attempts,
        "budgets":{c:2 if c=="1" else 1 for c in clients},"strategy":"equal_clipped","seed":42,
        "quorum":2 if len(clients)==3 else 3,"deadline_seconds":60,"evaluate_every":1,
        "startup_timeout_seconds":120,"persistent_worker":False,"wait_for_rejoining_nodes":True,"rejoin_wait_seconds":90,
        "runtime_kind":"flower_encoder_laptop_deployment","deployment_scope":"Operator-managed devices; physical topology must be independently recorded",
        "require_filesystem_isolation":False}
    if role=="client": config["local_client_id"]=client_id
    else:
        mapping=json.loads(Path(node_map).read_text()) if node_map else {}
        if (set(mapping)!=set(clients) or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in mapping.values())
                or len(set(mapping.values()))!=len(clients)):
            raise ValueError("Server requires the complete authenticated client-to-node map")
        config["node_map"]=mapping
    # Validate all non-filesystem protocol choices before creating artifacts.
    validate_config(config)
    if role=="server" and initial_checkpoint:
        from fedlora_edge.encoder_inference import load_head_checkpoint
        from fedlora_edge.checkpoint import CheckpointStore
        saved=load_head_checkpoint(initial_checkpoint)
        if any(saved["identity"].get(key) != identity[key] for key in
               ("encoder_identity_hash", "model_revision", "feature_dim", "labels", "head_schema")):
            raise ValueError("Initial head and local features use different encoder contracts")
        target=output.parent/"initial"
        if target.exists(): raise FileExistsError("Initial checkpoint directory already exists")
        CheckpointStore(target).save(run_id=config["run_id"],attempt_id=0,version=0,state=saved["state"],
            identity=identity,config_hash="parameter-preserving-migration",metadata={"original_run_id":saved["run_id"],"original_head_hash":saved["b_hash"]})
        config["initial_checkpoint_dir"]=str(target)
        config["initial_checkpoint_sha256"]=sha256(target/"checkpoint.json")
    write_json(output,config)
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--role",choices=["server","client"],required=True)
    parser.add_argument("--client-id")
    parser.add_argument("--node-map",type=Path)
    parser.add_argument("--initial-checkpoint",type=Path)
    parser.add_argument("--attempts",type=int,default=5)
    parser.add_argument("--run-id")
    args=parser.parse_args()
    result=configure(args.dataset,args.output,args.role,args.client_id,args.node_map,args.initial_checkpoint,args.attempts,args.run_id)
    print(json.dumps({"role":result["role"],"run_id":result["run_id"],"config":str(args.output.resolve())}))


if __name__=="__main__":main()
