"""Authenticated Flower head deployment with per-role files and OS mount isolation.

Run in Ubuntu WSL. This prepares portable role data/configs and tests all roles
on one host. It does not configure Windows LAN routing or claim physical laptops.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
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
from fedlora_edge.data import write_json
from fedlora_edge.encoder_data import load_encoder_manifest, sha256


def read(path):
    return json.loads(Path(path).read_text())


def isolation_command(project, app, role, certificates, name):
    command = ["bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--ro-bind", "/", "/",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", str(project),
        "--ro-bind", str(app), "/tmp/app", "--bind", str(role), "/tmp/role",
        "--ro-bind", str(certificates / "ca.crt"), "/tmp/ca.crt"]
    if name == "server":
        command += ["--ro-bind", str(certificates / "server.crt"), "/tmp/server.crt",
                    "--ro-bind", str(certificates / "server.key"), "/tmp/server.key"]
    else:
        command += ["--ro-bind", str(certificates / f"node_{name}.key"), "/tmp/node.key"]
    for key, value in {"HOME":"/tmp/role/home", "FLWR_HOME":"/tmp/role/flower",
        "PATH":str(Path(sys.executable).parent)+os.pathsep+os.environ.get("PATH",os.defpath),
        "PYTHONPATH":"/tmp/app", "FEDLORA_ENCODER_CONFIG":"/tmp/role/config.json", "FEDLORA_ISOLATED":"1",
        "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "FLWR_TELEMETRY_ENABLED":"0",
        "FLWR_DISABLE_UPDATE_CHECK":"1", "FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION":"1",
        "TOKENIZERS_PARALLELISM":"false", "CUDA_VISIBLE_DEVICES":""}.items():
        command += ["--setenv", key, value]
    return command + ["--chdir", "/tmp/app", "--"]


def prepare(local_data, output, port, recovery=True):
    from fedlora_edge.encoder_head import HeadConfig
    from fedlora_edge.encoder_inference import load_head_checkpoint
    from fedlora_edge.checkpoint import CheckpointStore
    from tools.run_encoder_study import make_identity
    if output.exists():
        raise FileExistsError("Use a new deployment directory")
    evidence=read(local_data/"result.json")
    manifest=load_encoder_manifest(local_data)
    if evidence["status"]!="completed" or evidence["manifest_hash"]!=manifest["manifest_hash"]:
        raise ValueError("Local preparation and parity must have completed")
    clients=sorted(manifest["clients"])
    if len(clients) not in (3,5):
        raise ValueError("Expected three or five local clients")
    output.mkdir(parents=True)
    app=output/"app"
    (app/"fedlora_edge").mkdir(parents=True)
    for path in (ROOT/"fedlora_edge").glob("*.py"):
        shutil.copy2(path,app/"fedlora_edge"/path.name)
    for name in ("README.md","LICENSE","pyproject.toml"):
        shutil.copy2(ROOT/name,app/name)
    project=(app/"pyproject.toml").read_text().replace("fedlora_edge.flower_app:server_app", "fedlora_edge.encoder_deployment_app:server_app")
    project=project.replace("fedlora_edge.flower_app:client_app", "fedlora_edge.encoder_deployment_app:client_app")
    (app/"pyproject.toml").write_text(project)
    head=asdict(HeadConfig(learning_rate=0.1,batch_size=1024,class_balance=True))
    original=load_head_checkpoint(ROOT/"runs/encoder-flower-balanced-v2/selected")
    identity=make_identity(manifest,original["state"],head)
    quorum=2 if len(clients)==3 else 3
    schedule=([{ "attempt_id":2,"client_id":"1","after_steps":1},
        *[{"attempt_id":4,"client_id":c,"after_steps":1} for c in clients[:len(clients)-quorum+1]]] if recovery else [])
    common={"run_id":output.name,"dataset_dir":"/tmp/role/data","run_dir":"/tmp/role/run",
        "identity":identity,"manifest_hash":manifest["manifest_hash"],"head_config":head,
        "attempts":5 if recovery else 1,"evaluate_every":1,"seed":42,"quorum":quorum,
        "budgets":{c:2 if c=="1" else 1 for c in clients},"strategy":"equal_clipped",
        "deadline_seconds":60,"startup_timeout_seconds":120,"between_attempt_seconds":5,
        "wait_for_rejoining_nodes":True,"rejoin_wait_seconds":90,
        "persistent_worker":False,"external_node_interruptions":schedule,
        "runtime_kind":"flower_deployment_tls_isolated_roles","deployment_scope":"All roles on one physical laptop in separate mount/PID namespaces",
        "require_filesystem_isolation":True,"forbidden_host_paths":[str(ROOT/"assets/encoder-bank20-v1"),str(local_data),str(output/"roles")]}
    for name in ["server",*clients]:
        role=output/"roles"/name
        for folder in ("data","home","flower"):
            (role/folder).mkdir(parents=True)
        source=local_data/"participants"/name/"output/data"
        entry=manifest["validation"] if name=="server" else manifest["clients"][name]
        destination=role/"data"/entry["file"]
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/entry["file"],destination)
        write_json(role/"data/manifest.json",manifest)
        config={**common,"role":"server" if name=="server" else "client"}
        if name=="server":
            # Exact existing parameters, re-bound to the new local-data contract.
            CheckpointStore(role/"initial").save(run_id=output.name,attempt_id=0,version=0,
                state=original["state"],identity=identity,config_hash="migration",
                metadata={"original_run_id":original["run_id"],"original_head_hash":original["b_hash"],
                          "interpretation":"Parameter-preserving migration, not new training or confirmation"})
            config["initial_checkpoint_dir"]="/tmp/role/initial"
            config["initial_checkpoint_sha256"]=sha256(role/"initial/checkpoint.json")
        else:
            config["local_client_id"]=name
        write_json(role/"config.json",config)
        (role/"flower/config.toml").write_text('[superlink]\ndefault="rehearsal"\n[superlink.rehearsal]\n'
            f'address="127.0.0.1:{port}"\nroot-certificates="/tmp/ca.crt"\n')
    report={"status":"prepared","run_id":output.name,"clients":clients,"quorum":quorum,
        "port":port,"attempts":common["attempts"],"deadline_seconds":60,"external_node_interruptions":schedule,
        "source_local_manifest_hash":manifest["manifest_hash"],"initial_head_hash":original["b_hash"],
        "source_local_data":str(local_data),"sources":{p.name:sha256(p) for p in (app/"fedlora_edge").glob("*.py")},
        "launcher_sha256":sha256(Path(__file__)),
        "limits":"Same-host isolated roles; not physical device or production privacy evidence"}
    write_json(output/"package.json",report)
    return report


def run(output, require_offline=False):
    from fedlora_edge.federation import generate_certificates,_binary,_stop_tree
    from fedlora_edge.checkpoint import read_progress,EventLog
    import psutil
    if sys.platform!="linux" or not shutil.which("bwrap"):
        raise RuntimeError("Use the tested Ubuntu WSL environment with bubblewrap")
    package=read(output/"package.json")
    if package["launcher_sha256"]!=sha256(Path(__file__)) or any(
            sha256(output/"app/fedlora_edge"/name)!=digest for name,digest in package["sources"].items()):
        raise ValueError("Deployment launcher or packaged source changed after preparation")
    if (output/"deployment_started.json").exists():
        raise FileExistsError("Do not overwrite a started deployment")
    network={"interfaces":[name for _,name in socket.if_nameindex()]}
    if require_offline:
        if network["interfaces"]!=["lo"] or Path("/proc/net/route").read_text().splitlines()[1:]:
            raise RuntimeError("Offline test requires a fresh loopback-only network namespace")
        try:
            with socket.create_connection(("1.1.1.1",443),timeout=0.5):
                raise RuntimeError("Unexpected external connectivity")
        except OSError as error:
            network["external_connection_error"]=str(error)
    certs=ROOT/"certificates"/package["run_id"]
    generate_certificates(certs,package["clients"])
    app=output/"app"
    logs=output/"logs"
    logs.mkdir()
    write_json(output/"deployment_started.json",{"network":network,"require_offline":require_offline,"started":time.time()})
    processes,streams,nodes,commands=[],[],{},{}
    events=EventLog(output,package["run_id"])
    port=package["port"]
    started=time.perf_counter()
    peak={name:0 for name in ["server",*package["clients"]]}
    owners={}

    def start(name,command,logname):
        full=isolation_command(ROOT,app,output/"roles"/name,certs,name)+command
        stream=(logs/logname).open("wb")
        streams.append(stream)
        process=subprocess.Popen(full,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
        processes.append(process)
        owners[process.pid]=name
        return process
    try:
        link=start("server",[_binary("flower-superlink"),"--host","127.0.0.1","--port",str(port),
            "--ssl-ca-certfile","/tmp/ca.crt","--ssl-certfile","/tmp/server.crt","--ssl-keyfile","/tmp/server.key",
            "--enable-supernode-auth","--disable-runtime-dependency-installation","--database",":flwr-in-memory:"],"superlink.log")
        deadline=time.monotonic()+120
        while True:
            if link.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError("SuperLink failed to become ready; inspect logs")
            try:
                with socket.create_connection(("127.0.0.1",port),timeout=.2): break
            except OSError: time.sleep(.1)
        # Registration uses only public node keys and a separate public CLI profile.
        registration_home=output/"registration-home"
        registration_home.mkdir()
        (registration_home/"config.toml").write_text('[superlink]\ndefault="rehearsal"\n[superlink.rehearsal]\n'
            f'address="127.0.0.1:{port}"\nroot-certificates={json.dumps(str(certs/"ca.crt"))}\n')
        env={**os.environ,"FLWR_HOME":str(registration_home),"FLWR_TELEMETRY_ENABLED":"0"}
        mapping={}
        for offset,c in enumerate(package["clients"]):
            registered=subprocess.run([_binary("flwr"),"supernode","register",str(certs/f"node_{c}.pub"),
                "rehearsal","--name",f"client-{c}","--format","json"],env=env,capture_output=True,text=True,check=True,timeout=30)
            mapping[c]=int(json.loads(registered.stdout)["node-id"])
            commands[c]=[sys.executable,"-m","fedlora_edge.supernode_launcher","--activation-wait-seconds","75","--",
                _binary("flower-supernode"),"--superlink",f"127.0.0.1:{port}","--root-certificates","/tmp/ca.crt",
                "--auth-supernode-private-key","/tmp/node.key","--host","127.0.0.1","--port",str(port+1+offset),
                "--node-config",f"partition-id={c}"]
            nodes[c]=start(c,commands[c],f"node-{c}.log")
        server_path=output/"roles/server/config.json"
        config=read(server_path)
        config["node_map"]=mapping
        config["transport_security"]={"tls":True,"authenticated_registered_nodes":True,"ca_sha256":sha256(certs/"ca.crt")}
        write_json(server_path,config)
        process=start("server",[_binary("flwr"),"run","/tmp/app","rehearsal","--stream"],"run.log")
        schedule=[dict(item,triggered=False,restarted=False) for item in package["external_node_interruptions"]]
        deadline=time.monotonic()+900
        while process.poll() is None:
            if time.monotonic()>deadline: raise TimeoutError("Bounded deployment exceeded 900 seconds")
            results_path=output/"roles/server/run/round_results.json"
            attempts=[r["attempt_id"] for r in read(results_path)] if results_path.exists() else []
            for item in schedule:
                c,attempt=item["client_id"],item["attempt_id"]
                progress=read_progress(output/"roles"/c/"run/client_jobs"/f"attempt_{attempt}"/c/"progress.json")
                if not item["triggered"] and progress.get("optimizer_steps",0)>=item["after_steps"]:
                    _stop_tree(nodes[c])
                    item["triggered"]=True
                    events.emit("node_process_tree_killed",client_id=c,attempt_id=attempt,completed_steps=progress["optimizer_steps"])
                if item["triggered"] and not item["restarted"] and attempt in attempts:
                    nodes[c]=start(c,commands[c],f"node-{c}-rejoin-{attempt}.log")
                    item["restarted"]=True
                    events.emit("node_restarted",client_id=c,after_attempt_id=attempt,node_id=mapping[c])
            totals={name:0 for name in peak}
            for child in processes:
                if child.poll() is not None: continue
                try:
                    parent=psutil.Process(child.pid)
                    totals[owners[child.pid]]+=sum(p.memory_info().rss for p in [parent,*parent.children(recursive=True)] if p.is_running())
                except (psutil.NoSuchProcess,psutil.AccessDenied): pass
            for name,total in totals.items(): peak[name]=max(peak[name],total)
            time.sleep(.1)
        if process.returncode!=0: raise RuntimeError("Flower deployment failed; inspect logs/run.log")
        final=read(output/"roles/server/run/final_results.json")
        if len(final["rounds"])!=package["attempts"]: raise RuntimeError("Incomplete deployment")
        expected=([len(mapping),len(mapping)-1,len(mapping),package["quorum"]-1,len(mapping)]
                  if schedule else [len(mapping)])
        actual=[len(r["accepted"]) for r in final["rounds"]]
        if actual!=expected or any(not item["triggered"] or not item["restarted"] for item in schedule):
            write_json(output/"gate-failure.json",{"expected":expected,"observed":actual,"schedule":schedule})
            raise RuntimeError(f"Recovery gate failed: expected {expected}, observed {actual}")
        if schedule and (final["rounds"][3]["status"]!="skipped"
                or final["rounds"][3]["checkpoint_hash"]!=final["rounds"][2]["checkpoint_hash"]):
            raise RuntimeError("Below-quorum checkpoint retention failed")
        result={"status":"completed","run_id":package["run_id"],"package":package,"network":network,
            "tls_authenticated":True,"node_map":mapping,"recovery_schedule":schedule,"accepted_counts":actual,
            "peak_process_tree_rss_bytes":peak,"memory_scope":"Sum of sampled RSS per role process tree; shared pages may be double counted",
            "wall_seconds":time.perf_counter()-started,"final":final,
            "isolation":{name:read(output/"roles"/name/"isolation.json") for name in ["server",*mapping]}}
        write_json(output/"result.json",result)
        print(json.dumps({"status":"completed","accepted_counts":actual,"peak_process_tree_rss_bytes":peak}),flush=True)
    finally:
        for child in reversed(processes): _stop_tree(child)
        for stream in streams: stream.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","run"])
    parser.add_argument("--local-data",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--port",type=int,default=19892)
    parser.add_argument("--offline",action="store_true")
    parser.add_argument("--no-recovery",action="store_true")
    args=parser.parse_args()
    if args.action=="prepare":
        if args.local_data is None: parser.error("--local-data is required")
        print(json.dumps(prepare(args.local_data.resolve(),args.output.resolve(),args.port,not args.no_recovery)))
    else: run(args.output.resolve(),args.offline)


if __name__=="__main__": main()
