import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fedlora_edge.data import content_hash
from fedlora_edge.encoder_flower import validate_config
from tests.test_encoder_flower import fixture_config
from tools.run_encoder_deployment import isolation_command


class DeploymentConfigTests(unittest.TestCase):
    def test_only_stale_activation_lease_is_retryable(self):
        from fedlora_edge.supernode_launcher import is_lease_conflict
        conflict="Exit Code: 305\n[code: 2004] Failed to activate SuperNode."
        self.assertTrue(is_lease_conflict(1,conflict))
        self.assertFalse(is_lease_conflict(0,conflict))
        self.assertFalse(is_lease_conflict(1,"[code: 2004] Failed to activate SuperNode."))
        self.assertFalse(is_lease_conflict(1,"Exit Code: 305\n[code: 2004] No SuperNode found with the given public key."))
        self.assertFalse(is_lease_conflict(1,"Exit Code: 305\n[code: 2005] could not be activated"))

    def test_activation_retry_is_bounded_and_does_not_change_command(self):
        from fedlora_edge.supernode_launcher import run
        with patch("fedlora_edge.supernode_launcher._run_once",return_value=(1,"Exit Code: 305\n[code: 2004] Failed to activate SuperNode.")) as start:
            with patch("fedlora_edge.supernode_launcher.time.monotonic",side_effect=[0,0,76]):
                with patch("fedlora_edge.supernode_launcher.time.sleep") as sleep:
                    self.assertEqual(run(["node","--same-key"],75),1)
            self.assertEqual(start.call_args.args[0],["node","--same-key"])
            self.assertEqual(start.call_count,1)
            sleep.assert_not_called()

    def test_transient_activation_failure_retries_the_same_registered_node(self):
        from fedlora_edge.supernode_launcher import run
        results=[(1,"Exit Code: 305\n[code: 2004] Failed to activate SuperNode."),(0,"SuperNode ID: 42")]
        with patch("fedlora_edge.supernode_launcher._run_once",side_effect=results) as start:
            with patch("fedlora_edge.supernode_launcher.time.monotonic",side_effect=[0,0,1,3]):
                with patch("fedlora_edge.supernode_launcher.time.sleep") as sleep:
                    self.assertEqual(run(["node","--same-key"],75),0)
            self.assertEqual(start.call_count,2)
            self.assertEqual([call.args[0] for call in start.call_args_list],[["node","--same-key"]]*2)
            sleep.assert_called_once_with(2.0)

    @unittest.skipUnless(os.name == "posix", "POSIX process-group cleanup test")
    def test_orphan_child_cannot_hold_activation_failure_pipe_open(self):
        import sys
        import time
        from fedlora_edge.supernode_launcher import _run_once
        script=("import subprocess,sys; "
                "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                "print('Exit Code: 305\\n[code: 2004] Failed to activate SuperNode.',flush=True); sys.exit(1)")
        started=time.monotonic()
        code,output=_run_once([sys.executable,"-c",script])
        self.assertEqual(code,1)
        self.assertIn("[code: 2004]",output)
        self.assertLess(time.monotonic()-started,5)

    @unittest.skipUnless(os.name == "posix", "POSIX process-group cleanup test")
    def test_startup_hang_is_bounded_but_activated_node_can_keep_running(self):
        import sys
        import time
        from fedlora_edge.supernode_launcher import _run_once
        started=time.monotonic()
        code,_=_run_once([sys.executable,"-c","import time; time.sleep(30)"],.2)
        self.assertEqual(code,124)
        self.assertLess(time.monotonic()-started,2)
        script="import time; print('INFO : SuperNode ID: 42',flush=True); time.sleep(.4)"
        started=time.monotonic()
        code,_=_run_once([sys.executable,"-c",script],.2)
        self.assertEqual(code,0)
        self.assertGreater(time.monotonic()-started,.3)

    @unittest.skipUnless(os.name == "posix", "POSIX process-group cleanup test")
    def test_ctrl_c_cleans_up_the_launchers_own_running_process_group(self):
        import signal
        import subprocess
        import sys
        import time
        with tempfile.TemporaryDirectory() as directory:
            pidfile=Path(directory)/"child.pid"
            child=("import os,time; from pathlib import Path; "
                   f"Path({str(pidfile)!r}).write_text(str(os.getpid())); "
                   "print('SuperNode ID: 42',flush=True); time.sleep(30)")
            wrapper=("from fedlora_edge.supernode_launcher import _run_once; "
                     f"_run_once({[sys.executable,'-c',child]!r})")
            with subprocess.Popen([sys.executable,"-c",wrapper],stdout=subprocess.PIPE,stderr=subprocess.STDOUT) as process:
                try:
                    deadline=time.monotonic()+5
                    while (not pidfile.exists() or not pidfile.read_text().isdigit()) and process.poll() is None and time.monotonic()<deadline:
                        time.sleep(.01)
                    self.assertTrue(pidfile.exists())
                    child_pid=int(pidfile.read_text())
                    process.send_signal(signal.SIGINT)
                    process.communicate(timeout=5)
                    state=Path(f"/proc/{child_pid}/stat")
                    self.assertTrue(not state.exists() or state.read_text().split()[2]=="Z")
                finally:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)

    def test_three_client_quorum_and_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            config,_=fixture_config(directory)
            path=Path(directory)/"manifest.json"
            manifest=json.loads(path.read_text())
            manifest["clients"]={str(i):{"count":4} for i in range(3)}
            manifest["manifest_hash"]=content_hash({k:v for k,v in manifest.items() if k!="manifest_hash"})
            path.write_text(json.dumps(manifest))
            config["identity"]["manifest_hash"]=manifest["manifest_hash"]
            config["identity"]["contract_hash"]=content_hash({k:v for k,v in config["identity"].items() if k!="contract_hash"})
            config["budgets"]={str(i):2 for i in range(3)}
            config["quorum"]=2
            config["external_node_interruptions"]=[{"attempt_id":2,"client_id":"1","after_steps":1}]
            self.assertEqual(len(validate_config(config)["clients"]),3)
            for change in ({"quorum":4},{"between_attempt_seconds":-1},
                           {"process_interruptions":config["external_node_interruptions"]}):
                with self.assertRaises(ValueError): validate_config({**config,**change})

    def test_sandbox_does_not_mount_other_node_private_keys(self):
        command=isolation_command(Path("/project"),Path("/project/app"),Path("/project/role0"),Path("/project/certificates"),"0")
        self.assertIn("/project/certificates/node_0.key",command)
        self.assertNotIn("/project/certificates",command)
        self.assertNotIn("/project/certificates/ca.key",command)
        self.assertNotIn("/project/certificates/node_1.key",command)
        self.assertIn("--tmpfs",command)


@unittest.skipUnless(importlib.util.find_spec("flwr") is not None,"Flower optional environment required")
class LocalPackageTests(unittest.TestCase):
    def test_readiness_checks_pinned_identity_and_zero_timeout_is_bounded(self):
        from types import SimpleNamespace
        from fedlora_edge.encoder_flower import await_rejoining_nodes
        from flwr.supercore.task_identity import TaskIdentity
        reply=SimpleNamespace(has_error=lambda:False,metadata=SimpleNamespace(src_node_id=123),
                              content={"metadata":{"client_id":"1"}})
        with patch.object(TaskIdentity,"_run_id",1),patch.object(TaskIdentity,"_task_id",1),patch.object(TaskIdentity,"_node_id",0):
            with patch("fedlora_edge.flower_app._collect",return_value=[(reply,0)]):
                result=await_rejoining_nodes(None,{"1":123},{"1"},1)
            self.assertEqual(result["ready"],["1"])
            self.assertEqual(result["still_unavailable"],[])
            with patch("fedlora_edge.flower_app._collect") as collect:
                result=await_rejoining_nodes(None,{"1":123},{"1"},0)
            collect.assert_not_called()
            self.assertEqual(result["still_unavailable"],["1"])

    def test_other_client_files_and_wrong_role_are_rejected(self):
        from fedlora_edge.encoder_deployment_app import local_config
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config,_=fixture_config(root)
            path=root/"manifest.json"
            manifest=json.loads(path.read_text())
            manifest["clients"]["0"].update(file="own.npz",records_file="own.json")
            manifest["manifest_hash"]=content_hash({k:v for k,v in manifest.items() if k!="manifest_hash"})
            path.write_text(json.dumps(manifest))
            config.update(role="client",local_client_id="0")
            config_path=root.parent/(root.name+"-config.json")
            self.addCleanup(lambda:config_path.unlink(missing_ok=True))
            config_path.write_text(json.dumps(config))
            with patch.dict(os.environ,{"FEDLORA_ENCODER_CONFIG":str(config_path)}):
                self.assertEqual(local_config("client")["local_client_id"],"0")
                with self.assertRaises(ValueError): local_config("server")
                (root/"other-client.npz").write_bytes(b"fixture")
                with self.assertRaises(ValueError): local_config("client")


if __name__=="__main__": unittest.main()
