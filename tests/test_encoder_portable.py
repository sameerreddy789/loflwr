"""Portable role configuration and code packaging; no model accuracy assertions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from fedlora_edge.data import INTENTS, content_hash
from fedlora_edge.encoder_data import sha256
from tools.build_encoder_app import build
from tools.configure_encoder_laptop import configure


def local_fixture(root, role):
    """Store only one role's archive; missing other archives catch stray reads."""
    import numpy as np
    root.mkdir()
    name="validation.npz" if role=="server" else f"clients/client_{role}.npz"
    feature=root/name
    feature.parent.mkdir(parents=True,exist_ok=True)
    values=np.zeros((2,768),dtype=np.float32)
    values[:,0]=1
    np.savez(feature,X=values,y=np.array([0,1],dtype=np.int64),source_ids=np.array(["train:1","train:2"]))
    def entry(stem):
        return {"file":stem+".npz","records_file":stem+".json","count":2,
                "sha256":sha256(feature)}
    manifest={"format_version":1,"kind":"mpnet_local_feature_dataset","feature_dim":768,
              "labels":list(INTENTS),"model_revision":"fixture-revision","encoder_identity_hash":"fixture-encoder",
              "clients":{str(i):entry(f"clients/client_{i}") for i in range(3)},"validation":entry("validation")}
    manifest["manifest_hash"]=content_hash(manifest)
    (root/"manifest.json").write_text(json.dumps(manifest))
    return root


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None,"NumPy optional environment required")
class PortableConfigTests(unittest.TestCase):
    def test_client_uses_own_archive_and_rejects_server_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dataset=local_fixture(root/"data","1")
            output=root/"client/config.json"
            config=configure(dataset,output,"client","1",run_id="portable-fixture")
            self.assertEqual(config["local_client_id"],"1")
            self.assertEqual(config["quorum"],2)
            self.assertEqual(config["run_id"],"portable-fixture")
            self.assertNotIn("node_map",config)
            self.assertNotIn("initial_checkpoint_dir",config)
            with self.assertRaises(FileExistsError): configure(dataset,output,"client","1")
            with self.assertRaises(ValueError):
                configure(dataset,root/"invalid.json","client","1",node_map=root/"not-readable")
            self.assertFalse((root/"invalid.json").exists())

    def test_server_uses_validation_only_and_requires_unique_complete_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dataset=local_fixture(root/"data","server")
            mapping=root/"nodes.json"
            mapping.write_text(json.dumps({"0":101,"1":102,"2":103}))
            result=configure(dataset,root/"server/config.json","server",node_map=mapping)
            self.assertEqual(result["node_map"],{"0":101,"1":102,"2":103})
            for invalid in ({},{"0":1,"1":1,"2":3},{"0":True,"1":2,"2":3}):
                mapping.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    configure(dataset,root/"invalid.json","server",node_map=mapping)
            self.assertFalse((root/"invalid.json").exists())

    def test_migration_rejects_a_head_from_another_encoder_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dataset=local_fixture(root/"data","server")
            mapping=root/"nodes.json"
            mapping.write_text(json.dumps({"0":101,"1":102,"2":103}))
            with patch("fedlora_edge.encoder_inference.load_head_checkpoint",return_value={"identity":{"encoder_identity_hash":"other"}}):
                with self.assertRaisesRegex(ValueError,"different encoder contracts"):
                    configure(dataset,root/"server/config.json","server",node_map=mapping,initial_checkpoint=root/"source")
            self.assertFalse((root/"server").exists())


class AppBundleTests(unittest.TestCase):
    def test_bundle_has_encoder_entry_points_and_no_private_or_runtime_files(self):
        with tempfile.TemporaryDirectory() as directory:
            app=build(Path(directory)/"app")
            components=tomllib.loads((app/"pyproject.toml").read_text())["tool"]["flwr"]["app"]["components"]
            self.assertEqual(components,{"serverapp":"fedlora_edge.encoder_deployment_app:server_app",
                                         "clientapp":"fedlora_edge.encoder_deployment_app:client_app"})
            for path in app.rglob("*"):
                if path.is_file():
                    self.assertTrue(path.suffix==".py" or path.name in {"README.md","LICENSE","pyproject.toml"})
            with self.assertRaises(FileExistsError): build(app)


if __name__=="__main__": unittest.main()
