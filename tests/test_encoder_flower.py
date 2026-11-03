"""Transport contracts for the head variant; fixture tests are not model evidence."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fedlora_edge.data import INTENTS, content_hash
from fedlora_edge.encoder_flower import client_seed, validate_config, load_initial_checkpoint
from fedlora_edge.encoder_head import HeadConfig
from fedlora_edge.protocol import RoundRequest
from fedlora_edge.state import TensorData, hash_state, state_schema
from fedlora_edge.training import derive_client_seed


def fixture_config(directory):
    root = Path(directory)
    manifest = {"format_version": 1, "kind": "mpnet_local_feature_dataset", "feature_dim": 768,
                "labels": list(INTENTS), "model_revision": "revision", "encoder_identity_hash": "encoder-hash",
                "clients": {str(i): {"count": 4} for i in range(5)}}
    manifest["manifest_hash"] = content_hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    state = {"head.weight": TensorData((20, 768), [0] * (20 * 768)),
             "head.bias": TensorData((20,), [0] * 20)}
    head = asdict(HeadConfig())
    identity = {"kind": "frozen_mpnet_linear_head", "model_revision": "revision",
                "common_a_hash": "encoder-hash", "encoder_identity_hash": "encoder-hash",
                "feature_dim": 768, "labels": list(INTENTS), "head_schema": state_schema(state),
                "head_config": head, "manifest_hash": manifest["manifest_hash"]}
    identity["contract_hash"] = content_hash(identity)
    return {"dataset_dir": str(root), "run_dir": str(root / "run"), "run_id": "fixture",
            "attempts": 3, "budgets": {str(i): 2 for i in range(5)}, "identity": identity,
            "head_config": head, "deadline_seconds": 10, "quorum": 3}, state


class EncoderFlowerConfigTests(unittest.TestCase):
    def test_warm_start_is_bound_to_model_dataset_and_actual_file(self):
        from fedlora_edge.checkpoint import CheckpointStore
        with tempfile.TemporaryDirectory() as directory:
            config, state = fixture_config(directory)
            manifest = validate_config(config)
            initial = Path(directory) / "initial"
            CheckpointStore(initial).save(run_id="source", attempt_id=5, version=5,
                state=state, identity=config["identity"], config_hash="source-config")
            config["initial_checkpoint_dir"] = str(initial)
            loaded = load_initial_checkpoint(config, manifest)
            self.assertEqual(loaded["state"], state)
            self.assertEqual(loaded["version"], 5)
            with self.assertRaises(ValueError):
                load_initial_checkpoint({**config, "initial_checkpoint_sha256": "wrong"}, manifest)
            with self.assertRaises(ValueError):
                load_initial_checkpoint(config, {**manifest, "manifest_hash": "another-dataset"})

    def test_identity_binds_shape_classes_encoder_and_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = fixture_config(directory)
            self.assertEqual(validate_config(config)["feature_dim"], 768)
            for change in ({"contract_hash": "wrong"}, {"feature_dim": 12},
                           {"common_a_hash": "another-encoder"}, {"labels": list(reversed(INTENTS))}):
                changed = {**config, "identity": {**config["identity"], **change}}
                with self.assertRaises(ValueError):
                    validate_config(changed)
            with self.assertRaises(ValueError):
                validate_config({**config, "head_config": {**config["head_config"], "learning_rate": 0.5}})

    def test_rejects_bad_deadline_work_and_failure_schedules(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = fixture_config(directory)
            cases = [{"evaluate_every": 0}, {"deadline_seconds": float("nan")}, {"quorum": 6},
                     {"attempts": True}, {"seed": True}, {"device": "cuda"},
                     {"budgets": {str(i): (1.5 if i == 1 else 2) for i in range(5)}},
                     {"process_interruptions": [{"attempt_id": 2, "client_id": "1", "after_steps": 3}]},
                     {"process_interruptions": [{"attempt_id": 2, "client_id": "1", "after_steps": 1}],
                      "simulated_failures": [{"attempt_id": 2, "client_id": "1"}]}]
            for change in cases:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    validate_config({**config, **change})

    def test_client_seed_matches_sequential_reference_and_ignores_run_name(self):
        args = dict(run_id="one", attempt_id=3, version=2, base_b_hash="b", contract_hash="c",
                    common_a_hash="a", model_revision="r", client_id="4", optimizer_steps=2,
                    deadline_unix=time.time() + 10, seed=42)
        first = client_seed(RoundRequest(**args))
        self.assertEqual(first, derive_client_seed(42, 3, "4"))
        self.assertEqual(first, client_seed(RoundRequest(**{**args, "run_id": "two"})))


@unittest.skipUnless(importlib.util.find_spec("flwr") is not None, "Flower optional environment required")
class EncoderFlowerCodecTests(unittest.TestCase):
    def setUp(self):
        from flwr.supercore.task_identity import TaskIdentity
        for name, value in (("_run_id", 1), ("_task_id", 1), ("_node_id", 0)):
            item = patch.object(TaskIdentity, name, value)
            item.start()
            self.addCleanup(item.stop)

    def test_only_named_head_arrays_cross_request_and_reply(self):
        from flwr.app import ConfigRecord, Context, Message, RecordDict
        from fedlora_edge.checkpoint import canonical_hash
        from fedlora_edge.encoder_flower import make_client_app
        from fedlora_edge.flower_app import _array_record, _tensor_state, serialized_message_bytes
        with tempfile.TemporaryDirectory() as directory:
            config, state = fixture_config(directory)
            manifest = validate_config(config)
            identity = config["identity"]
            request = RoundRequest("fixture", 1, 0, hash_state(state), identity["contract_hash"],
                identity["common_a_hash"], identity["model_revision"], "2", 2, time.time() + 10, 42,
                canonical_hash(manifest["clients"]["2"]))
            message = Message(RecordDict({"head": _array_record(state),
                "request": ConfigRecord({"json": json.dumps(request.to_dict())})}),
                dst_node_id=3, message_type="train", ttl=10)
            context = Context(run_id=1, node_id=3, node_config={"partition-id": 2},
                              state=RecordDict(), run_config={})
            result = {"status": "completed", "delta": state, "telemetry": {"optimizer_steps": 2}}
            with patch("fedlora_edge.encoder_flower._run_job", return_value=result) as job:
                reply = make_client_app(config)(message, context)
            self.assertEqual(set(reply.content), {"reply", "delta"})
            self.assertEqual(set(reply.content["delta"]), {"head.weight", "head.bias"})
            self.assertEqual(hash_state(_tensor_state(reply.content["delta"])), hash_state(state))
            self.assertGreater(serialized_message_bytes(reply), 61520)
            self.assertEqual(job.call_args[0][0].client_id, "2")

    def test_wrong_client_cannot_use_another_clients_request(self):
        from flwr.app import ConfigRecord, Context, Message, RecordDict
        from fedlora_edge.encoder_flower import make_client_app
        from fedlora_edge.flower_app import _array_record
        with tempfile.TemporaryDirectory() as directory:
            config, state = fixture_config(directory)
            identity = config["identity"]
            request = RoundRequest("fixture", 1, 0, hash_state(state), identity["contract_hash"],
                identity["common_a_hash"], identity["model_revision"], "1", 2, time.time() + 10, 42)
            message = Message(RecordDict({"head": _array_record(state),
                "request": ConfigRecord({"json": json.dumps(request.to_dict())})}),
                dst_node_id=3, message_type="train", ttl=10)
            context = Context(run_id=1, node_id=3, node_config={"partition-id": 2},
                              state=RecordDict(), run_config={})
            with patch("fedlora_edge.encoder_flower._run_job") as job, self.assertRaises(ValueError):
                make_client_app(config)(message, context)
            job.assert_not_called()


if __name__ == "__main__":
    unittest.main()
