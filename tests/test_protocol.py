"""Round contract and crash-safety checks, separate from real Flower rehearsals."""

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fedlora_edge.checkpoint import CheckpointStore, atomic_json
from fedlora_edge.protocol import ClientReply, RoundCoordinator, RoundRequest, validate_request
from fedlora_edge.state import TensorData, hash_state


IDENTITY = {"common_a_hash": "actual-a", "contract_hash": "contract", "model_revision": "pinned"}
BASE = {"q.lora_B": TensorData((1, 2), (0, 0)), "v.lora_B": TensorData((1,), (0,))}
DELTA = {"q.lora_B": TensorData((1, 2), (3, 4)), "v.lora_B": TensorData((1,), (0,))}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.coordinator = self.new_coordinator()
        self.requests = self.coordinator.begin({str(i): 2 for i in range(5)}, now=100)

    def new_coordinator(self, **kwargs):
        return RoundCoordinator(run_dir=self.directory.name, run_id="test", initial_state=BASE,
                                identity=IDENTITY, clients={str(i): 160 for i in range(5)},
                                deadline_seconds=10, **kwargs)

    def reply(self, client="0", **kwargs):
        return ClientReply(request=kwargs.get("request", self.requests[client]),
                           delta=kwargs.get("delta", DELTA), status=kwargs.get("status", "completed"),
                           telemetry=kwargs.get("telemetry", {"optimizer_steps": 2}))

    def receive(self, client="0", **kwargs):
        return self.coordinator.receive(self.reply(client, **kwargs),
                                        authenticated_client_id=client, received_at=105)

    def reason(self):
        return self.coordinator.rejected[-1]["reason"]

    def test_quorum_commits_and_survivor_weights_are_one_third(self):
        for client in ("0", "1", "2"):
            self.assertTrue(self.receive(client))
        result = self.coordinator.finish(now=110)
        self.assertEqual(result["status"], "committed")
        self.assertEqual(result["version_after"], 1)
        self.assertEqual(result["aggregation"]["coefficients"], dict.fromkeys(["0", "1", "2"], 1 / 3))
        self.assertEqual(result["absent"], ["3", "4"])
        self.assertEqual(CheckpointStore(self.directory.name).load()["state"], self.coordinator.state)

    def test_below_quorum_retains_state_but_consumes_attempt(self):
        self.receive()
        result = self.coordinator.finish(now=110)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(self.coordinator.version, 0)
        self.assertEqual(self.coordinator.state, BASE)
        self.assertEqual(CheckpointStore(self.directory.name).load()["attempt_id"], 1)

    def test_rejoining_client_receives_latest_committed_b(self):
        old = self.reply("4")
        for client in ("0", "1", "2"):
            self.receive(client)
        self.coordinator.finish(now=109)
        fresh = self.coordinator.begin({"3": 2, "4": 2}, now=200)
        self.assertEqual(fresh["4"].version, 1)
        self.assertEqual(fresh["4"].base_b_hash, hash_state(self.coordinator.state))
        self.assertFalse(self.coordinator.receive(old, authenticated_client_id="4", received_at=201))
        self.assertEqual(self.reason(), "attempt_id_mismatch")

    def test_every_echoed_identity_is_validated(self):
        replacements = {"run_id": "other", "attempt_id": 9, "version": 9,
                        "base_b_hash": "wrong", "contract_hash": "wrong", "common_a_hash": "wrong",
                        "model_revision": "other", "client_id": "4", "optimizer_steps": 3,
                        "deadline_unix": 999, "seed": 123, "manifest_hash": "other"}
        for field, value in replacements.items():
            with self.subTest(field=field):
                self.coordinator.seen.clear()
                self.assertFalse(self.receive(request=replace(self.requests["0"], **{field: value})))
                self.assertEqual(self.reason(), f"{field}_mismatch")

    def test_authenticated_identity_cannot_be_claimed_by_payload(self):
        self.assertFalse(self.coordinator.receive(self.reply(), authenticated_client_id="attacker",
                                                  received_at=105))
        self.assertEqual(self.reason(), "unknown_or_unselected_client")

    def test_wire_identifiers_reject_boolean_and_wrong_types(self):
        for field, value in (("attempt_id", True), ("version", False), ("optimizer_steps", True),
                             ("client_id", 0), ("seed", True), ("deadline_unix", math.inf)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                RoundRequest.from_dict({**self.requests["0"].to_dict(), field: value})

    def test_duplicate_cannot_satisfy_quorum(self):
        self.assertTrue(self.receive())
        self.assertFalse(self.receive())
        self.assertEqual(self.reason(), "duplicate_reply")
        self.assertEqual(len(self.coordinator.accepted), 1)

    def test_deadline_boundary_and_closed_attempt_reject(self):
        self.assertFalse(self.coordinator.receive(self.reply(), authenticated_client_id="0", received_at=110))
        self.assertEqual(self.reason(), "late_reply")
        self.coordinator.finish(now=110)
        self.assertFalse(self.receive())
        self.assertEqual(self.reason(), "attempt_closed")

    def test_nonfinite_missing_extra_and_wrong_shapes_reject(self):
        payloads = [{"q.lora_B": DELTA["q.lora_B"]},
                    {**DELTA, "extra": TensorData((1,), (1,))},
                    {**DELTA, "q.lora_B": TensorData((2, 1), (3, 4))}]
        payloads += [{**DELTA, "v.lora_B": TensorData((1,), (v,))}
                     for v in (math.nan, math.inf, -math.inf)]
        for payload in payloads:
            with self.subTest(payload=payload):
                self.coordinator.seen.clear()
                self.assertFalse(self.receive(delta=payload))

    def test_zero_work_partial_and_invalid_telemetry_reject(self):
        cases = [{"telemetry": {"optimizer_steps": 0}}, {"telemetry": {"optimizer_steps": True}},
                 {"telemetry": {"optimizer_steps": 3}}, {"status": "partial"},
                 {"telemetry": {"optimizer_steps": 2, "loss": math.nan}}]
        for case in cases:
            with self.subTest(case=case):
                self.coordinator.seen.clear()
                self.assertFalse(self.receive(**case))

    def test_zero_delta_with_completed_work_is_valid(self):
        self.assertTrue(self.receive(delta=BASE))

    def test_invalid_reply_is_observed_and_traffic_is_measured(self):
        self.coordinator.record_transport_reply("0", 128)
        self.assertFalse(self.receive(status="partial"))
        result = self.coordinator.finish(now=110)
        self.assertNotIn("0", result["absent"])
        self.assertEqual(result["serialized_reply_bytes"], 128)

    def test_client_validates_loaded_a_model_and_base(self):
        request = self.requests["0"]
        validate_request(request, BASE, IDENTITY, now=105)
        for field in IDENTITY:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_request(request, BASE, {**IDENTITY, field: "wrong"}, now=105)
        with self.assertRaises(ValueError):
            validate_request(request, DELTA, IDENTITY, now=105)
        with self.assertRaises(ValueError):
            validate_request(request, BASE, IDENTITY, now=110)

    def test_resume_consumes_unfinished_attempt_and_never_reuses_it(self):
        restored = self.new_coordinator(resume=True)
        self.assertEqual((restored.attempt_id, restored.version), (1, 0))
        requests = restored.begin({"0": 2}, now=200)
        self.assertEqual(requests["0"].attempt_id, 2)

    def test_resume_rejects_changed_contract(self):
        with self.assertRaises(ValueError):
            self.new_coordinator(resume=True, strategy="sample")

    def test_resume_recovers_durable_commit_missing_from_result_index(self):
        for client in ("0", "1", "2"):
            self.receive(client)
        self.coordinator.finish(now=109)
        (Path(self.directory.name) / "round_results.json").unlink()
        restored = self.new_coordinator(resume=True)
        self.assertEqual(len(restored.results), 1)
        self.assertEqual(restored.results[0]["version_after"], 1)
        self.assertEqual(restored.results[0]["status"], "committed")

    def test_extra_configuration_cannot_override_contract(self):
        with self.assertRaises(ValueError):
            self.new_coordinator(resume=True, config_extra={"run_id": "override"})

    def test_reply_before_first_attempt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = RoundCoordinator(run_dir=directory, run_id="new", initial_state=BASE,
                                            identity=IDENTITY, clients={"0": 1}, quorum=1)
            self.assertFalse(coordinator.receive(self.reply(), authenticated_client_id="0"))
            self.assertEqual(coordinator.rejected[-1]["reason"], "attempt_closed")

    def test_interrupted_replace_preserves_old_checkpoint(self):
        path = Path(self.directory.name) / "checkpoint.json"
        original = path.read_bytes()
        with patch("fedlora_edge.checkpoint.os.replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                atomic_json(path, {"broken": "new"})
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_corrupt_checkpoint_hash_rejects(self):
        path = Path(self.directory.name) / "checkpoint.json"
        data = json.loads(path.read_text())
        data["state"]["q.lora_B"]["values"][0] = 1
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            CheckpointStore(self.directory.name).load()


if __name__ == "__main__":
    unittest.main()
