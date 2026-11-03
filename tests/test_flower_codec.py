"""Optional Flower wire/API checks; no training metrics or LM results are produced."""

import importlib.util
import threading
import time
import unittest
from unittest.mock import patch

from fedlora_edge.state import TensorData, hash_state


@unittest.skipUnless(importlib.util.find_spec("flwr") is not None, "Flower optional environment required")
class FlowerCodecTests(unittest.TestCase):
    def setUp(self):
        # The runtime populates these before invoking either app. Standalone
        # codec tests provide the same task context without starting a runtime.
        from flwr.supercore.task_identity import TaskIdentity

        for name, value in (("_run_id", 1), ("_task_id", 1), ("_node_id", 0)):
            item = patch.object(TaskIdentity, name, value)
            item.start()
            self.addCleanup(item.stop)

    def test_named_tensor_round_trip_and_real_serialization_size(self):
        from flwr.app import Message, RecordDict
        from fedlora_edge.flower_app import _array_record, _tensor_state, serialized_message_bytes

        state = {"q.lora_B.weight": TensorData((1, 2), (1.25, -0.5)),
                 "v.lora_B.weight": TensorData((1,), (0,))}
        record = _array_record(state)
        restored = _tensor_state(record)
        self.assertEqual(hash_state(restored), hash_state(state))
        message = Message(RecordDict({"b": record}), dst_node_id=1, message_type="train", ttl=5)
        self.assertGreater(serialized_message_bytes(message), 12)

    def test_wrong_dtype_rejected_without_casting(self):
        import numpy as np
        from flwr.app import Array, ArrayRecord
        from fedlora_edge.flower_app import _tensor_state

        with self.assertRaises(ValueError):
            _tensor_state(ArrayRecord({"b": Array(np.array([1.0], dtype=np.float64))}))

    def test_current_clientapp_query_routing(self):
        from flwr.app import Context, Message, RecordDict
        from fedlora_edge.flower_app import make_client_app

        app = make_client_app()
        context = Context(run_id=1, node_id=2, node_config={"partition-id": 4},
                          state=RecordDict(), run_config={})
        request = Message(RecordDict(), dst_node_id=2, message_type="query.identity", ttl=5)
        response = app(request, context)
        self.assertEqual(response.content["metadata"]["client_id"], "4")

    def test_collection_deadline_bounds_a_blocked_transport_call(self):
        from fedlora_edge.flower_app import _collect

        release = threading.Event()

        class BlockedGrid:
            def push_messages(self, messages):
                return ["pending"]

            def pull_messages(self, pending):
                release.wait(timeout=5)
                return []

        try:
            started = time.monotonic()
            self.assertEqual(list(_collect(BlockedGrid(), [], timeout=0.05)), [])
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()
