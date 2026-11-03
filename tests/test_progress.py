"""A disappearing advisory progress snapshot must not reject a model update."""

import errno
import unittest
from pathlib import Path
from unittest.mock import patch

from fedlora_edge.checkpoint import read_progress


class ProgressReadTests(unittest.TestCase):
    def test_replacement_gap_preserves_last_snapshot_then_advances(self):
        previous = {"status": "training", "optimizer_steps": 1}
        with patch.object(Path, "read_text", side_effect=[FileNotFoundError(),
                          '{"status":"training","optimizer_steps":2}']):
            current = read_progress("progress.json", previous)
            self.assertEqual(current, previous)
            self.assertEqual(read_progress("progress.json", current)["optimizer_steps"], 2)

    def test_initial_missing_or_incomplete_snapshot_is_not_completion(self):
        for observation in (FileNotFoundError(), PermissionError(), '{"status":', '[]'):
            with self.subTest(observation=observation):
                kwargs = {"side_effect": observation} if isinstance(observation, Exception) else {"return_value": observation}
                with patch.object(Path, "read_text", **kwargs):
                    self.assertEqual(read_progress("progress.json"), {})

    def test_unrelated_io_failure_remains_visible(self):
        with patch.object(Path, "read_text", side_effect=OSError(errno.EIO, "read failure")):
            with self.assertRaises(OSError):
                read_progress("progress.json")


if __name__ == "__main__":
    unittest.main()
