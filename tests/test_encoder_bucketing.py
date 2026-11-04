"""Synthetic preparation fixtures; no BANKING77 or held-out queries are read."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fedlora_edge.data import DOMAINS, INTENTS, content_hash, write_json
from fedlora_edge.encoder_adapter import MODEL_ID, MODEL_REVISION, POOLING
from fedlora_edge.encoder_bucketing import (
    FEATURE_SETTINGS, REGISTRATION_KIND, assemble, build_plan, encode_bucketed, prepare, validate_plan,
)
from fedlora_edge.encoder_data import load_feature_split
from fedlora_edge.encoder_local_data import (
    assemble_local_manifest, validate_plan as validate_old_plan,
    validate_registration as validate_old_registration,
)
from tools.prepare_encoder_bucketed import validate_output

try:
    import numpy as np
except ImportError:
    np = None


def rehash(value, key):
    value[key] = content_hash({k: v for k, v in value.items() if k != key})
    return value


def fixture(layout="five"):
    rows = {str(i): [{"source_id": f"train:{i + 1}", "source_split": "train",
                     "query": f"synthetic owner {i}", "label": DOMAINS[i][1][0]}] for i in range(5)}
    rows["server"] = [{"source_id": "train:100", "source_split": "train",
                       "query": "synthetic validation", "label": INTENTS[0]}]
    encoder = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "pooling": POOLING,
               "max_length": 384, "files_sha256": {}}

    def entry(records):
        return {"count": len(records), "records_hash": content_hash(records),
                "label_counts": {records[0]["label"]: len(records)}}

    source = {"format_version": 1, "kind": "mpnet_local_feature_dataset", "feature_dim": 768,
              "labels": list(INTENTS), "encoder": encoder, "encoder_identity_hash": content_hash(encoder),
              "model_revision": MODEL_REVISION, "clients": {str(i): entry(rows[str(i)]) for i in range(5)},
              "validation": entry(rows["server"])}
    return build_plan(rehash(source, "manifest_hash"), layout), rows


def registration(plan, participant):
    source = plan["source_plan"]
    descriptor = source["participants"][participant]
    counts = {}
    for part in descriptor["parts"]:
        for label, count in part["label_counts"].items():
            counts[label] = counts.get(label, 0) + count
    stem = "validation" if participant == "server" else f"clients/client_{participant}"
    result = {
        "kind": REGISTRATION_KIND, "plan_hash": plan["plan_hash"], "participant": participant,
        "role": descriptor["role"], "recipe_hash": plan["recipe_hash"],
        "encoder_identity_hash": source["encoder_identity_hash"],
        "entry": {"file": stem + ".npz", "records_file": stem + ".json", "sha256": "a" * 64,
                  "records_hash": descriptor["parts"][0]["records_hash"] if len(descriptor["parts"]) == 1 else "b" * 64,
                  "count": sum(counts.values()), "label_counts": counts},
    }
    return rehash(result, "registration_hash")


class BucketedIdentityTests(unittest.TestCase):
    def test_new_plan_is_metadata_only_and_rejected_by_old_contract(self):
        plan, rows = fixture("three")
        self.assertNotEqual(plan["plan_hash"], plan["source_plan"]["plan_hash"])
        self.assertNotEqual(plan["feature_settings"], plan["source_plan"]["feature_settings"])
        self.assertEqual(plan["feature_settings"]["batch_size"], 8)
        self.assertEqual(plan["feature_settings"]["cpu_threads"], 1)
        self.assertEqual(set(plan["source_plan"]["participants"]), {"0", "1", "2", "server"})
        for records in rows.values():
            self.assertNotIn(records[0]["query"], json.dumps(plan))
        with self.assertRaises(ValueError):
            validate_old_plan(plan)

    def test_plan_tampering_fails_with_and_without_recomputed_hash(self):
        for rehash_plan in (False, True):
            plan, _ = fixture()
            plan["feature_settings"]["batch_size"] = 16
            if rehash_plan:
                rehash(plan, "plan_hash")
            with self.assertRaises(ValueError):
                validate_plan(plan)
        plan, _ = fixture()
        plan["source_plan"]["participants"]["0"]["parts"][0]["count"] = 9
        rehash(plan["source_plan"], "plan_hash")
        rehash(plan, "plan_hash")
        with self.assertRaises(ValueError):
            validate_plan(plan)
        plan, _ = fixture()
        plan["source_plan"]["encoder"]["pooling"] = "modified_pooling"
        plan["source_plan"]["encoder_identity_hash"] = content_hash(plan["source_plan"]["encoder"])
        rehash(plan["source_plan"], "plan_hash")
        rehash(plan, "plan_hash")
        with self.assertRaisesRegex(ValueError, "pooling recipe"):
            validate_plan(plan)

    def test_metadata_assembly_keeps_new_identity_and_does_not_open_files(self):
        plan, _ = fixture("three")
        registrations = [registration(plan, participant) for participant in plan["source_plan"]["participants"]]
        originals = copy.deepcopy(registrations)
        with patch("pathlib.Path.read_text", side_effect=AssertionError("No metadata-assembly reads")):
            manifest = assemble(plan, registrations)
        self.assertEqual(registrations, originals)
        self.assertEqual(manifest["kind"], "mpnet_local_feature_dataset")
        self.assertEqual(manifest["local_preparation_plan_hash"], plan["plan_hash"])
        self.assertEqual(manifest["feature_recipe_hash"], plan["recipe_hash"])
        self.assertEqual(manifest["feature_settings"], FEATURE_SETTINGS)
        self.assertEqual(manifest["clients"]["0"]["count"], 2)
        with self.assertRaises(ValueError):
            validate_old_registration(plan["source_plan"], registrations[0])
        with self.assertRaises(ValueError):
            assemble_local_manifest(plan["source_plan"], registrations)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            assemble(plan, registrations + registrations[:1])
        with self.assertRaisesRegex(ValueError, "missing"):
            assemble(plan, registrations[:-1])

    def test_rehashed_registration_tampering_cannot_cross_identity_or_paths(self):
        plan, _ = fixture()
        for mutate in (
            lambda item: item.update(plan_hash=plan["source_plan"]["plan_hash"]),
            lambda item: item.update(recipe_hash="0" * 64),
            lambda item: item.update(kind="mpnet_local_registration_v1"),
            lambda item: item.update(queries=["unexpected raw data"]),
            lambda item: item["entry"].update(file="../other.npz"),
            lambda item: item.update(encoder_identity_hash="0" * 64),
        ):
            registrations = [registration(plan, participant) for participant in plan["source_plan"]["participants"]]
            mutate(registrations[0])
            rehash(registrations[0], "registration_hash")
            with self.assertRaises(ValueError):
                assemble(plan, registrations)

    def test_cli_output_is_fresh_and_confined_to_ignored_project_roots(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("tools.prepare_encoder_bucketed.ROOT", root):
                self.assertEqual(validate_output(root / "runs/new/plan.json"), root / "runs/new/plan.json")
                with self.assertRaises(ValueError):
                    validate_output(root / "tracked/result.json")
                with self.assertRaises(ValueError):
                    validate_output(root / "runs/../../escape.json")
                with self.assertRaises(ValueError):
                    validate_output(root / "assets")
                existing = root / "runs/used"
                existing.mkdir(parents=True)
                with self.assertRaises(FileExistsError):
                    validate_output(existing)


@unittest.skipIf(np is None, "Optional NumPy not installed")
class BucketedPreparationTests(unittest.TestCase):
    def test_own_archive_remains_loader_compatible_and_aligned(self):
        plan, rows = fixture("three")
        X = np.zeros((2, 768), dtype=np.float32)
        X[0, 0], X[1, 1] = 1, 1
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = []
            for participant in ("1", "0"):
                path = root / f"{participant}.json"
                write_json(path, {"records": rows[participant], "records_hash": content_hash(rows[participant])})
                paths.append(path)
            output = root / "prepared"
            with patch("fedlora_edge.encoder_bucketing._encode_records", return_value=(X, {"fixture": True})) as encoding:
                own = prepare(plan, paths, output, "unused-fixture", "0")
            self.assertEqual(encoding.call_args.args[0], rows["0"] + rows["1"])
            registrations = [own if participant == "0" else registration(plan, participant)
                             for participant in plan["source_plan"]["participants"]]
            write_json(output / "manifest.json", assemble(plan, registrations))
            actual, labels, ids = load_feature_split(output, "0")
            np.testing.assert_array_equal(actual, X)
            self.assertEqual(ids.tolist(), ["train:1", "train:2"])
            self.assertEqual(labels.tolist(), [INTENTS.index(rows["0"][0]["label"]), INTENTS.index(rows["1"][0]["label"])])
            self.assertFalse((output / "clients/client_1.npz").exists())
            self.assertFalse((output / "validation.npz").exists())
            self.assertNotIn(rows["0"][0]["query"], (output / "registration.json").read_text())
            with self.assertRaises(FileExistsError):
                prepare(plan, paths, output, "unused-fixture", "0")

    def test_wrong_ownership_and_nonfinite_features_fail_before_output_creation(self):
        plan, rows = fixture()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "other.json"
            write_json(path, {"records": rows["1"], "records_hash": content_hash(rows["1"])})
            output = root / "prepared"
            with patch("fedlora_edge.encoder_bucketing._encode_records", side_effect=AssertionError("Validate first")):
                with self.assertRaisesRegex(ValueError, "do not belong"):
                    prepare(plan, [path], output, "unused-fixture", "0")
            self.assertFalse(output.exists())
            with patch("fedlora_edge.encoder_bucketing._encode_records",
                       return_value=(np.full((1, 768), np.nan, dtype=np.float32), {})):
                with self.assertRaisesRegex(ValueError, "invalid"):
                    prepare(plan, [path], output, "unused-fixture", "1")
            self.assertFalse(output.exists())


@unittest.skipUnless(np is not None and importlib.util.find_spec("torch"), "Optional Torch/NumPy not installed")
class BucketedEncodingTests(unittest.TestCase):
    def setUp(self):
        import torch

        class Tokenizer:
            padding_side = "right"
            truncation_side = "right"
            values = {"long": [11] * 400, "short": [3], "medium": [7] * 5, "same": [9]}

            def __init__(self):
                self.calls, self.prepared, self.batch_rows = 0, [], []

            def __call__(self, queries, **kwargs):
                self.calls += 1
                if kwargs != {"add_special_tokens": False, "padding": False,
                              "truncation": False, "return_attention_mask": False}:
                    raise AssertionError("Expected one raw string-tokenization pass")
                return {"input_ids": [list(self.values[query]) for query in queries]}

            def num_special_tokens_to_add(self, pair=False):
                return 2

            def prepare_for_model(self, ids, **kwargs):
                if kwargs != {"add_special_tokens": True, "truncation": True, "max_length": 384,
                              "return_attention_mask": True, "padding": False}:
                    raise AssertionError("Expected special-aware truncation")
                encoded = {"input_ids": [101] + ids[:382] + [102],
                           "attention_mask": [1] * min(len(ids) + 2, 384)}
                self.prepared.append(encoded)
                return encoded

            def pad(self, rows, **kwargs):
                longest = max(len(row["input_ids"]) for row in rows)
                self.batch_rows.append([row["input_ids"][1] for row in rows])
                return {key: torch.tensor([row[key] + [0] * (longest - len(row[key])) for row in rows], dtype=torch.long)
                        for key in ("input_ids", "attention_mask")}

        class Encoder(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                if torch.is_grad_enabled():
                    raise AssertionError("Expected inference_mode")
                hidden = torch.zeros((*input_ids.shape, 768), dtype=torch.float32)
                hidden[:, :, 0] = input_ids
                hidden[:, :, 1] = 1
                hidden[attention_mask == 0] = 99999
                return SimpleNamespace(last_hidden_state=hidden)

        self.tokenizer = Tokenizer()
        self.encoder = Encoder().eval()

    def test_order_special_tokens_truncation_masking_and_work_accounting(self):
        queries = ["long", "short", "medium", "same"]
        X, truncated, work = encode_bucketed(self.encoder, self.tokenizer, queries, batch_size=2)
        self.assertEqual(self.tokenizer.calls, 1)
        self.assertEqual(self.tokenizer.batch_rows, [[3, 9], [7, 11]])
        self.assertTrue(all(row["input_ids"][0] == 101 and row["input_ids"][-1] == 102
                            for row in self.tokenizer.prepared))
        means = [(101 + 382 * 11 + 102) / 384, (101 + 3 + 102) / 3,
                 (101 + 5 * 7 + 102) / 7, (101 + 9 + 102) / 3]
        expected = np.asarray([[value, 1] for value in means], dtype=np.float32)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(X[:, :2], expected, rtol=0, atol=1e-7)
        self.assertEqual(truncated, 1)
        self.assertEqual(work["original_token_count"], 402 + 3 + 7 + 3)
        self.assertEqual(work["clipped_token_count"], 384 + 3 + 7 + 3)
        self.assertEqual(work["baseline_padded_token_slots"], 2 * 384 + 2 * 7)
        self.assertEqual(work["candidate_padded_token_slots"], 2 * 3 + 2 * 384)
        self.assertEqual(work["candidate_attention_cells"], 2 * 3**2 + 2 * 384**2)

    def test_equal_length_order_is_stable_and_final_batch_is_counted(self):
        X, _, work = encode_bucketed(self.encoder, self.tokenizer, ["same", "short", "medium"], 2)
        self.assertEqual(self.tokenizer.batch_rows, [[9, 3], [7]])
        self.assertEqual(X.shape, (3, 768))
        self.assertEqual(work["batch_count"], 2)
        self.assertEqual(work["candidate_padded_token_slots"], 2 * 3 + 7)
        self.assertEqual(work["candidate_attention_cells"], 2 * 3**2 + 7**2)

    def test_empty_bad_arguments_and_train_mode_are_rejected(self):
        for queries, size in (([], 8), ([" "], 8), ([1], 8), (["short"], True),
                              (["short"], 0), (["short"], 1.5), ("short", 8)):
            with self.assertRaises(ValueError):
                encode_bucketed(self.encoder, self.tokenizer, queries, size)
        self.encoder.train()
        with self.assertRaisesRegex(ValueError, "evaluation mode"):
            encode_bucketed(self.encoder, self.tokenizer, ["short"])
        self.assertEqual(self.tokenizer.calls, 0)

    def test_trainable_parameters_wrong_precision_and_padding_are_rejected(self):
        import torch
        self.encoder.weight = torch.nn.Parameter(torch.ones(1))
        with self.assertRaisesRegex(ValueError, "frozen"):
            encode_bucketed(self.encoder, self.tokenizer, ["short"])
        self.encoder.requires_grad_(False).double()
        with self.assertRaisesRegex(ValueError, "CPU FP32"):
            encode_bucketed(self.encoder, self.tokenizer, ["short"])
        self.encoder.float()
        self.tokenizer.padding_side = "left"
        with self.assertRaisesRegex(ValueError, "right padding"):
            encode_bucketed(self.encoder, self.tokenizer, ["short"])


if __name__ == "__main__":
    unittest.main()
