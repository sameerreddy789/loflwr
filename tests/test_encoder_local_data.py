"""Local-data boundary fixtures; no real BANKING77/test queries are loaded."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fedlora_edge.data import DOMAINS, INTENTS, content_hash, write_json
from fedlora_edge.encoder_adapter import MODEL_ID, MODEL_REVISION, POOLING
from fedlora_edge.encoder_data import load_feature_split
from fedlora_edge.encoder_local_data import (
    FEATURE_SETTINGS, assemble_local_manifest, build_local_plan, encode_batches,
    load_own_records, prepare_local_features, validate_plan, validate_registration)

try:
    import numpy as np
except ImportError:
    np = None


def fixture():
    rows = {str(i): [{"source_id": f"train:{i+1}", "source_split": "train",
                     "query": f"fixture domain {i}", "label": DOMAINS[i][1][0]}] for i in range(5)}
    rows["server"] = [{"source_id": "train:100", "source_split": "train",
                       "query": "fixture validation", "label": INTENTS[0]}]
    encoder = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "pooling": POOLING,
               "max_length": 384, "files_sha256": {}}

    def entry(records):
        return {"count": len(records), "records_hash": content_hash(records),
                "label_counts": {records[0]["label"]: len(records)}}
    source = {"format_version": 1, "kind": "mpnet_local_feature_dataset", "feature_dim": 768,
              "labels": list(INTENTS), "encoder": encoder, "encoder_identity_hash": content_hash(encoder),
              "model_revision": MODEL_REVISION, "clients": {str(i): entry(rows[str(i)]) for i in range(5)},
              "validation": entry(rows["server"])}
    source["manifest_hash"] = content_hash(source)
    return source, rows


def save_bundle(path, rows):
    write_json(path, {"records": rows, "records_hash": content_hash(rows)})
    return path


def fake_registration(plan, participant):
    descriptor = plan["participants"][participant]
    counts = {}
    for part in descriptor["parts"]:
        for label, count in part["label_counts"].items():
            counts[label] = counts.get(label, 0) + count
    stem = "validation" if participant == "server" else f"clients/client_{participant}"
    entry = {"file": stem+".npz", "records_file": stem+".json", "sha256": "a"*64,
             "records_hash": descriptor["parts"][0]["records_hash"] if len(descriptor["parts"]) == 1 else "b"*64,
             "count": sum(counts.values()), "label_counts": counts}
    registration = {"kind": "mpnet_local_registration_v1", "participant": participant,
                    "role": descriptor["role"], "plan_hash": plan["plan_hash"],
                    "encoder_identity_hash": plan["encoder_identity_hash"], "entry": entry}
    registration["registration_hash"] = content_hash(registration)
    return registration


class LocalPlanAndRecordsTests(unittest.TestCase):
    def test_three_layout_groups_original_domains_and_contains_no_queries(self):
        source, rows = fixture()
        plan = build_local_plan(source, "three")
        self.assertEqual([p["original_client"] for p in plan["participants"]["0"]["parts"]], ["0", "1"])
        self.assertEqual([p["original_client"] for p in plan["participants"]["1"]["parts"]], ["2", "3"])
        self.assertEqual(set(plan["participants"]), {"0", "1", "2", "server"})
        for values in rows.values():
            self.assertNotIn(values[0]["query"], json.dumps(plan))
        self.assertNotIn("confirmation", plan)
        self.assertNotIn("test", plan)

    def test_modified_plan_or_wrong_source_identity_fails(self):
        source, _ = fixture()
        plan = build_local_plan(source)
        plan["feature_settings"]["max_length"] = 128
        with self.assertRaisesRegex(ValueError, "plan_hash"):
            validate_plan(plan)
        source["encoder_identity_hash"] = "0"*64
        source["manifest_hash"] = content_hash({k:v for k,v in source.items() if k != "manifest_hash"})
        with self.assertRaisesRegex(ValueError, "encoder identity"):
            build_local_plan(source)

    def test_only_own_file_is_required_and_other_bundle_is_rejected(self):
        source, rows = fixture()
        plan = build_local_plan(source)
        with tempfile.TemporaryDirectory() as folder:
            path = save_bundle(Path(folder)/"own.json", rows["0"])
            self.assertEqual(load_own_records(plan, "0", [path]), rows["0"])
            with self.assertRaisesRegex(ValueError, "do not belong"):
                load_own_records(plan, "1", [path])
            with self.assertRaisesRegex(ValueError, "exactly"):
                load_own_records(plan, "0", [path, path])

    def test_grouped_bundles_use_frozen_order(self):
        source, rows = fixture()
        plan = build_local_plan(source, "three")
        with tempfile.TemporaryDirectory() as folder:
            paths = [save_bundle(Path(folder)/f"{i}.json", rows[str(i)]) for i in (0, 1)]
            self.assertEqual(load_own_records(plan, "0", paths[::-1]), rows["0"] + rows["1"])

    def test_changed_record_hash_is_rejected(self):
        source, rows = fixture()
        plan = build_local_plan(source)
        with tempfile.TemporaryDirectory() as folder:
            path = save_bundle(Path(folder)/"own.json", rows["0"])
            data = json.loads(path.read_text())
            data["records"][0]["query"] = "tampered"
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_own_records(plan, "0", [path])

    def test_test_rows_invalid_ids_and_empty_text_fail_even_with_matching_hash(self):
        for update in ({"source_split": "test"}, {"source_id": "test:1"}, {"query": "  "},
                       {"source_id": "train:0"}, {"label": "unknown_intent"}):
            source, rows = fixture()
            plan = build_local_plan(source)
            rows["0"][0].update(update)
            plan["participants"]["0"]["parts"][0]["records_hash"] = content_hash(rows["0"])
            plan["plan_hash"] = content_hash({k:v for k,v in plan.items() if k != "plan_hash"})
            with tempfile.TemporaryDirectory() as folder:
                path = save_bundle(Path(folder)/"own.json", rows["0"])
                with self.assertRaisesRegex(ValueError, "approved original-training"):
                    load_own_records(plan, "0", [path])

    def test_duplicate_normalized_queries_fail(self):
        source, rows = fixture()
        rows["0"].append({**rows["0"][0], "source_id": "train:99", "query": rows["0"][0]["query"].upper()})
        source["clients"]["0"].update(count=2, records_hash=content_hash(rows["0"]), label_counts={rows["0"][0]["label"]:2})
        source["manifest_hash"] = content_hash({k:v for k,v in source.items() if k != "manifest_hash"})
        plan = build_local_plan(source)
        with tempfile.TemporaryDirectory() as folder:
            path = save_bundle(Path(folder)/"own.json", rows["0"])
            with self.assertRaisesRegex(ValueError, "Duplicate local"):
                load_own_records(plan, "0", [path])

    def test_server_reads_only_approved_validation_bundle(self):
        source, rows = fixture()
        plan = build_local_plan(source)
        with tempfile.TemporaryDirectory() as folder:
            path = save_bundle(Path(folder)/"validation.json", rows["server"])
            self.assertEqual(load_own_records(plan, "server", [path]), rows["server"])
            with self.assertRaises(ValueError):
                load_own_records(plan, "confirmation", [path])

    def test_metadata_assembly_needs_no_data_files_and_rejects_duplicates(self):
        source, _ = fixture()
        plan = build_local_plan(source, "three")
        registrations = [fake_registration(plan, participant) for participant in plan["participants"]]
        with patch("pathlib.Path.read_text", side_effect=AssertionError("No file reads allowed")):
            manifest = assemble_local_manifest(plan, registrations)
        self.assertEqual(set(manifest["clients"]), {"0", "1", "2"})
        self.assertEqual(manifest["clients"]["0"]["count"], 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            assemble_local_manifest(plan, registrations + registrations[:1])
        with self.assertRaisesRegex(ValueError, "missing"):
            assemble_local_manifest(plan, registrations[:-1])

    def test_registration_rejects_paths_extra_content_and_wrong_identity(self):
        source, _ = fixture()
        plan = build_local_plan(source)
        for mutate in (lambda r:r["entry"].update(file="../other.npz"), lambda r:r.update(queries=["leak"]),
                       lambda r:r.update(encoder_identity_hash="bad")):
            registration = fake_registration(plan, "0")
            mutate(registration)
            registration["registration_hash"] = content_hash({k:v for k,v in registration.items() if k != "registration_hash"})
            with self.assertRaises(ValueError):
                validate_registration(plan, registration)


@unittest.skipIf(np is None, "Optional NumPy not installed")
class LocalFeaturePreparationTests(unittest.TestCase):
    def test_own_archive_roundtrip_without_other_clients_or_server_data(self):
        source, rows = fixture()
        plan = build_local_plan(source)
        X = np.zeros((1, 768), dtype=np.float32)
        X[:, 0] = 1
        with tempfile.TemporaryDirectory() as folder:
            path = save_bundle(Path(folder)/"input.json", rows["0"])
            output = Path(folder)/"participant"
            with patch("fedlora_edge.encoder_local_data.encode_local_records", return_value=(X, {"fixture":True})):
                own = prepare_local_features(plan, [path], output, "unused-fixture-assets", "0")
            registrations = [own if participant == "0" else fake_registration(plan, participant) for participant in plan["participants"]]
            write_json(output/"manifest.json", assemble_local_manifest(plan, registrations))
            actual, y, ids = load_feature_split(output, "0")
            np.testing.assert_array_equal(actual, X)
            self.assertEqual(y.tolist(), [0])
            self.assertEqual(ids.tolist(), ["train:1"])
            self.assertFalse((output/"clients/client_1.npz").exists())
            self.assertFalse((output/"validation.npz").exists())
            self.assertNotIn("fixture domain 0", (output/"registration.json").read_text())
            with self.assertRaises(FileExistsError):
                prepare_local_features(plan, [path], output, "unused", "0")

    def test_invalid_input_does_not_load_encoder_or_create_output(self):
        source, rows = fixture()
        plan = build_local_plan(source)
        with tempfile.TemporaryDirectory() as folder:
            path = save_bundle(Path(folder)/"input.json", rows["1"])
            output = Path(folder)/"participant"
            with patch("fedlora_edge.encoder_local_data.encode_local_records", side_effect=AssertionError("Must validate first")):
                with self.assertRaises(ValueError):
                    prepare_local_features(plan, [path], output, "unused", "0")
            self.assertFalse(output.exists())

    def test_masked_pooling_normalization_batch_parity_and_truncation(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional Torch not installed")

        class Tokenizer:
            def __call__(self, queries, **kwargs):
                lengths = [2 if query == "short" else 400 for query in queries]
                if kwargs.get("return_length"):
                    return {"length": lengths}
                lengths = [min(n, kwargs["max_length"]) for n in lengths]
                ids = torch.zeros((len(queries), max(lengths)), dtype=torch.int64)
                mask = torch.zeros_like(ids)
                for i,n in enumerate(lengths):
                    ids[i,:n] = torch.arange(1,n+1)
                    mask[i,:n] = 1
                return {"input_ids": ids, "attention_mask": mask}

        class Encoder:
            def __call__(self, input_ids, attention_mask):
                self_gradient = torch.is_grad_enabled()
                if self_gradient:
                    raise AssertionError("Features must use inference mode")
                hidden = torch.zeros((*input_ids.shape,768),dtype=torch.float32)
                hidden[:,:,0] = input_ids
                hidden[:,:,1] = 1
                hidden[attention_mask == 0] = 99999
                return SimpleNamespace(last_hidden_state=hidden)

        X,truncated = encode_batches(Encoder(),Tokenizer(),["short","long"],2)
        single,_ = encode_batches(Encoder(),Tokenizer(),["short","long"],1)
        expected = np.array([[1.5,1],[192.5,1]],dtype=np.float32)
        expected /= np.linalg.norm(expected,axis=1,keepdims=True)
        np.testing.assert_allclose(X[:,:2],expected,atol=1e-7)
        np.testing.assert_allclose(X,single,atol=1e-7)
        self.assertEqual(truncated,1)


if __name__ == "__main__":
    unittest.main()
