"""Validation-only centralized MPNet representation diagnostic, not federation.

Train the last two or four encoder layers and the warm head with a persistent
optimizer. Never open confirmation/test queries or replace the published model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedlora_edge.checkpoint import atomic_json
from fedlora_edge.data import INTENTS, content_hash, normalize_query
from fedlora_edge.encoder_data import load_encoder_manifest, load_encoder_records, load_feature_split
from fedlora_edge.encoder_inference import load_head_checkpoint, verify_inference_assets
from fedlora_edge.training import classification_metrics


def validate_records(train, validation):
    """Reject duplicates, overlap, or test-origin rows before any optimization."""
    for rows in (train, validation):
        ids = [r["source_id"] for r in rows]
        queries = [normalize_query(r["query"]) for r in rows]
        if (not rows or len(set(ids)) != len(ids) or len(set(queries)) != len(queries)
                or any(not i.startswith("train:") for i in ids)
                or any(r["source_split"] != "train" or r["label"] not in INTENTS for r in rows)):
            raise ValueError("Only unique, identified training-origin development rows are allowed.")
    if ({r["source_id"] for r in train} & {r["source_id"] for r in validation}
            or {normalize_query(r["query"]) for r in train}
            & {normalize_query(r["query"]) for r in validation}):
        raise ValueError("Training and validation overlap.")
    if set(r["label"] for r in train) != set(INTENTS):
        raise ValueError("All twenty training classes must be present.")


def select_top_layers(encoder, count):
    layers = encoder.encoder.layer
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(layers):
        raise ValueError("Invalid number of trainable encoder layers.")
    encoder.requires_grad_(False)
    for layer in layers[-count:]:
        layer.requires_grad_(True)
    return [p for p in encoder.parameters() if p.requires_grad]


def parameter_hash(encoder, *, trainable):
    digest = hashlib.sha256()
    for name, parameter in sorted(encoder.named_parameters()):
        if parameter.requires_grad == trainable:
            digest.update(name.encode())
            digest.update(str(tuple(parameter.shape)).encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoTokenizer

    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs") or output == ROOT / "runs" or output.exists():
        raise ValueError("Choose a fresh directory inside ignored runs/.")
    if not torch.cuda.is_available():
        raise RuntimeError("This bounded reference requires the prepared CUDA environment.")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = load_encoder_manifest(args.dataset_dir)
    assets = verify_inference_assets(args.model_dir)
    if assets["files_sha256"] != manifest["encoder"]["files_sha256"]:
        raise ValueError("Dataset and encoder assets disagree.")
    saved = load_head_checkpoint(args.head_dir)
    if saved["identity"]["encoder_identity_hash"] != manifest["encoder_identity_hash"]:
        raise ValueError("Warm head belongs to a different encoder.")
    train = [r for i in range(5) for r in load_encoder_records(args.dataset_dir, i)]
    validation = load_encoder_records(args.dataset_dir)
    validate_records(train, validation)
    if len(train) != 2760 or len(validation) != 200:
        raise ValueError("This diagnostic requires the existing 2,760/200 development split.")
    Xval, yval, _ = load_feature_split(args.dataset_dir)
    cached_logits = Xval @ saved["state"]["head.weight"].to_numpy().T + saved["state"]["head.bias"].to_numpy()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.padding_side = "right"

    def tokenize(rows):
        tokens = tokenizer([r["query"] for r in rows], truncation=False, padding=False)
        lengths = [len(i) for i in tokens["input_ids"]]
        encoded = [{k: v[j][:384] for k, v in tokens.items()} for j in range(len(rows))]
        return encoded, sum(n > 384 for n in lengths)

    train_tokens, train_truncated = tokenize(train)
    val_tokens, val_truncated = tokenize(validation)
    if train_truncated or val_truncated:
        raise ValueError("Declare a separate truncation study; this diagnostic expects no truncated rows.")
    labels = torch.tensor([INTENTS.index(r["label"]) for r in train], device="cuda")
    counts = Counter(labels.cpu().tolist())
    weights = torch.tensor([len(train) / (20 * counts[i]) for i in range(20)], device="cuda")
    variants = [{"name": "last_two_layers", "layers": 2}, {"name": "last_four_layers", "layers": 4}]
    plan = {"kind": "centralized_representation_diagnostic", "run_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(), "split": "reused_validation",
        "variants": variants, "seed": 42, "epochs": 3, "microbatch_size": 8,
        "effective_batch_size": 32, "encoder_learning_rate": 1e-5, "head_learning_rate": 1e-3,
        "weight_decay": 0.01, "gradient_clip": 1., "precision": "FP32", "max_length": 384,
        "optimizer": "persistent AdamW across all minibatches and epochs",
        "class_weighting": "N/(20*n_class), loss normalized by summed weights per effective batch",
        "selection": "Highest validation accuracy, then macro F1; earliest epoch breaks remaining ties; complete all epochs.",
        "maximum_seconds_per_variant": 1800, "device": torch.cuda.get_device_name(),
        "manifest_hash": manifest["manifest_hash"], "training_records_hash": content_hash(train),
        "validation_records_hash": content_hash(validation), "train_count": len(train), "validation_count": len(validation),
        "initial_head_hash": saved["b_hash"], "initial_head_run_id": saved["run_id"],
        "asset_hash": assets["asset_hash"], "model_revision": manifest["model_revision"],
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ("tools/run_encoder_representation_study.py", "fedlora_edge/encoder_data.py",
             "fedlora_edge/encoder_inference.py", "fedlora_edge/training.py")},
        "environment": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "safetensors")},
        "interpretation": "Centralized public-training-data diagnostic only. Not LoRA, Flower, privacy evidence, or an independent confirmation result. Published model stays unchanged."}
    plan["config_hash"] = content_hash(plan)
    output.mkdir(parents=True)
    atomic_json(output / "plan.json", plan)
    summary = {"status": "running", "plan": plan, "results": []}
    atomic_json(output / "summary.json", summary)

    for variant in variants:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        directory = output / variant["name"]
        directory.mkdir()
        encoder = AutoModel.from_pretrained(args.model_dir, local_files_only=True, dtype=torch.float32).to("cuda")
        head = torch.nn.Linear(768, 20).to("cuda")
        head.load_state_dict({k.removeprefix("head."): v.to_torch().to("cuda") for k, v in saved["state"].items()})
        params = select_top_layers(encoder, variant["layers"])
        frozen_before = parameter_hash(encoder, trainable=False)
        adaptable_before = parameter_hash(encoder, trainable=True)
        trainable_count = sum(p.numel() for p in params) + sum(p.numel() for p in head.parameters())
        optimizer = torch.optim.AdamW([{"params": params, "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        history, selected, optimizer_steps = [], None, 0

        def logits(tokens):
            batch = tokenizer.pad(tokens, padding=True, return_tensors="pt").to("cuda")
            hidden = encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            return head(torch.nn.functional.normalize(pooled, p=2, dim=1))

        def evaluate(epoch):
            nonlocal selected
            encoder.eval()
            head.eval()
            with torch.inference_mode():
                scores = np.concatenate([logits(val_tokens[i:i + 16]).cpu().numpy() for i in range(0, len(validation), 16)])
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite validation scores.")
            if epoch == 0 and (not np.array_equal(scores.argmax(1), cached_logits.argmax(1))
                    or float(np.max(np.abs(scores - cached_logits))) > 1e-3):
                raise ValueError("Fresh encoder and published warm head failed initial parity.")
            predicted = [INTENTS[int(i)] for i in scores.argmax(1)]
            metrics = classification_metrics([r["label"] for r in validation], predicted)
            metrics["cross_entropy"] = float(torch.nn.functional.cross_entropy(torch.from_numpy(scores), torch.from_numpy(yval)))
            row = {"epoch": epoch, "optimizer_steps": optimizer_steps, "metrics": metrics}
            history.append(row)
            if selected is None or (metrics["accuracy"], metrics["macro_f1"]) > (selected["metrics"]["accuracy"], selected["metrics"]["macro_f1"]):
                selected = row
                state = {f"encoder.{n}": p.detach().cpu().contiguous() for n, p in encoder.named_parameters() if p.requires_grad}
                state.update({f"head.{n}": p.detach().cpu().contiguous() for n, p in head.named_parameters()})
                save_file(state, str(directory / "selected.safetensors"), metadata={"config_hash": plan["config_hash"], "variant": variant["name"]})
                atomic_json(directory / "selected.json", {"selection": row, "state_sha256": hashlib.sha256((directory / "selected.safetensors").read_bytes()).hexdigest()})
            atomic_json(directory / "progress.json", {"status": "running", "history": history, "selected": selected})
            print(f"{variant['name']} epoch {epoch}/3: validation {metrics['accuracy']:.1%}, worst {metrics['worst_domain_accuracy']:.1%}", flush=True)

        evaluate(0)
        status = "completed"
        for epoch in range(1, 4):
            encoder.eval()
            for layer in encoder.encoder.layer[-variant["layers"]:]:
                layer.train()
            head.train()
            generator = torch.Generator().manual_seed(42 + epoch)
            order = torch.randperm(len(train), generator=generator).tolist()
            for start in range(0, len(order), 32):
                if time.perf_counter() - started > plan["maximum_seconds_per_variant"]:
                    status = "time_budget_exhausted"
                    break
                group = order[start:start + 32]
                denominator = weights[labels[group]].sum()
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, len(group), 8):
                    idx = group[offset:offset + 8]
                    loss = torch.nn.functional.cross_entropy(logits([train_tokens[i] for i in idx]), labels[idx], weight=weights, reduction="sum") / denominator
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss.")
                    loss.backward()
                torch.nn.utils.clip_grad_norm_([*params, *head.parameters()], 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer_steps += 1
                if optimizer_steps % 25 == 0:
                    print(f"{variant['name']}: {optimizer_steps} optimizer steps, {time.perf_counter() - started:.1f}s", flush=True)
            if status != "completed":
                break
            evaluate(epoch)
        frozen_after = parameter_hash(encoder, trainable=False)
        adaptable_after = parameter_hash(encoder, trainable=True)
        if frozen_before != frozen_after or adaptable_before == adaptable_after:
            raise AssertionError("Frozen-layer/updated-layer invariant failed.")
        result = {"name": variant["name"], "status": status, "history": history, "selected": selected,
            "completed_epochs": history[-1]["epoch"], "optimizer_steps": optimizer_steps,
            "trainable_parameters": trainable_count, "raw_fp32_parameter_bytes": trainable_count * 4,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(), "wall_seconds": time.perf_counter() - started,
            "frozen_before": frozen_before, "frozen_after": frozen_after,
            "adaptable_before": adaptable_before, "adaptable_after": adaptable_after}
        atomic_json(directory / "result.json", result)
        summary["results"].append(result)
        atomic_json(output / "summary.json", summary)
        del optimizer, params, head, encoder
        torch.cuda.empty_cache()
    summary["status"] = "completed" if all(r["status"] == "completed" for r in summary["results"]) else "incomplete"
    atomic_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-dir", default="assets/encoder-bank20-v1")
    parser.add_argument("--model-dir", default="assets/reference-mpnet")
    parser.add_argument("--head-dir", default="runs/encoder-flower-balanced-v2/selected")
    run(parser.parse_args())
