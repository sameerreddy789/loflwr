"""Separate MPNet classification variant: common frozen A, trainable B and head.

The historical SmolLM2 completion contract is unchanged. This module performs
fresh encoder forward passes; frozen-reference feature caches are not training
inputs after adapters change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .data import INTENTS, content_hash, write_json
from .state import (TensorData, TensorState, finite_state, hash_state,
                    state_from_dict, state_nbytes, state_schema, state_to_dict)
from .training import _MemoryMonitor, _seed_everything, classification_metrics, example_indices

MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
WEIGHT_SHA256 = "78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658"
ADAPTER_SETTINGS = {"rank": 8, "alpha": 16, "dropout": 0.05,
                    "target_modules": ["q", "v"], "scaling": 2.0}
POOLING = "attention_mask_mean_then_l2_normalize"
SCORING_RULE = "twenty_way_linear_head_argmax"


@dataclass(frozen=True)
class EncoderAdapterConfig:
    adapter_lr: float = 2e-4
    head_lr: float = 1e-2
    microbatch_size: int = 4
    gradient_accumulation: int = 1
    max_length: int = 384
    gradient_clip: float = 1.0
    weight_decay: float = 0.0
    cpu_threads: int = 1
    eval_batch_size: int = 16
    activation_checkpointing: bool = False

    def __post_init__(self):
        for name in ("adapter_lr", "head_lr", "gradient_clip", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric.")
            if value < 0 or (name != "weight_decay" and value == 0):
                raise ValueError(f"Invalid {name}.")
        for name in ("microbatch_size", "gradient_accumulation", "max_length", "cpu_threads", "eval_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_length > 384:
            raise ValueError("This variant caps sequence length at the reference encoder's 384 tokens.")
        if not isinstance(self.activation_checkpointing, bool):
            raise ValueError("activation_checkpointing must be boolean.")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_encoder_assets(asset_dir: str | Path) -> dict:
    """Verify the existing offline Hub cache, including actual pinned weights."""
    root = Path(asset_dir)
    files = {path.relative_to(root).as_posix(): _file_hash(path)
             for path in sorted(root.rglob("*")) if path.is_file() and ".cache" not in path.parts}
    if files.get("model.safetensors") != WEIGHT_SHA256:
        raise ValueError("MPNet weights do not match the pinned model revision.")
    for name in files:
        if name == "model.safetensors":
            continue
        metadata = root / ".cache/huggingface/download" / (name + ".metadata")
        if not metadata.is_file() or metadata.read_text(encoding="utf-8").splitlines()[0] != MODEL_REVISION:
            raise ValueError(f"Unverified encoder asset revision: {name}")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if name not in files:
            raise ValueError(f"Missing offline encoder asset: {name}")
    return {"files_sha256": files, "asset_hash": content_hash(files)}


def _build(asset_dir, seed):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    _seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(str(asset_dir), local_files_only=True)
    tokenizer.padding_side = "right"
    base = AutoModel.from_pretrained(str(asset_dir), local_files_only=True, dtype=torch.float32)
    if base.config.model_type != "mpnet" or base.config.hidden_size != 768 or base.config.num_hidden_layers != 12:
        raise ValueError("Unexpected MPNet architecture.")
    expected = {f"encoder.layer.{i}.attention.attn.{projection}" for i in range(12) for projection in ("q", "v")}
    actual = {name for name, module in base.named_modules()
              if isinstance(module, torch.nn.Linear) and name.endswith((".q", ".v"))}
    if actual != expected:
        raise ValueError("Unexpected MPNet LoRA target modules.")
    encoder = get_peft_model(base, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
        r=ADAPTER_SETTINGS["rank"], lora_alpha=ADAPTER_SETTINGS["alpha"],
        lora_dropout=ADAPTER_SETTINGS["dropout"], target_modules=sorted(expected),
        bias="none", init_lora_weights=True))
    for name, parameter in encoder.named_parameters():
        parameter.requires_grad_(".lora_B." in name)
    head = torch.nn.Linear(base.config.hidden_size, len(INTENTS))
    torch.nn.init.normal_(head.weight, std=0.02)
    torch.nn.init.zeros_(head.bias)
    return encoder, head, tokenizer


def _parameters(encoder, head) -> dict:
    return {**{f"encoder.{name}": value for name, value in encoder.named_parameters() if ".lora_B." in name},
            **{f"head.{name}": value for name, value in head.named_parameters()}}


def _a_state(encoder) -> TensorState:
    return {f"encoder.{name}": TensorData.from_torch(value)
            for name, value in encoder.named_parameters() if ".lora_A." in name}


def _assign(parameters: Mapping[str, Any], state: Mapping[str, TensorData]) -> None:
    import torch

    expected = {name: {"shape": list(value.shape), "dtype": "float32"} for name, value in parameters.items()}
    if state_schema(state) != expected or not finite_state(state):
        raise ValueError("State schema mismatch or nonfinite parameters.")
    with torch.no_grad():
        for name, value in parameters.items():
            value.copy_(state[name].to_torch().to(device=value.device))


def save_encoder_state(path: str | Path, state: Mapping[str, TensorData], identity: Mapping[str, Any], **metadata) -> None:
    if not finite_state(state) or state_schema(state) != identity["state_schema"]:
        raise ValueError("Invalid encoder state.")
    write_json(path, {**metadata, "state": state_to_dict(state), "state_hash": hash_state(state), "identity": dict(identity)})


def load_encoder_state(path: str | Path, identity: Mapping[str, Any]) -> TensorState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("identity", {}).get("contract_hash") != identity["contract_hash"]:
        raise ValueError("Encoder state contract mismatch.")
    state = state_from_dict(payload["state"])
    if (not finite_state(state) or state_schema(state) != identity["state_schema"]
            or hash_state(state) != payload.get("state_hash")):
        raise ValueError("Encoder state hash/schema mismatch.")
    return state


def prepare_encoder_adapter(asset_dir: str | Path = "assets/reference-mpnet",
                            adapter_dir: str | Path = "assets/encoder-adapter", seed: int = 42) -> dict:
    """Prepare one shared A and fresh B/head; existing output is never overwritten."""
    from safetensors.torch import save_file

    target = Path(adapter_dir)
    if target.exists():
        if not (target / "identity.json").exists():
            raise ValueError("Adapter directory exists without a complete identity; choose a new path.")
        trainer = EncoderAdapterTrainer(asset_dir, adapter_dir)
        if trainer.identity["initialization_seed"] != seed:
            raise ValueError("Existing adapter seed differs.")
        return trainer.identity
    assets = verify_encoder_assets(asset_dir)
    encoder, head, _ = _build(asset_dir, seed)
    a = _a_state(encoder)
    state = {name: TensorData.from_torch(value) for name, value in _parameters(encoder, head).items()}
    b_count = sum(value.numel() for name, value in encoder.named_parameters() if ".lora_B." in name)
    head_count = sum(value.numel() for value in head.parameters())
    if b_count != 147456 or head_count != 15380:
        raise ValueError("Unexpected realized encoder B/head count.")
    if any(value != 0 for name, tensor in state.items() if ".lora_B." in name for value in tensor.values):
        raise ValueError("Initial B must be zero.")
    identity = {"schema_version": 1, "variant": "mpnet-frozen-a-b-plus-head-v1",
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION, **assets,
                "initialization_seed": seed, "adapter_settings": ADAPTER_SETTINGS,
                "pooling": POOLING, "scoring_rule": SCORING_RULE, "intents": list(INTENTS),
                "common_a_hash": hash_state(a), "a_schema": state_schema(a),
                "state_schema": state_schema(state), "b_schema": state_schema(state),
                "b_parameters": b_count, "head_parameters": head_count,
                "trainable_parameters": b_count + head_count, "wire_dtype": "float32",
                "initial_state_hash": hash_state(state), "raw_update_bytes": state_nbytes(state)}
    identity["contract_hash"] = content_hash(identity)
    target.mkdir(parents=True)
    save_file({name: tensor.to_torch() for name, tensor in a.items()}, str(target / "common_a.safetensors"))
    save_encoder_state(target / "initial_state.json", state, identity, kind="fresh-head-zero-B")
    write_json(target / "identity.json", identity)
    return identity


def masked_mean_normalize(hidden, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


class EncoderAdapterTrainer:
    def __init__(self, asset_dir="assets/reference-mpnet", adapter_dir="assets/encoder-adapter",
                 device="cpu", config: EncoderAdapterConfig | None = None):
        import torch
        from safetensors.torch import load_file

        self.config = config or EncoderAdapterConfig()
        torch.set_num_threads(self.config.cpu_threads)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only validated FP32 CPU/CUDA paths are supported.")
        self.identity = json.loads((Path(adapter_dir) / "identity.json").read_text(encoding="utf-8"))
        if content_hash({k: v for k, v in self.identity.items() if k != "contract_hash"}) != self.identity.get("contract_hash"):
            raise ValueError("Adapter identity hash mismatch.")
        required = {"variant": "mpnet-frozen-a-b-plus-head-v1", "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION, "adapter_settings": ADAPTER_SETTINGS,
                    "pooling": POOLING, "scoring_rule": SCORING_RULE, "intents": list(INTENTS),
                    "b_parameters": 147456, "head_parameters": 15380,
                    "trainable_parameters": 162836, "wire_dtype": "float32", "raw_update_bytes": 651344}
        if any(self.identity.get(key) != value for key, value in required.items()):
            raise ValueError("Unsupported encoder-adapter contract.")
        assets = verify_encoder_assets(asset_dir)
        if assets["asset_hash"] != self.identity["asset_hash"]:
            raise ValueError("Encoder/tokenizer files changed.")
        self.encoder, self.head, self.tokenizer = _build(asset_dir, self.identity["initialization_seed"])
        a = {name: TensorData.from_torch(value) for name, value in load_file(str(Path(adapter_dir) / "common_a.safetensors")).items()}
        if hash_state(a) != self.identity["common_a_hash"] or state_schema(a) != self.identity["a_schema"]:
            raise ValueError("Saved common A identity mismatch.")
        _assign({f"encoder.{name}": value for name, value in self.encoder.named_parameters() if ".lora_A." in name}, a)
        self.encoder.to(self.device)
        self.head.to(self.device)
        if self.config.activation_checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        initial = load_encoder_state(Path(adapter_dir) / "initial_state.json", self.identity)
        if hash_state(initial) != self.identity["initial_state_hash"]:
            raise ValueError("Saved initial state differs from the prepared contract.")
        self.set_state(initial)
        actual = {f"encoder.{name}" for name, parameter in self.encoder.named_parameters() if parameter.requires_grad}
        actual |= {f"head.{name}" for name, parameter in self.head.named_parameters() if parameter.requires_grad}
        if actual != set(_parameters(self.encoder, self.head)):
            raise AssertionError("Only encoder B and head may be trainable.")
        self.encoder.eval()
        self.head.eval()

    def get_state(self) -> TensorState:
        return {name: TensorData.from_torch(value) for name, value in _parameters(self.encoder, self.head).items()}

    def set_state(self, state: Mapping[str, TensorData]) -> None:
        if state_schema(state) != self.identity["state_schema"]:
            raise ValueError("Incoming state is incompatible with encoder/head schema.")
        _assign(_parameters(self.encoder, self.head), state)

    def set_head_state(self, state: Mapping[str, TensorData]) -> None:
        """Explicit warm-start hook; the caller must record its training origin."""
        _assign({f"head.{name}": value for name, value in self.head.named_parameters()}, state)

    def frozen_hashes(self) -> dict[str, str]:
        digest = hashlib.sha256()
        for name, parameter in sorted(self.encoder.named_parameters()):
            if ".lora_" not in name:
                digest.update(name.encode("utf-8"))
                digest.update(str(tuple(parameter.shape)).encode("ascii"))
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return {"base_hash": digest.hexdigest(), "common_a_hash": hash_state(_a_state(self.encoder))}

    def _encode(self, texts):
        batch = self.tokenizer(list(texts), padding=True, truncation=True,
                               max_length=self.config.max_length, return_tensors="pt")
        return {name: value.to(self.device) for name, value in batch.items()}

    def _logits(self, batch):
        hidden = self.encoder(**batch).last_hidden_state
        return self.head(masked_mean_normalize(hidden, batch["attention_mask"]))

    def _truncated_count(self, texts):
        return sum(len(self.tokenizer.encode(text, add_special_tokens=True)) > self.config.max_length
                   for text in texts)

    def _new_optimizer(self):
        import torch

        b = [value for name, value in self.encoder.named_parameters() if ".lora_B." in name]
        head = list(self.head.parameters())
        optimizer = torch.optim.AdamW([{"params": b, "lr": self.config.adapter_lr},
                                       {"params": head, "lr": self.config.head_lr}],
                                      weight_decay=self.config.weight_decay)
        expected = {id(value) for value in _parameters(self.encoder, self.head).values()}
        if {id(value) for group in optimizer.param_groups for value in group["params"]} != expected:
            raise AssertionError("Optimizer includes unexpected parameters.")
        return optimizer

    def train(self, records: Sequence[Mapping[str, Any]], steps: int, seed: int,
              deadline_unix: float | None = None) -> dict:
        import torch

        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1 or not records:
            raise ValueError("Training needs nonempty records and positive steps.")
        if deadline_unix is not None and not math.isfinite(deadline_unix):
            raise ValueError("Deadline must be finite.")
        if any(record.get("label") not in INTENTS or not isinstance(record.get("query"), str)
               or not record["query"].strip() for record in records):
            raise ValueError("Training queries and supported intent labels are required.")
        if hash_state(_a_state(self.encoder)) != self.identity["common_a_hash"]:
            raise ValueError("Common A changed before training.")
        _seed_everything(seed)
        stream = example_indices(len(records), seed)
        optimizer = self._new_optimizer()
        parameters = list(_parameters(self.encoder, self.head).values())
        self.encoder.train()
        self.head.train()
        losses, norms = [], []
        completed = examples = tokens = microbatches = 0
        nonzero_b_gradient = False
        start = time.perf_counter()
        with _MemoryMonitor(self.device) as monitor:
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                step_losses = []
                interrupted = False
                for _ in range(self.config.gradient_accumulation):
                    if deadline_unix is not None and time.time() >= deadline_unix:
                        interrupted = True
                        break
                    chosen = [records[next(stream)] for _ in range(self.config.microbatch_size)]
                    batch = self._encode([row["query"] for row in chosen])
                    labels = torch.tensor([INTENTS.index(row["label"]) for row in chosen], device=self.device)
                    logits = self._logits(batch)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite classification loss.")
                    (loss / self.config.gradient_accumulation).backward()
                    step_losses.append(float(loss.detach()))
                    examples += len(chosen)
                    tokens += int(batch["attention_mask"].sum())
                    microbatches += 1
                if interrupted or (deadline_unix is not None and time.time() >= deadline_unix):
                    optimizer.zero_grad(set_to_none=True)
                    break
                nonzero_b_gradient |= any(parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
                    for name, parameter in self.encoder.named_parameters() if ".lora_B." in name)
                norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip, error_if_nonfinite=True)
                norms.append(float(norm))
                optimizer.step()
                completed += 1
                losses.extend(step_losses)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        self.encoder.eval()
        self.head.eval()
        optimizer.zero_grad(set_to_none=True)
        if hash_state(_a_state(self.encoder)) != self.identity["common_a_hash"]:
            raise AssertionError("Common A changed during training.")
        state = self.get_state()
        if not finite_state(state):
            raise ValueError("Nonfinite trained encoder state.")
        telemetry = {"optimizer_steps": completed, "requested_optimizer_steps": steps,
                     "examples_processed": examples, "tokens_processed": tokens,
                     "microbatches": microbatches, "manifest_examples": len(records),
                     "records_hash": content_hash(list(records)), "seed": seed,
                     "truncated_manifest_queries": self._truncated_count([row["query"] for row in records]),
                     "task_loss": sum(losses) / len(losses) if losses else None,
                     "gradient_norms_before_clip": norms, "nonzero_b_gradient": nonzero_b_gradient,
                     "local_train_seconds": time.perf_counter() - start,
                     "device": str(self.device), "precision": "float32", "config": asdict(self.config),
                     "optimizer_reset": True, "raw_update_bytes": state_nbytes(state), **monitor.metrics()}
        return {"status": "completed" if completed == steps else "partial",
                "state_hash": hash_state(state), "telemetry": telemetry}

    def predict_scores(self, records_or_texts):
        import numpy as np
        import torch

        texts = [row if isinstance(row, str) else row["query"] for row in records_or_texts]
        if not texts:
            return np.empty((0, len(INTENTS)), dtype=np.float32)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Nonempty query text is required.")
        self.encoder.eval()
        self.head.eval()
        scores = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.config.eval_batch_size):
                scores.append(self._logits(self._encode(texts[start:start + self.config.eval_batch_size])).cpu().numpy())
        result = np.concatenate(scores)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite classifier scores.")
        return result

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> dict:
        import torch

        if not records:
            raise ValueError("Evaluation records cannot be empty.")
        truth = [row["label"] for row in records]
        scores = self.predict_scores(records)
        predictions = [INTENTS[int(index)] for index in scores.argmax(axis=1)]
        metrics = classification_metrics(truth, predictions)
        labels = torch.tensor([INTENTS.index(label) for label in truth])
        metrics.update(task_loss=float(torch.nn.functional.cross_entropy(torch.from_numpy(scores), labels)),
                       state_hash=hash_state(self.get_state()), contract_hash=self.identity["contract_hash"],
                       records_hash=content_hash(list(records)),
                       truncated_queries=self._truncated_count([row["query"] for row in records]))
        return metrics


def environment_versions() -> dict:
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "safetensors", "numpy", "psutil")}
