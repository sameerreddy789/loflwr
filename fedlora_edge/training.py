"""Offline SmolLM2 intent completion with a shared frozen A and trainable B.

Heavy ML imports are lazy: the protocol/data tools need only Python's standard
library. Training, Flower and the query UI all call the same LocalTrainer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .data import DOMAINS, INTENTS, canonical_json, content_hash, load_manifest, write_json
from .state import TensorData, TensorState, finite_state, hash_state, state_from_dict, state_schema, state_to_dict, subtract_states

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
ADAPTER_SETTINGS = {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"], "scaling": 2.0}
EXPECTED_B_PARAMETERS = 184320
SCORING_RULE = "sum_completion_logprob_including_eos"
SYSTEM_PROMPT = "Classify the banking request. Reply with only the exact intent label."
TOKENIZATION_VERSION = "chat-prompt-explicit-label-eos-left-truncate-common-budget-v1"


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    microbatch_size: int = 1
    gradient_accumulation: int = 4
    max_length: int = 128
    gradient_clip: float = 1.0
    proximal_mu: float = 0.0
    activation_checkpointing: bool = False
    cpu_threads: int = 4
    eval_batch_size: int = 20

    def __post_init__(self) -> None:
        for name in ("learning_rate", "weight_decay", "gradient_clip", "proximal_mu"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        for name in ("microbatch_size", "gradient_accumulation", "max_length", "cpu_threads", "eval_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.proximal_mu < 0:
            raise ValueError("Invalid optimizer configuration.")
        if self.gradient_clip <= 0:
            raise ValueError("Gradient clipping must be positive.")


def derive_client_seed(run_seed: int, attempt_id: int, client_id: int | str) -> int:
    """Independent streams: omitting a client cannot shift any other stream."""
    return int.from_bytes(hashlib.sha256(canonical_json(["client-stream-v1", run_seed, attempt_id, str(client_id)])).digest()[:8], "little") % (2**31)


def example_indices(size: int, seed: int) -> Iterator[int]:
    if size < 1:
        raise ValueError("Training requires at least one example.")
    rng = random.Random(seed)
    while True:
        order = list(range(size))
        rng.shuffle(order)
        yield from order


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(asset_dir: str | Path = "assets/model") -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    root = Path(asset_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(root), cache_dir=str(root.parent / "huggingface"), allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "*.txt", "*.md", "LICENSE*"])
    files = {str(path.relative_to(root)).replace("\\", "/"): {"sha256": _file_hash(path), "bytes": path.stat().st_size} for path in sorted(root.rglob("*")) if path.is_file() and ".cache" not in path.parts and path.name != "asset_identity.json"}
    identity = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "files": files}
    identity["asset_hash"] = content_hash(identity)
    write_json(root / "asset_identity.json", identity)
    return identity


def verify_model_assets(asset_dir: str | Path) -> dict[str, Any]:
    root = Path(asset_dir)
    identity = json.loads((root / "asset_identity.json").read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in identity.items() if key != "asset_hash"}
    if content_hash(unsigned) != identity["asset_hash"] or identity["model_revision"] != MODEL_REVISION or identity["model_id"] != MODEL_ID:
        raise ValueError("Model asset identity mismatch.")
    for name, metadata in identity["files"].items():
        if _file_hash(root / name) != metadata["sha256"]:
            raise ValueError(f"Model/tokenizer file changed: {name}")
    return identity


def _tokenizer_asset_hash(assets: Mapping[str, Any]) -> str:
    files = {name: value for name, value in assets["files"].items() if any(part in name for part in ("tokenizer", "vocab", "merges", "special_tokens"))}
    return content_hash(files)


def _load_base(asset_dir: str | Path) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(asset_dir), local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define a real EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(str(asset_dir), local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    return model, tokenizer


def _add_adapter(base: Any) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=ADAPTER_SETTINGS["rank"], lora_alpha=ADAPTER_SETTINGS["alpha"], lora_dropout=ADAPTER_SETTINGS["dropout"], target_modules=ADAPTER_SETTINGS["target_modules"], bias="none", init_lora_weights=True))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".lora_B." in name)
    return model


def _tensor_state(model: Any, fragment: str) -> TensorState:
    return {name: TensorData.from_torch(value) for name, value in model.named_parameters() if fragment in name}


def _assign_state(model: Any, state: Mapping[str, TensorData], fragment: str) -> None:
    import torch

    parameters = {name: value for name, value in model.named_parameters() if fragment in name}
    actual_schema = {name: {"shape": list(value.shape), "dtype": "float32"} for name, value in sorted(parameters.items())}
    if state_schema(state) != actual_schema or not finite_state(state):
        raise ValueError(f"Invalid {fragment} tensors: schema or finite-value check failed.")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to_torch().to(parameter.device))


def load_b_state(path: str | Path, expected_identity: Mapping[str, Any] | None = None) -> TensorState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    identity = payload.get("identity")
    if expected_identity is not None:
        if identity is None:
            raise ValueError("Checkpoint has no adapter identity.")
        for key in ("contract_hash", "common_a_hash", "model_revision"):
            if identity.get(key) != expected_identity.get(key):
                raise ValueError(f"Checkpoint {key} mismatch.")
    state = state_from_dict(payload.get("state", payload.get("b_state", payload)))
    if not finite_state(state):
        raise ValueError("Checkpoint contains nonfinite or empty B tensors.")
    if expected_identity is not None and state_schema(state) != expected_identity.get("b_schema"):
        raise ValueError("Checkpoint B schema mismatch.")
    claimed_hash = payload.get("b_hash", payload.get("state_hash"))
    if claimed_hash is not None and hash_state(state) != claimed_hash:
        raise ValueError("Checkpoint B hash mismatch.")
    return state


def save_b_state(path: str | Path, state: Mapping[str, TensorData], identity: Mapping[str, Any], **metadata: Any) -> None:
    from .checkpoint import atomic_json

    if not finite_state(state):
        raise ValueError("Cannot save nonfinite B state.")
    if "b_schema" in identity and state_schema(state) != identity["b_schema"]:
        raise ValueError("Cannot save B with a mismatched schema.")
    atomic_json(path, {**metadata, "identity": dict(identity), "b_hash": hash_state(state), "state": state_to_dict(state)})


def prepare_adapter(asset_dir: str | Path = "assets/model", adapter_dir: str | Path = "assets/adapter", seed: int = 42) -> dict[str, Any]:
    from safetensors.torch import save_file

    target = Path(adapter_dir)
    if (target / "identity.json").exists():
        trainer = LocalTrainer(asset_dir, adapter_dir, device="cpu")
        if trainer.identity["initialization_seed"] != seed:
            raise ValueError("Existing common A was initialized with a different seed. Choose a new adapter directory.")
        return trainer.identity
    assets = verify_model_assets(asset_dir)
    _seed_everything(seed)
    base, tokenizer = _load_base(asset_dir)
    model = _add_adapter(base)
    a_state, b_state = _tensor_state(model, ".lora_A."), _tensor_state(model, ".lora_B.")
    count = sum(len(value.values) for value in b_state.values())
    if count != EXPECTED_B_PARAMETERS:
        raise ValueError(f"Realized B count {count} differs from approved {EXPECTED_B_PARAMETERS}.")
    if any(value != 0 for tensor in b_state.values() for value in tensor.values):
        raise ValueError("Initial B must be exactly zero.")
    identity = {"schema_version": 1, "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "asset_hash": assets["asset_hash"], "tokenizer_hash": _tokenizer_asset_hash(assets), "adapter_settings": ADAPTER_SETTINGS, "common_a_hash": hash_state(a_state), "a_schema": state_schema(a_state), "b_schema": state_schema(b_state), "b_parameters": count, "wire_dtype": "float32", "initialization_seed": seed, "tokenization_version": TOKENIZATION_VERSION, "system_prompt": SYSTEM_PROMPT, "scoring_rule": SCORING_RULE, "intents": list(INTENTS), "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
    identity["contract_hash"] = content_hash(identity)
    target.mkdir(parents=True, exist_ok=True)
    save_file({name: value.detach().cpu().float().contiguous() for name, value in model.named_parameters() if ".lora_A." in name}, str(target / "common_a.safetensors"), metadata={"common_a_hash": identity["common_a_hash"]})
    save_b_state(target / "initial_b.json", b_state, identity, kind="zero-initialized-B")
    write_json(target / "identity.json", identity)
    return identity


class CompletionEncoder:
    """Explicit completion positions preserve EOS even when EOS equals PAD."""

    def __init__(self, tokenizer: Any, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.completions = {label: tokenizer.encode(label, add_special_tokens=False) + [tokenizer.eos_token_id] for label in INTENTS}
        self.completion_budget = max(map(len, self.completions.values()))
        if self.completion_budget >= max_length:
            raise ValueError("Sequence limit cannot preserve every allowed completion and a prompt token.")

    def prompt(self, query: str) -> tuple[list[int], bool]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must contain text.")
        text = self.tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}], tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        limit = self.max_length - self.completion_budget
        return tokens[-limit:], len(tokens) > limit

    def encode(self, query: str, label: str) -> dict[str, Any]:
        if label not in self.completions:
            raise ValueError(f"Intent is outside the fixed twenty-label contract: {label}")
        prompt, truncated = self.prompt(query)
        completion = self.completions[label]
        if not prompt:
            raise ValueError("Prompt has no tokens.")
        return {"input_ids": prompt + completion, "attention_mask": [1] * (len(prompt) + len(completion)), "labels": [-100] * len(prompt) + completion, "prompt_length": len(prompt), "completion_length": len(completion), "truncated_query": truncated}

    def collate(self, examples: Sequence[Mapping[str, Any]], device: Any = "cpu") -> dict[str, Any]:
        import torch

        if not examples:
            raise ValueError("Cannot collate an empty batch.")
        length = max(len(example["input_ids"]) for example in examples)
        # Use explicit lengths/attention, never token-id equality, to mask PAD.
        return {key: torch.tensor([list(example[key]) + [fill] * (length - len(example[key])) for example in examples], dtype=torch.long, device=device) for key, fill in (("input_ids", self.tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100))}


def classification_metrics(truth: Sequence[str], predictions: Sequence[str]) -> dict[str, Any]:
    if len(truth) != len(predictions) or not truth:
        raise ValueError("Metrics require equal nonempty truth/prediction sequences.")
    if any(label not in INTENTS for label in [*truth, *predictions]):
        raise ValueError("Unknown label in metric input.")
    accuracy = sum(actual == predicted for actual, predicted in zip(truth, predictions)) / len(truth)
    per_intent = {}
    for label in INTENTS:
        tp = sum(actual == label and predicted == label for actual, predicted in zip(truth, predictions))
        fp = sum(actual != label and predicted == label for actual, predicted in zip(truth, predictions))
        fn = sum(actual == label and predicted != label for actual, predicted in zip(truth, predictions))
        denominator = 2 * tp + fp + fn
        per_intent[label] = {"f1": 2 * tp / denominator if denominator else 0.0, "support": tp + fn, "accuracy": tp / (tp + fn) if tp + fn else None}
    per_domain = {}
    for client, (domain, labels) in DOMAINS.items():
        relevant = [(actual, predicted) for actual, predicted in zip(truth, predictions) if actual in labels]
        if relevant:
            per_domain[domain] = sum(actual == predicted for actual, predicted in relevant) / len(relevant)
    return {"count": len(truth), "accuracy": accuracy, "macro_f1": sum(value["f1"] for value in per_intent.values()) / len(INTENTS), "per_intent": per_intent, "per_domain_accuracy": per_domain, "worst_domain_accuracy": min(per_domain.values()), "domain_gap": max(per_domain.values()) - min(per_domain.values())}


class _MemoryMonitor:
    def __init__(self, device: Any):
        import psutil
        import torch

        self.process = psutil.Process()
        self.torch = torch
        self.device = device
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

    def _sample(self) -> None:
        while not self.stop.wait(0.02):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> "_MemoryMonitor":
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=1)
        self.peak = max(self.peak, self.process.memory_info().rss)

    def metrics(self) -> dict[str, Any]:
        result = {"peak_rss_bytes": self.peak, "rss_sampling_interval_seconds": 0.02}
        if self.device.type == "cuda":
            result.update({"peak_gpu_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device), "peak_gpu_reserved_bytes": self.torch.cuda.max_memory_reserved(self.device)})
        return result


class LocalTrainer:
    def __init__(self, asset_dir: str | Path = "assets/model", adapter_dir: str | Path = "assets/adapter", device: str = "cpu", config: TrainingConfig | None = None):
        import torch
        from safetensors.torch import load_file

        self.config = config or TrainingConfig()
        torch.set_num_threads(self.config.cpu_threads)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; choose --device cpu explicitly.")
        assets = verify_model_assets(asset_dir)
        self.identity = json.loads((Path(adapter_dir) / "identity.json").read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in self.identity.items() if key != "contract_hash"}
        if content_hash(unsigned) != self.identity["contract_hash"] or assets["asset_hash"] != self.identity["asset_hash"]:
            raise ValueError("Model/adapter identity does not match actual assets.")
        required = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "adapter_settings": ADAPTER_SETTINGS, "tokenization_version": TOKENIZATION_VERSION, "tokenizer_hash": _tokenizer_asset_hash(assets), "system_prompt": SYSTEM_PROMPT, "scoring_rule": SCORING_RULE, "intents": list(INTENTS), "wire_dtype": "float32", "b_parameters": EXPECTED_B_PARAMETERS}
        for key, expected in required.items():
            if self.identity.get(key) != expected:
                raise ValueError(f"Unsupported or mismatched adapter contract field: {key}")
        _seed_everything(self.identity["initialization_seed"])
        base, self.tokenizer = _load_base(asset_dir)
        if self.tokenizer.eos_token_id != self.identity["eos_token_id"] or self.tokenizer.pad_token_id != self.identity["pad_token_id"]:
            raise ValueError("Actual tokenizer EOS/PAD identity mismatch.")
        self.model = _add_adapter(base)
        saved_a = {name: TensorData.from_torch(value) for name, value in load_file(str(Path(adapter_dir) / "common_a.safetensors")).items()}
        if hash_state(saved_a) != self.identity["common_a_hash"]:
            raise ValueError("Actual common A tensors do not match their identity.")
        _assign_state(self.model, saved_a, ".lora_A.")
        self.set_b_state(load_b_state(Path(adapter_dir) / "initial_b.json", self.identity))
        self.model.to(self.device)
        if self.config.activation_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        self.encoder = CompletionEncoder(self.tokenizer, self.config.max_length)
        self.b_parameters = {name: parameter for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        if not self.b_parameters or any(".lora_B." not in name for name in self.b_parameters):
            raise AssertionError("Only B may be trainable.")
        if sum(parameter.numel() for parameter in self.b_parameters.values()) != EXPECTED_B_PARAMETERS:
            raise AssertionError("Unexpected realized B parameter count.")
        self.model.eval()

    def validate_identity(self, other: Mapping[str, Any]) -> None:
        for key in ("contract_hash", "common_a_hash", "model_revision", "tokenizer_hash"):
            if other.get(key) != self.identity.get(key):
                raise ValueError(f"Incompatible {key}.")

    def get_b_state(self) -> TensorState:
        return _tensor_state(self.model, ".lora_B.")

    def set_b_state(self, state: Mapping[str, TensorData]) -> None:
        if state_schema(state) != self.identity["b_schema"]:
            raise ValueError("Incoming B schema does not match the contract.")
        _assign_state(self.model, state, ".lora_B.")

    def frozen_hashes(self) -> dict[str, str]:
        digest = hashlib.sha256()
        for name, parameter in sorted(self.model.named_parameters()):
            if ".lora_" not in name:
                digest.update(name.encode("utf-8"))
                digest.update(str(tuple(parameter.shape)).encode("ascii"))
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return {"base_hash": digest.hexdigest(), "common_a_hash": hash_state(_tensor_state(self.model, ".lora_A."))}

    def _new_optimizer(self) -> Any:
        import torch

        optimizer = torch.optim.AdamW(list(self.b_parameters.values()), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        if {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} != {id(parameter) for parameter in self.b_parameters.values()}:
            raise AssertionError("Optimizer contains parameters other than B.")
        return optimizer

    def train(self, records: Sequence[Mapping[str, Any]], incoming_b: Mapping[str, TensorData], optimizer_steps: int, run_seed: int, attempt_id: int, client_id: int | str, deadline_unix: float | None = None, stop_after_steps: int | None = None, progress_callback: Callable[[int, Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
        if isinstance(optimizer_steps, bool) or not isinstance(optimizer_steps, int) or optimizer_steps < 1:
            raise ValueError("Optimizer-step budget must be positive.")
        if deadline_unix is not None and not math.isfinite(deadline_unix):
            raise ValueError("Deadline must be finite.")
        if stop_after_steps is not None and (isinstance(stop_after_steps, bool) or not isinstance(stop_after_steps, int) or stop_after_steps < 0):
            raise ValueError("Interruption step must be a nonnegative integer.")
        if not records:
            raise ValueError("Training manifest is empty.")
        if hash_state(_tensor_state(self.model, ".lora_A.")) != self.identity["common_a_hash"]:
            raise ValueError("Common A changed before training.")
        self.set_b_state(incoming_b)
        seed = derive_client_seed(run_seed, attempt_id, client_id)
        _seed_everything(seed)
        stream = example_indices(len(records), seed)
        encoded = [self.encoder.encode(record["query"], record["label"]) for record in records]
        batches = ([encoded[next(stream)] for _ in range(self.config.microbatch_size)] for _ in range(optimizer_steps * self.config.gradient_accumulation))
        result = self._train_batches(batches, optimizer_steps, self._new_optimizer(), incoming_b, deadline_unix, stop_after_steps, progress_callback)
        result["telemetry"].update({"client_seed": seed, "run_seed": run_seed, "attempt_id": attempt_id, "client_id": client_id, "manifest_examples": len(records), "records_hash": content_hash(list(records)), "truncated_manifest_queries": sum(example["truncated_query"] for example in encoded)})
        return result

    def train_centralized(self, client_records: Mapping[int, Sequence[Mapping[str, Any]]], incoming_b: Mapping[str, TensorData], budgets: Mapping[int, int] | Sequence[int], run_seed: int, attempt_id: int, deadline_unix: float | None = None, progress_callback: Callable[[int, Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
        """One optimizer-reset block with matched per-client minibatch replay.

        Cycle client IDs, taking one optimizer step per nonempty queue until its
        budget is exhausted. Independent example and dropout RNG streams match
        local clients; the shared B trajectory deliberately differs.
        """
        import torch

        clients = sorted(client_records)
        limits = {client: budgets[client] for client in clients}
        if not clients or any(isinstance(limits[client], bool) or not isinstance(limits[client], int) or limits[client] < 1 or not client_records[client] for client in clients):
            raise ValueError("Centralized replay needs nonempty clients and positive budgets.")
        if deadline_unix is not None and not math.isfinite(deadline_unix):
            raise ValueError("Deadline must be finite.")
        self.set_b_state(incoming_b)
        streams, encoded, rng_states = {}, {}, {}
        for client in clients:
            seed = derive_client_seed(run_seed, attempt_id, client)
            streams[client] = example_indices(len(client_records[client]), seed)
            encoded[client] = [self.encoder.encode(record["query"], record["label"]) for record in client_records[client]]
            _seed_everything(seed)
            rng_states[client] = (torch.get_rng_state(), torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None)
        work = {client: 0 for client in clients}
        replay = hashlib.sha256()
        current_client: int | None = None

        def batches() -> Iterator[list[Mapping[str, Any]]]:
            nonlocal current_client
            for step in range(max(limits.values())):
                for client in clients:
                    if step >= limits[client]:
                        continue
                    current_client = client
                    cpu_state, cuda_state = rng_states[client]
                    torch.set_rng_state(cpu_state)
                    if cuda_state is not None:
                        torch.cuda.set_rng_state(cuda_state, self.device)
                    for _ in range(self.config.gradient_accumulation):
                        indices = [next(streams[client]) for _ in range(self.config.microbatch_size)]
                        replay.update(canonical_json([client, indices]))
                        yield [encoded[client][index] for index in indices]
                    rng_states[client] = (torch.get_rng_state(), torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None)

        def report_progress(step: int, telemetry: Mapping[str, Any]) -> None:
            if current_client is None:
                raise AssertionError("Centralized step has no source client.")
            work[current_client] += 1
            if progress_callback is not None:
                progress_callback(step, {**telemetry, "client_id": current_client})

        iterator = batches()
        result = self._train_batches(iterator, sum(limits.values()), self._new_optimizer(), incoming_b, deadline_unix, progress_callback=report_progress)
        # Resume the exhausted generator once so its last completed-step record
        # is accounted for; no additional minibatch may be consumed.
        if result["status"] == "completed":
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise AssertionError("Centralized replay generated excess work.")
        result["telemetry"].update({"run_seed": run_seed, "attempt_id": attempt_id, "client_id": "centralized", "replay_rule": "round-robin-one-step-per-client-v1", "completed_steps_by_client": work, "replay_hash": replay.hexdigest(), "client_budgets": limits, "records_hash_by_client": {client: content_hash(list(records)) for client, records in client_records.items()}})
        return result

    def _train_batches(self, batches: Iterator[list[Mapping[str, Any]]], optimizer_steps: int, optimizer: Any, incoming_b: Mapping[str, TensorData], deadline_unix: float | None, stop_after_steps: int | None = None, progress_callback: Callable[[int, Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
        import torch

        start_b = {name: parameter.detach().clone() for name, parameter in self.b_parameters.items()} if self.config.proximal_mu else {}
        started = time.perf_counter()
        completed = microbatches = examples = tokens = supervised = truncated_examples = 0
        task_loss_sum = 0.0
        objective_loss_sum = 0.0
        gradient_norms = []
        status = "completed"
        self.model.train()
        with _MemoryMonitor(self.device) as memory:
            try:
                for _ in range(optimizer_steps):
                    optimizer.zero_grad(set_to_none=True)
                    if stop_after_steps is not None and completed >= stop_after_steps:
                        status = "interrupted"
                        break
                    for _ in range(self.config.gradient_accumulation):
                        if deadline_unix is not None and time.time() >= deadline_unix:
                            status = "partial"
                            break
                        items = next(batches)
                        batch = self.encoder.collate(items, self.device)
                        output = self.model(**batch)
                        loss = output.loss
                        loss_value = float(loss.detach())
                        if not math.isfinite(loss_value):
                            raise FloatingPointError("Nonfinite task loss.")
                        objective = loss
                        if start_b:
                            proximal = sum((parameter - start_b[name]).square().sum() for name, parameter in self.b_parameters.items())
                            objective = objective + self.config.proximal_mu * 0.5 * proximal
                        (objective / self.config.gradient_accumulation).backward()
                        # Collation only adds masked padding; count the CPU rows
                        # without synchronizing GPU reductions after backward.
                        batch_supervised = sum(label != -100 for item in items for label in item["labels"][1:])
                        task_loss_sum += loss_value * batch_supervised
                        objective_loss_sum += (loss_value if objective is loss else float(objective.detach())) * batch_supervised
                        microbatches += 1
                        examples += len(items)
                        truncated_examples += sum(bool(item["truncated_query"]) for item in items)
                        tokens += sum(sum(item["attention_mask"]) for item in items)
                        supervised += batch_supervised
                    if status != "completed":
                        optimizer.zero_grad(set_to_none=True)
                        break
                    grads = [parameter.grad for parameter in self.b_parameters.values()]
                    if any(gradient is None for gradient in grads):
                        raise RuntimeError("Missing B gradients; check frozen-base/checkpointing path.")
                    # The strict total-norm check rejects every NaN/Inf gradient
                    # before clipping or stepping, without one host sync per B.
                    norm = torch.nn.utils.clip_grad_norm_(list(self.b_parameters.values()), self.config.gradient_clip, error_if_nonfinite=True)
                    gradient_norms.append(float(norm))
                    optimizer.step()
                    completed += 1
                    if progress_callback is not None:
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        progress_callback(completed, {"optimizer_steps": completed, "microbatches": microbatches, "examples_processed": examples, "tokens_processed": tokens, "local_train_seconds": time.perf_counter() - started})
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            finally:
                optimizer.zero_grad(set_to_none=True)
                self.model.eval()
        local = self.get_b_state()
        if not finite_state(local):
            raise FloatingPointError("Nonfinite local B after optimization.")
        if hash_state(_tensor_state(self.model, ".lora_A.")) != self.identity["common_a_hash"]:
            raise AssertionError("Frozen A changed during optimization.")
        telemetry = {"optimizer_steps": completed, "completed_optimizer_steps": completed, "requested_optimizer_steps": optimizer_steps, "microbatches": microbatches, "examples_processed": examples, "tokens_processed": tokens, "supervised_tokens": supervised, "truncated_examples_processed": truncated_examples, "supervision_truncated_examples": 0, "task_loss": task_loss_sum / supervised if supervised else None, "penalized_objective_loss": objective_loss_sum / supervised if supervised else None, "local_train_seconds": time.perf_counter() - started, "gradient_norms_before_clip": gradient_norms, "nonzero_b_gradient": any(norm > 0 for norm in gradient_norms), "device": str(self.device), "precision": "float32", "config": asdict(self.config), **memory.metrics()}
        return {"status": status, "b_state": local, "delta": subtract_states(local, incoming_b), "b_hash": hash_state(local), "telemetry": telemetry}

    def _scores_for_examples(self, examples: Sequence[Mapping[str, Any]]) -> list[tuple[float, int]]:
        import torch
        import torch.nn.functional as functional

        self.model.eval()
        results = []
        with torch.inference_mode():
            for offset in range(0, len(examples), self.config.eval_batch_size):
                batch = self.encoder.collate(examples[offset:offset + self.config.eval_batch_size], self.device)
                logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :].float()
                labels = batch["labels"][:, 1:]
                mask = labels != -100
                # Prompt/padding rows have exactly zero loss. Select supervised
                # rows before CE to avoid a full prompt-sized logits copy and
                # log-softmax, then preserve the original per-candidate sum.
                losses = torch.zeros_like(labels, dtype=logits.dtype)
                losses[mask] = functional.cross_entropy(logits[mask], labels[mask], reduction="none")
                scores = -(losses * mask).sum(dim=1)
                counts = mask.sum(dim=1)
                results.extend((float(score), int(count)) for score, count in zip(scores.cpu(), counts.cpu()))
        return results

    def score(self, query: str) -> dict[str, Any]:
        encoded = [self.encoder.encode(query, label) for label in INTENTS]
        scores = dict(zip(INTENTS, (value for value, count in self._scores_for_examples(encoded))))
        return {"intent": max(INTENTS, key=lambda label: scores[label]), "scores": scores, "scoring_rule": SCORING_RULE, "truncated_query": encoded[0]["truncated_query"], "b_hash": hash_state(self.get_b_state()), "contract_hash": self.identity["contract_hash"]}

    def evaluate(self, records: Sequence[Mapping[str, Any]], b_state: Mapping[str, TensorData] | None = None) -> dict[str, Any]:
        if b_state is not None:
            self.set_b_state(b_state)
        if not records:
            raise ValueError("Evaluation manifest is empty.")
        started = time.perf_counter()
        predictions, truth, details = [], [], []
        negative_logprob, supervised = 0.0, 0
        with _MemoryMonitor(self.device) as memory:
            for record in records:
                encoded = [self.encoder.encode(record["query"], label) for label in INTENTS]
                scored = self._scores_for_examples(encoded)
                scores = [item[0] for item in scored]
                predicted = INTENTS[max(range(len(INTENTS)), key=lambda index: scores[index])]
                actual_index = INTENTS.index(record["label"])
                negative_logprob -= scores[actual_index]
                supervised += scored[actual_index][1]
                truth.append(record["label"])
                predictions.append(predicted)
                details.append({"source_id": record.get("source_id"), "label": record["label"], "prediction": predicted, "true_completion_logprob": scores[actual_index], "truncated_query": encoded[0]["truncated_query"]})
        return {**classification_metrics(truth, predictions), "task_loss": negative_logprob / supervised, "supervised_tokens": supervised, "evaluation_seconds": time.perf_counter() - started, "scoring_rule": SCORING_RULE, "truncated_queries": sum(item["truncated_query"] for item in details), "b_hash": hash_state(self.get_b_state()), "contract_hash": self.identity["contract_hash"], "records_hash": content_hash(list(records)), "predictions": details, **memory.metrics()}


def environment_versions() -> dict[str, Any]:
    import platform
    import torch

    return {"python": platform.python_version(), "platform": platform.platform(), "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "huggingface-hub", "safetensors", "psutil")}, "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def smoke(trainer: LocalTrainer, records: Sequence[Mapping[str, Any]], steps: int = 2) -> dict[str, Any]:
    import torch

    initial = trainer.get_b_state()
    frozen_before = trainer.frozen_hashes()
    examples = [trainer.encoder.encode(record["query"], record["label"]) for record in records]
    before = trainer._scores_for_examples(examples)
    result = trainer.train(records, initial, optimizer_steps=steps, run_seed=42, attempt_id=0, client_id=0)
    after = trainer._scores_for_examples(examples)
    frozen_after = trainer.frozen_hashes()
    longest_label = max(INTENTS, key=lambda label: len(trainer.encoder.completions[label]))
    probes = [trainer.encoder.encode("Where is my card?", "card_arrival"), trainer.encoder.encode("Please help with this banking request. " * 100, longest_label)]
    batch = trainer.encoder.collate(probes, trainer.device)
    padding_mask_correct = bool(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))
    padding_positions = int((batch["attention_mask"] == 0).sum())
    all_candidates = [trainer.encoder.encode(records[0]["query"], label) for label in INTENTS]
    real_eos_preserved = all(example["labels"][-1] == trainer.tokenizer.eos_token_id and sum(token != -100 for token in example["labels"]) == example["completion_length"] for example in [*probes, *all_candidates])
    common_candidate_prompt = all(example["input_ids"][:example["prompt_length"]] == all_candidates[0]["input_ids"][:all_candidates[0]["prompt_length"]] for example in all_candidates)
    b_changed = hash_state(initial) != result["b_hash"]
    # Check the scorer itself under different batch padding shapes.
    old_config = trainer.config
    batched = trainer._scores_for_examples(probes)
    try:
        trainer.config = TrainingConfig(**{**asdict(old_config), "eval_batch_size": 1})
        individual = trainer._scores_for_examples(probes)
    finally:
        trainer.config = old_config
    max_batch_score_difference = max(abs(left[0] - right[0]) for left, right in zip(batched, individual))
    passed = b_changed and frozen_before == frozen_after and padding_mask_correct and padding_positions > 0 and real_eos_preserved and common_candidate_prompt and probes[1]["truncated_query"] and result["telemetry"]["nonzero_b_gradient"] and max_batch_score_difference < 1e-3
    if not passed:
        raise AssertionError(f"Model smoke invariants failed: B={b_changed}, frozen={frozen_before == frozen_after}, padding={padding_mask_correct}, EOS={real_eos_preserved}, score_difference={max_batch_score_difference}")
    return {"kind": "real_model_optimization_smoke", "passed": passed, "identity": trainer.identity, "environment": environment_versions(), "b_changed": b_changed, "frozen_before": frozen_before, "frozen_after": frozen_after, "padding_mask_correct": padding_mask_correct, "checked_padding_positions": padding_positions, "real_eos_preserved": real_eos_preserved, "pad_equals_eos": trainer.tokenizer.pad_token_id == trainer.tokenizer.eos_token_id, "common_candidate_prompt": common_candidate_prompt, "long_query_truncation_preserves_completion": probes[1]["truncated_query"] and probes[1]["labels"][-1] == trainer.tokenizer.eos_token_id, "max_batch_score_difference": max_batch_score_difference, "before_task_loss": -sum(score for score, count in before) / sum(count for score, count in before), "after_task_loss": -sum(score for score, count in after) / sum(count for score, count in after), "telemetry": result["telemetry"], "b_hash": result["b_hash"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "prepare", "smoke", "train", "evaluate", "query"))
    parser.add_argument("--asset-dir", default="assets/model")
    parser.add_argument("--adapter-dir", default="assets/adapter")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--manifest", default="assets/banking77/manifests/imbalanced/client_0.json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--client", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--query")
    parser.add_argument("--run-id")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--activation-checkpointing", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        result = download_model(args.asset_dir)
    elif args.command == "prepare":
        result = prepare_adapter(args.asset_dir, args.adapter_dir, args.seed)
    else:
        run_id = args.run_id or f"{args.command}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        trainer = LocalTrainer(args.asset_dir, args.adapter_dir, args.device, TrainingConfig(cpu_threads=args.cpu_threads, eval_batch_size=args.eval_batch_size, activation_checkpointing=args.activation_checkpointing))
        if args.checkpoint:
            trainer.set_b_state(load_b_state(args.checkpoint, trainer.identity))
        if args.command == "query":
            if not args.query:
                parser.error("query requires --query")
            result = trainer.score(args.query)
        else:
            records = load_manifest(args.manifest)
            if args.limit:
                records = records[:args.limit]
            if args.command == "smoke":
                result = smoke(trainer, records[:4], args.steps)
            elif args.command == "evaluate":
                result = trainer.evaluate(records)
            else:
                trained = trainer.train(records, trainer.get_b_state(), args.steps, args.seed, args.attempt, args.client)
                output = args.output or f"runs/local-seed{args.seed}/checkpoint.json"
                save_b_state(output, trained["b_state"], trainer.identity, run_id=run_id, manifest_hash=json.loads(Path(args.manifest).read_text(encoding="utf-8"))["manifest_hash"], telemetry=trained["telemetry"], status=trained["status"])
                result = {key: value for key, value in trained.items() if key not in ("b_state", "delta")}
                result["checkpoint"] = output
        result["identity"] = trainer.identity
        result["run_id"] = run_id
        result["configuration"] = asdict(trainer.config)
        if args.command != "query":
            result["manifest_path"] = args.manifest
            result["manifest_hash"] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["manifest_hash"]
    if args.output and args.command != "train":
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
