"""CPU classification-head optimization on locally cached frozen encoder features.

This is a separate architecture variant, not LoRA or the historical B-only
trainer. Torch and psutil imports are lazy. No model assets, network services,
dataset files, or final-test labels are read by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import threading
import time
from typing import Any, Callable, Mapping

from .state import TensorData, TensorState, finite_state, hash_state, state_schema


@dataclass(frozen=True)
class HeadConfig:
    learning_rate: float = 0.1
    batch_size: int = 32
    weight_decay: float = 0.0
    optimizer: str = "sgd"
    gradient_clip: float = 10.0
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        for name in ("learning_rate", "weight_decay", "gradient_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rate/clip must be positive and weight decay nonnegative.")
        for name in ("batch_size", "cpu_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.optimizer not in {"sgd", "adamw"}:
            raise ValueError("Head optimizer must be sgd or adamw.")


def _positive_integer(value: Any, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _seed(value: int) -> None:
    _positive_integer(value, "seed", 0)
    if value >= 2**63:
        raise ValueError("Seed must be less than 2**63.")


class _RssMonitor:
    """Sample the whole current process; this does not measure the whole laptop."""

    interval = 0.02

    def __init__(self) -> None:
        try:
            import psutil
        except ImportError:
            self.process = None
        else:
            self.process = psutil.Process()
        self.peak = self.process.memory_info().rss if self.process else None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def sample(self) -> None:
        if self.process:
            self.peak = max(self.peak or 0, self.process.memory_info().rss)

    def _sample(self) -> None:
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self) -> "_RssMonitor":
        if self.process:
            self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop.set()
        if self.process:
            self.thread.join(timeout=1)
        self.sample()


class HeadTrainer:
    """A fresh affine head; callers supply only their own frozen features.

    Every train call resets optimizer state and derives minibatch permutations
    from a private generator. Call set_state before a federated attempt. Failed
    or expired calls restore incoming weights. All num_classes logits enter CE,
    even when the local dataset contains only a subset of the label universe.
    Weight decay applies to weights only (coupled SGD or decoupled AdamW).
    """

    def __init__(self, input_dim: int = 768, num_classes: int = 20,
                 seed: int = 42, config: HeadConfig | None = None):
        _positive_integer(input_dim, "input_dim")
        _positive_integer(num_classes, "num_classes", 2)
        _seed(seed)
        import torch

        self.torch = torch
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = config or HeadConfig()
        if not isinstance(self.config, HeadConfig):
            raise ValueError("config must be a HeadConfig.")
        torch.set_num_threads(self.config.cpu_threads)
        # Constructor initialization must not consume another client's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            self.head = torch.nn.Linear(input_dim, num_classes, device="cpu", dtype=torch.float32)
        self._schema = {
            "head.weight": {"shape": [num_classes, input_dim], "dtype": "float32"},
            "head.bias": {"shape": [num_classes], "dtype": "float32"},
        }

    @property
    def parameter_count(self) -> int:
        return (self.input_dim + 1) * self.num_classes

    def get_state(self) -> TensorState:
        return {"head.weight": TensorData.from_torch(self.head.weight),
                "head.bias": TensorData.from_torch(self.head.bias)}

    def set_state(self, state: Mapping[str, TensorData]) -> None:
        if (not isinstance(state, Mapping) or set(state) != set(self._schema)
                or any(not isinstance(tensor, TensorData) for tensor in state.values())
                or state_schema(state) != self._schema or not finite_state(state)):
            raise ValueError("Head state must have the exact finite FP32 weight/bias schema.")
        # Validate everything before changing either parameter.
        with self.torch.no_grad():
            self.head.weight.copy_(state["head.weight"].to_torch())
            self.head.bias.copy_(state["head.bias"].to_torch())

    def _features(self, features: Any, allow_empty: bool = False) -> Any:
        torch = self.torch
        value = torch.as_tensor(features)
        if value.is_complex() or value.dtype == torch.bool:
            raise ValueError("Features must be real numeric values.")
        value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if (value.ndim != 2 or value.shape[1] != self.input_dim
                or (not allow_empty and value.shape[0] == 0)
                or not torch.isfinite(value).all().item()):
            raise ValueError("Features must be finite [examples, input_dim] values.")
        return value

    def _labels(self, labels: Any, size: int) -> Any:
        torch = self.torch
        value = torch.as_tensor(labels)
        if (value.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
                or value.ndim != 1 or len(value) != size):
            raise ValueError("Labels must be aligned one-dimensional integer class indices.")
        value = value.detach().to(device="cpu", dtype=torch.long).contiguous()
        if value.min().item() < 0 or value.max().item() >= self.num_classes:
            raise ValueError("A label is outside the configured class universe.")
        return value

    def _optimizer(self) -> Any:
        groups = [{"params": [self.head.weight], "weight_decay": self.config.weight_decay},
                  {"params": [self.head.bias], "weight_decay": 0.0}]
        kind = self.torch.optim.SGD if self.config.optimizer == "sgd" else self.torch.optim.AdamW
        return kind(groups, lr=self.config.learning_rate)

    def _loss(self, features: Any, labels: Any) -> float:
        with self.torch.no_grad():
            value = self.torch.nn.functional.cross_entropy(self.head(features), labels).item()
        if not math.isfinite(value):
            raise ValueError("Nonfinite head loss.")
        return value

    def train(self, features: Any, labels: Any, steps: int, seed: int, *,
              deadline_unix: float | None = None,
              progress_callback: Callable[[int, Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
        """Run exactly steps minibatches, or fail and restore the incoming state.

        Batch size >= local row count means one full local batch per step.
        Smaller batches traverse independently shuffled epochs and retain the
        shorter tail batch, so examples_processed is measured rather than assumed.
        The optional deadline is checked before work and before committing each
        optimizer step. A process/network timeout remains the caller's concern.
        """
        _positive_integer(steps, "steps")
        _seed(seed)
        if deadline_unix is not None and (isinstance(deadline_unix, bool)
                or not isinstance(deadline_unix, (int, float)) or not math.isfinite(deadline_unix)):
            raise ValueError("Deadline must be a finite Unix timestamp.")

        def check_deadline() -> None:
            if deadline_unix is not None and time.time() >= deadline_unix:
                raise TimeoutError("Head-training deadline elapsed; no update committed.")

        check_deadline()
        started = time.perf_counter()
        torch = self.torch
        torch.set_num_threads(self.config.cpu_threads)
        before = self.get_state()
        before_hash = hash_state(before)
        examples = 0
        observed = set()
        gradient_norms = []
        weighted_loss = 0.0
        with _RssMonitor() as memory:
            x = self._features(features)
            y = self._labels(labels, len(x))
            generator = torch.Generator(device="cpu").manual_seed(seed)
            optimizer = self._optimizer()
            pending = torch.empty(0, dtype=torch.long)
            before_loss = self._loss(x, y)
            optimization_started = time.perf_counter()
            self.head.train()
            try:
                for step in range(1, steps + 1):
                    check_deadline()
                    if self.config.batch_size >= len(x):
                        indices = torch.arange(len(x))
                    else:
                        if len(pending) == 0:
                            pending = torch.randperm(len(x), generator=generator)
                        indices, pending = pending[:self.config.batch_size], pending[self.config.batch_size:]
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(self.head(x[indices]), y[indices])
                    if not torch.isfinite(loss).item():
                        raise ValueError("Nonfinite classification-head task loss.")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.config.gradient_clip,
                                                         error_if_nonfinite=True)
                    check_deadline()
                    optimizer.step()
                    if any(not torch.isfinite(parameter).all().item() for parameter in self.head.parameters()):
                        raise ValueError("Nonfinite classification-head update.")
                    gradient_norms.append(float(norm))
                    examples += len(indices)
                    observed.update(indices.tolist())
                    weighted_loss += loss.item() * len(indices)
                    if progress_callback is not None:
                        progress_callback(step, {"optimizer_steps": step, "examples_processed": examples,
                                                 "local_train_seconds": time.perf_counter() - started})
                check_deadline()
                optimization_seconds = time.perf_counter() - optimization_started
                after_loss = self._loss(x, y)
                check_deadline()
                after = self.get_state()
            except BaseException:
                self.set_state(before)
                self.head.zero_grad(set_to_none=True)
                raise
            self.head.zero_grad(set_to_none=True)
        norms = x.norm(dim=1)
        return {"kind": "frozen_encoder_head_training", "device": "cpu", "precision": "float32",
                "config": asdict(self.config), "optimizer_state_reset": True, "seed": seed,
                "manifest_examples": len(x), "input_dim": self.input_dim, "num_classes": self.num_classes,
                "parameter_count": self.parameter_count, "raw_parameter_bytes": 4 * self.parameter_count,
                "requested_optimizer_steps": steps, "completed_optimizer_steps": steps,
                "optimizer_steps": steps, "microbatches": steps, "examples_processed": examples,
                "unique_examples_processed": len(observed), "task_loss": weighted_loss / examples,
                "training_loss_before": before_loss, "training_loss_after": after_loss,
                "gradient_norms_before_clip": gradient_norms,
                "nonzero_head_gradient": any(value > 0 for value in gradient_norms),
                "state_hash_before": before_hash, "state_hash_after": hash_state(after),
                "local_train_seconds": time.perf_counter() - started,
                "optimization_seconds": optimization_seconds,
                "peak_rss_bytes": memory.peak,
                "rss_sampling_interval_seconds": memory.interval if memory.process else None,
                "memory_scope": "sampled current process RSS" if memory.process else "psutil unavailable",
                "feature_norm_min": float(norms.min()), "feature_norm_max": float(norms.max())}

    def predict_scores(self, features: Any) -> Any:
        """Return all class logits; no confidence calibration or threshold is implied."""
        torch = self.torch
        x = self._features(features, allow_empty=True)
        self.head.eval()
        with torch.inference_mode():
            scores = self.head(x)
            if not torch.isfinite(scores).all().item():
                raise ValueError("Nonfinite classification-head predictions.")
            return scores.numpy()
