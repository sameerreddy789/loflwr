"""Offline inference for the frozen-MPNet federated classification head."""
from __future__ import annotations
from pathlib import Path
import threading
import time

from .checkpoint import CheckpointStore
from .data import INTENTS, content_hash
from .encoder_adapter import MODEL_REVISION, verify_encoder_assets
from .encoder_head import HeadTrainer
from .responses import response_for


def load_head_checkpoint(directory):
    directory = Path(directory)
    if not (directory / "checkpoint.json").is_file():
        raise FileNotFoundError("Selected head checkpoint is missing. Choose a completed encoder run.")
    saved = CheckpointStore(directory).load()
    identity = saved["identity"]
    if (identity.get("kind") != "frozen_mpnet_linear_head"
            or identity.get("model_revision") != MODEL_REVISION
            or identity.get("labels") != list(INTENTS)
            or identity.get("feature_dim") != 768
            or content_hash({k: v for k, v in identity.items() if k != "contract_hash"}) != identity.get("contract_hash")):
        raise ValueError("Checkpoint does not match the pinned MPNet/20-intent head contract.")
    return saved


class EncoderPredictor:
    def __init__(self, asset_dir, checkpoint_dir, *, device="cpu", cpu_threads=2):
        import torch
        from transformers import AutoModel, AutoTokenizer
        saved = load_head_checkpoint(checkpoint_dir)
        self.asset_identity = verify_encoder_assets(asset_dir)
        torch.set_num_threads(cpu_threads)
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(str(asset_dir), local_files_only=True)
        self.tokenizer.padding_side = "right"
        self.encoder = AutoModel.from_pretrained(str(asset_dir), local_files_only=True,
                                                dtype=torch.float32).to(device).eval()
        self.encoder.requires_grad_(False)
        self.head = HeadTrainer(768, len(INTENTS))
        self.head.set_state(saved["state"])
        self.saved = saved
        self.lock = threading.Lock()

    def scores(self, queries, batch_size=16):
        import numpy as np
        if (not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1
                or not queries or any(not isinstance(q, str) or not q.strip() for q in queries)):
            raise ValueError("Provide nonempty queries and a positive batch size.")
        vectors, truncated = [], 0
        with self.lock, self.torch.inference_mode():
            for start in range(0, len(queries), batch_size):
                batch = queries[start:start + batch_size]
                lengths = self.tokenizer(batch, padding=False, truncation=False, return_length=True)["length"]
                truncated += sum(n > 384 for n in lengths)
                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=384,
                                         return_tensors="pt").to(self.device)
                hidden = self.encoder(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy())
            scores = self.head.predict_scores(np.concatenate(vectors))
        return scores, truncated

    def query(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a banking query before classifying it.")
        if len(text) > 4000:
            raise ValueError("Keep the query under 4,000 characters.")
        started = time.perf_counter()
        scores, truncated = self.scores([text])
        order = scores[0].argsort()[::-1][:3]
        intent = INTENTS[int(order[0])]
        return {"intent": intent, "response": response_for(intent),
                "candidates": [{"Intent": INTENTS[int(i)], "Uncalibrated logit": float(scores[0, i])} for i in order],
                "seconds": time.perf_counter() - started, "truncated": bool(truncated),
                "head_hash": self.saved["b_hash"], "run_id": self.saved["run_id"]}
