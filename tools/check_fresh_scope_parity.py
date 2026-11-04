"""Offline, authored unlabeled probes for the separate paired inference path.

These are functional inputs, never independent evaluation examples. This tool
does not access a query collection or create an evaluation/exposure claim.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import evaluate_fresh_scope as scope

PROBES = [
    "Functional probe: when does my replacement card arrive?",
    "Functional probe: can you explain a payment that is pending?",
    "Functional probe: I need help understanding a transfer charge.",
    "Functional probe: why is a cash withdrawal still pending?",
    "Functional probe: how can I change my personal details?",
    "Functional probe: what are the steps for identity verification?",
    "Functional probe: how can I activate this new card?",
    "Functional probe: my card has an upcoming expiry date.",
    "Functional probe: a transfer has not reached the other person.",
    "Functional probe: the amount of cash received looks wrong.",
    "Functional probe: can you tell me the weather?",
    "Functional probe: please help me book a flight.",
    "Functional probe: help.",
    "Functional probe: payment or transfer? I am unsure.",
    "Functional probe: CARD PAYMENT 123?",
    "Functional probe: first line\nsecond line with punctuation: 'help'.",
    "Functional probe: café payment receipt.",
    "Functional truncation probe: " + "This sentence repeats for a token-length check. " * 90,
]


def check(output):
    output = scope.base.private(output)
    if output.exists():
        raise FileExistsError("Choose a new parity report path")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import numpy as np
    from fedlora_edge.encoder_inference import EncoderPredictor
    from fedlora_edge.encoder_local_data import encode_batches

    started = time.perf_counter()
    contract = scope.model_contract()
    scope_model = scope.base.read(ROOT / contract["scope_file"])
    head = contract["head"]
    predictor = EncoderPredictor(ROOT / head["asset_dir"], ROOT / head["checkpoint_dir"], cpu_threads=2)
    baseline, baseline_truncated = predictor.scores(PROBES, batch_size=16)
    forwards = []
    hook = predictor.encoder.register_forward_hook(lambda _m, _args, _out: forwards.append(1))
    try:
        logits, scores, truncated = scope.paired_scores(predictor, PROBES, scope_model)
    finally:
        hook.remove()
    # Recompute reference features with the frozen recipe solely for this
    # functional parity check. Actual evaluation uses only the one paired pass.
    features, reference_truncated = encode_batches(predictor.encoder, predictor.tokenizer, PROBES, batch_size=16)
    reference_scope = features @ np.array(scope_model["coef"]) + scope_model["intercept"]
    records = [{"query_id": f"probe-{i}", "disposition": "secondary"} for i in range(len(PROBES))]
    old_decisions = scope.base.predictions_from_scores(records, baseline.tolist())
    new_decisions = scope.predictions_from_scores(records, logits.tolist(), scores.tolist())
    checks = {
        "head_logits_bitwise_equal": bool(np.array_equal(baseline, logits)),
        "scope_scores_bitwise_equal": bool(np.array_equal(reference_scope, scores)),
        "head_decisions_equal": old_decisions == [{k: p[k] for k in old_decisions[0]} for p in new_decisions],
        "scope_decisions_equal": [p["scope_passed"] for p in new_decisions] == (reference_scope >= scope.SCOPE_THRESHOLD).tolist(),
        "truncation_count_equal": baseline_truncated == reference_truncated == truncated == 1,
        "one_shared_encoder_pass": len(forwards) == math.ceil(len(PROBES) / 16),
        "head_identity_unchanged": predictor.saved["b_hash"] == head["head_hash"],
        "frozen_contract_unchanged": scope.model_contract() == contract,
    }
    result = {"kind": "authored_unlabeled_scope_parity_v1", "status": "passed" if all(checks.values()) else "failed",
        "probes": len(PROBES), "probe_hash": scope.base.digest(PROBES), "checks": checks,
        "scope": "Functional parity only; no labels, no independent accuracy and no old test-query inference",
        "encoder_forward_calls_in_paired_path": len(forwards), "batch_size": 16,
        "head_max_absolute_difference": float(np.abs(baseline - logits).max()),
        "scope_max_absolute_difference": float(np.abs(reference_scope - scores).max()),
        "truncated_probes": int(truncated), "head_hash": head["head_hash"],
        "scope_model_sha256": contract["scope_sha256"], "scope_lock": contract["scope_lock"],
        "requested_torch_threads": 2, "effective_torch_threads": predictor.torch.get_num_threads(),
        "runtime": scope.runtime(), "seconds": time.perf_counter() - started,
        "source_hashes": {name: scope.base.sha(ROOT / name) for name in
            ("tools/check_fresh_scope_parity.py", "tools/evaluate_fresh_scope.py", "fedlora_edge/encoder_inference.py",
             "fedlora_edge/encoder_local_data.py", "fedlora_edge/encoder_head.py")}}
    output.parent.mkdir(parents=True, exist_ok=True)
    scope.base.write_new(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    result = check(parser.parse_args().output)
    print({"status": result["status"], "probes": result["probes"], "checks": result["checks"]})
    raise SystemExit(0 if result["status"] == "passed" else 1)
