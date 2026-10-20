# Implementation status

Last updated: **20 October 2026**. This describes repository delivery, not planned features as if they already existed.

## Included in this delivery

- Root README and the research overview, landscape, system design, evaluation, roadmap, and source register.
- Four project-specific skills for frozen-A training, federated rounds, BANKING77 experiments, and laptop deployment.
- `fedlora_edge/state.py`: initial dependency-free tensor representation, canonical FP32 state handling, schema/hash/size/norm helpers, finite checks, and subtraction.
- Standard-library tensor-state contract tests and documentation/source-review utilities.
- Initial package metadata and ignored locations for assets, runs, caches, and private node keys.
- Legacy Qwen inspection metadata; the original `IIT-B` wrapper and weight file are absent.

## Outstanding implementation

| Component | Status |
| --- | --- |
| Dataset preparation and immutable manifests | Not implemented |
| Model/tokenizer caching and common-A initialization | Not implemented |
| CPU/GPU B-only training and intent scoring | Not implemented |
| Full request/reply protocol, strategies, and atomic checkpoints | Not implemented |
| Flower ClientApp/ServerApp and five-client runs | Not implemented |
| Tested CPU/CUDA dependency locks | Not produced |
| Comparative experiments and omission study | Not run |
| Dashboard and banking-query application | Not implemented |
| Three-laptop offline deployment and actual dropout/rejoin | Not tested |

## Evidence available

The source register distinguishes reviewed primary-page metadata/abstracts, framework documentation, publisher model/license metadata, arithmetic, and planned experiments. Model configuration and tokenizer files were inspected at the selected revision; these checks establish architecture/token identities, not task quality or runtime memory.

Tensor-state checks establish only their local mathematical/serialization invariants. No training loss curves, BANKING77 scores, measured communication traffic, peak training RAM/VRAM, convergence results, or weak-client benefits are available in this delivery.

## Next implementation gates

1. Actual one-client CPU/GPU learning with frozen base/A and correct intent/EOS masking.
2. Frozen BANKING77 manifests and tested dependency environments.
3. Five-client federation with deadlines, validated-response quorums, and recovery.
4. Controlled measurements with matched seeds/configuration and unfavorable results included.
5. Three-laptop offline rehearsal and live demonstration.

See [roadmap.md](roadmap.md) and [evaluation.md](evaluation.md) for the acceptance criteria. Update this file with run identities and evidence when those gates are completed.
