# Working on FedLoRA-Edge

The original benchmark is SmolLM2-135M-Instruct, twenty BANKING77 intents, shared
frozen A, B-only updates, and equal survivor weights with median norm clipping.
On 3 November the user authorized implementing accuracy-focused architecture
variants. The active development adds frozen MPNet plus a federated classifier
head, followed by shared-A MPNet LoRA B plus a classifier head. Preserve original
benchmark results and identities; do not mix these variants or call head-only
training LoRA. See `research/accuracy-recovery-plan.md` and measured run identities.
Later representation diagnostics in `tools/run_encoder_representation_study.py`
adapt transformer layers on pooled public training data. They are centralized
development references, not LoRA, Flower results, or a replacement for the
confirmed demo. Use `research/results/encoder-representation/` for their evidence.
The 300-query confirmation split in `assets/encoder-bank20-v1/confirmation.json`
was scored on 3 November after locking model/configuration and all margin rules
in commit `7413e6b` (lock `81fedc0777a0c98f02d071b9dd3adaef862c52c538a165ccc75d2b4f01187681`).
The fixed Flower head scored 94% (282/300). This is now evaluation evidence,
not tuning data. Do not change the model, thresholds, labels or exclusions based
on these results and claim another untouched confirmation result. Further model
selection needs a new independently held-out evaluation set. Validation remains
the reused 200-query development split. Keep original negative results.
`research/reference-team-brief/` preserves an earlier Qwen/77-intent/capped-weight
proposal. Do not treat it as the active specification.

Use the project skills when the corresponding task applies:

- `skills/frozen-a-training/SKILL.md`: model preparation, training, and adapter identity.
- `skills/federated-rounds/SKILL.md`: aggregation, deadlines, dropout, and checkpoint state.
- `skills/banking77-experiments/SKILL.md`: partitions, evaluation, and matched experiments.
- `skills/laptop-deployment/SKILL.md`: offline Flower deployment and live rehearsal.

Keep model/data downloads in ignored `assets/`, run outputs in ignored `runs/`,
and private node keys in ignored `certificates/`. Publish measured results with
their run identity and configuration. A toy protocol check is not a language-model
experiment. Do not add fabricated training metrics to documentation or the dashboard.

Run `python -m unittest discover -s tests` for the state, aggregation, protocol,
data, trainer, result-reader, and study-plan checks. Optional Torch, Flower, and
Streamlit checks skip when their dependencies are absent; read the skip count.
Real model optimization, Flower runtime, and physical-laptop recovery remain
separate evidence gates. Use the Python 3.11+ setup and tested environment pins
described in the README; update pins only after actual compatibility smoke tests.
