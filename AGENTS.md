# Working on FedLoRA-Edge

The selected target is SmolLM2-135M-Instruct, twenty BANKING77 intents, shared
frozen A, B-only updates, and equal survivor weights with median norm clipping.
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

The existing dependency-free tensor-state tests run with
`python -m unittest discover -s tests`. Protocol, data, and model tests remain
implementation work. Use the Python 3.11+ setup described in the README and freeze
the full training/Flower environments after actual compatibility smoke tests.
