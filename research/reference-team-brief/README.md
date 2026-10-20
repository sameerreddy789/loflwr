# Earlier proposal and model inspection

This folder retains metadata about an earlier **FairTune** proposal: Qwen2.5-0.5B, all 77 intents, a classifier head, and capped square-root aggregation coefficients. The active design instead uses SmolLM2-135M-Instruct, 20 intents, common frozen A, B-only updates, and equal survivor coefficients with median-norm clipping.

[legacy-model-inspection.json](legacy-model-inspection.json) records a standalone Qwen weights file inspected on 19 October 2026. Its recorded stored parameter count was 494,032,768 and its file size was 988,097,824 bytes. It did not include the tokenizer/configuration assets required for a complete usable checkpoint.

The owner removed the original `IIT-B` folder. Its absence was confirmed during the 20 October repository inventory. The verified [FairTune team brief PDF](FairTune_Team_Brief.pdf) was recovered from Downloads and is retained here as the earlier proposal. Its SHA-256 is `c3650404e3ff74ea13e5e53e274eb536d748b47b84628bfaf9d7cf3ca7491a69`. The original editable documents and diagrams were not recovered. The PDF describes the earlier Qwen design; the active implementation specification is in the current research dossier.

The removed model was unnecessary for the selected implementation. Its inspection measurements do not establish Qwen training performance or imply that Qwen is unusable on every 8 GB laptop.
