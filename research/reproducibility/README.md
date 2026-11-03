# Reproducibility records

`banking77-manifests.json` records the verified dataset revision, original-file hashes, deterministic selection seed, normalization rule, exclusion counts, and the twelve manifest identities. It contains counts and hashes, not the training or held-out queries.

Generate and verify the original ignored assets with:

```sh
python -m fedlora_edge.data download
python -m fedlora_edge.data prepare
python -m fedlora_edge.data verify
```

The committed index is a byte-for-byte copy of `assets/banking77/manifests/index.json` after successful verification on 2 November 2026. The manifest-set hash is `ae4831d655b0f0a0a56039d39b97357e4d25ae35be2236c63fca15899b1465fc`. The implementation conservatively excludes training matches to every official test query before selecting the held-out subset.

BANKING77 is supplied by [PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets) under CC BY 4.0. The download preserves its original LICENSE and README in the ignored source cache. This record establishes dataset preparation, not model accuracy or convergence.
