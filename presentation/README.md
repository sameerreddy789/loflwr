# Judging deck

[fedlora-edge-demo.pptx](fedlora-edge-demo.pptx) is the six-slide offline deck. It contains editable text
and three native charts with embedded data workbooks. Speaker notes provide a
three-minute script, exact source paths, hashes and measured run identities.
[source-manifest.json](source-manifest.json) records the source hashes and unrounded chart values.

The deck separates the frozen-MPNet head's locked 94% confirmation score from
reused-validation development studies and same-laptop process recovery. It keeps
the original LoRA benchmark distinct. The live query model remains unchanged.

## Present

Open the PPTX in PowerPoint or another compatible offline viewer. Use presenter
notes to rehearse. On slide 6, switch to the prepared local query demo described
in the [judging guide](../docs/judging-presentation.md). The recovery chart shows recorded evidence.
The complete five-attempt recovery rehearsal takes about five minutes, so it
does not fit inside the three-minute pitch.

If the browser demo is unavailable, keep this deck open and identify the results
as recorded evidence. Do not infer model accuracy from a few stage queries.

## Backup recording

Use the [short narrated walkthrough](narrated-demo.md) as the prepared fallback.
It has offline synthetic narration, burned captions, editable script and WebVTT
subtitles. Actual browser clips and evidence cards remain explicitly recorded.
Run the [preflight](../docs/demo-preflight.md) before the human rehearsal.
The original deck and recording below retain their 3 November provenance;
the [new delivery](../research/results/demo-extension-v1/README.md) records later checks.

[demo-backup.webm](demo-backup.webm) is an untrimmed, silent browser recording
(6 minutes 56 seconds), with idle time. It includes actual local query inference,
then the explicitly recorded five-client dropout/rejoin and missed-quorum views,
and the experiment comparisons. Scrub past the long pause on attempt 2 when
using it as a fallback. This is a UI walkthrough of completed training evidence,
not a recording of a new training run or the three-minute spoken pitch.

The video predates the final horizontal-chart formatting change; the results
and source run identities are unchanged. No metrics were inserted into frames.
Its source file and the published file are byte-identical. See the
[delivery verification](../research/results/demo-delivery-v1/verification.json).

## Rebuild

The generator reads existing evidence without training, scoring, downloading,
or changing a model. Use the Codex bundled runtime returned by
`load_workspace_dependencies`; the tested bundle is `26.1026.10807` with
`@oai/artifact-tool` `2.8.106`.

```powershell
& 'C:/Users/samee/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe' tools/build_judging_deck.mjs
```

For another installation, pass `--runtime` with the bundled `dependencies`
directory and `--skill-dir` with the presentation skill directory. No npm install
or internet connection is required. Drafts, slide renders and validation receipts
stay in ignored `runs/presentation-build/`. The generator finalizes a new staged
package before replacing the named deck and manifest.

Review all six rendered slides after a rebuild. PowerPoint's native editing
behavior is separate from the automated package, geometry and import checks.
