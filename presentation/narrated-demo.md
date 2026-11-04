# Narrated recorded walkthrough

[narrated-demo.mp4](narrated-demo.mp4) is the short offline backup demo. It combines
actual clips from the existing browser recording, the later captured comparison
screen, and legible evidence cards. **Recorded walkthrough** stays visible
throughout. Captions are burned in; [WebVTT subtitles](narrated-demo.vtt) and the
[spoken script](narrated-demo-script.md) are also provided.

Narration is an installed **Microsoft Zira Desktop synthetic voice**, generated
offline with Windows System.Speech/SAPI. It does not imitate the presenter and
uses no external TTS or paid service. No music is included.

The video identifies five logical clients on one physical laptop, the separate
frozen-MPNet federated head, 94% locked confirmation (282/300), and the fixed
clarification rule's 98.65% accuracy **at 74.33% coverage**. Development validation,
same-host Flower recovery, and pending physical-laptop/fresh-human-query gates
remain distinct. The head's 61,520 raw FP32 bytes per client per direction exclude
protocol and transport overhead. The example query is a functional demonstration,
not fresh accuracy evidence. It does not start training or score another dataset.

[narrated-demo-manifest.json](narrated-demo-manifest.json) records exact duration,
codecs, source intervals and crops, run identities, SHA-256 hashes, synthetic-voice
settings, and the full-decode check. The source query/recovery clips play at their
original speed; idle spans are removed by cuts and final frames can be held.
The retained [comparison capture](narrated-demo-source-comparisons.png) makes the
build independent of its original ignored screenshot path.

The six-slide deck, its original source manifest, and the 416.3-second silent
[raw recording](demo-backup.webm) are preserved unchanged.

## Rebuild offline

Use Python 3.11+ with Pillow, Windows System.Speech with the named installed voice,
and `ffmpeg`/`ffprobe` on PATH. The tested report environment is:

```powershell
& 'C:/Users/samee/.venvs/loflwr-report/Scripts/python.exe' tools/build_narrated_demo.py
```

`--prepare-only` renders frames and creates speech without video encoding;
`--encode-only` encodes those prepared files. Intermediates stay in ignored
`runs/narrated-demo/`. Coordinate encoding with any timing experiments on the same
laptop. The generator validates key source values, refuses a video over 120
seconds, fully decodes the staged result, then publishes the MP4 and manifest.
Different SAPI or codec versions can change hashes; each rebuild records its own.

Review the video with sound before presenting. Synthetic speech and rendered
captions are a backup, not evidence that a human rehearsal or physical deployment
has been completed.
