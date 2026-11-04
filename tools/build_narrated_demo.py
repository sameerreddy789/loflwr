"""Build the offline recorded walkthrough from existing, immutable evidence.

Requires Windows System.Speech, Pillow, ffmpeg and ffprobe. No training, scoring,
network access, browser control, or model changes occur. Intermediate files are
kept in ignored runs/narrated-demo. See presentation/narrated-demo.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "runs/narrated-demo"
OUT = ROOT / "presentation"
W, H, FPS = 1920, 1080, 24
BG, INK, MUTED, TEAL, GOLD = "#101f2c", "#f4f8fa", "#b9c8d3", "#7cd9ce", "#e9c77b"
VOICE = "Microsoft Zira Desktop"
SOURCES = [
    "presentation/demo-backup.webm",
    "presentation/narrated-demo-source-comparisons.png",
    "research/results/encoder-v2/confirmation-result.json",
    "research/results/encoder-v2/confirmation-plan.json",
    "research/results/encoder-v2/final-policy.json",
    "research/results/encoder-v2/flower-head.json",
    "research/results/encoder-matrix-v1/summary.json",
    "research/results/encoder-matrix-v1/README.md",
    "research/results/remaining-work-v1/five-recovery.json",
    "research/results/remaining-work-v1/README.md",
    "research/results/demo-delivery-v1/verification.json",
]

# Captions give compact equivalents of the narration, with numeric figures.
SCENES = [
    {
        "id": "01-intro", "title": "FedLoRA-Edge", "minimum_seconds": 14,
        "cues": [
            ["This recorded walkthrough shows banking intent learning across five logical clients on one laptop.", "Banking intent learning across five logical clients on one laptop."],
            ["The original frozen A LoRA benchmark is preserved. This confirmed demo uses frozen M P Net and a federated linear classifier head.", "Original frozen-A LoRA preserved. Confirmed demo: frozen MPNet + federated linear head."],
        ],
        "source_note": "Confirmed model: encoder-flower-balanced-v2 · selected version 500",
    },
    {
        "id": "02-query", "title": "A banking question,\nclassified locally", "minimum_seconds": 15,
        "cues": [
            ["Here, recorded local inference maps, my card payment is still pending, to pending card payment.", "Recorded local inference: “My card payment is still pending.” → Pending card payment."],
            ["The support response is team authored. Opening this screen does not start training.", "The support response is team authored. Opening this screen does not start training."],
        ],
        "source_note": "Actual browser recording · 14.0–29.0 seconds · functional example, not an accuracy test",
        "video": {"clips": [[14.0, 15.0]], "crop": [360, 410, 1000, 580], "box": [832, 260, 1012, 587]},
    },
    {
        "id": "03-confirmation", "title": "A locked, independent confirmation", "minimum_seconds": 16,
        "cues": [
            ["On the locked independent confirmation set, the model scored ninety four percent: two hundred eighty two of three hundred.", "Locked independent confirmation: 94% accuracy, 282 / 300 correct."],
            ["The fixed clarification rule reached ninety eight point six five percent on answered queries, at seventy four point three three percent coverage.", "Fixed clarification rule: 98.65% accuracy on answered queries at 74.33% coverage."],
        ],
        "source_note": "encoder-flower-balanced-v2 · pre-scoring lock commit 7413e6b · scored 3 November 2026",
    },
    {
        "id": "04-comparison", "title": "When one client owns 90% of the data", "minimum_seconds": 20,
        "cues": [
            ["Thirty three numerical federation development runs used reused validation.", "33 numerical-federation development runs · reused validation · seeds 42, 43, 44."],
            ["With ninety percent of data on one client, sample weighting scored sixty two point five percent. Equal weights scored seventy six point five.", "90% imbalance: sample weighted 62.50%; equal weights 76.50%."],
            ["Adding clipping reached seventy eight point one seven, plus or minus zero point two nine percentage points.", "Equal weights + clipping: 78.17% ± 0.29 percentage points (sample SD)."],
        ],
        "source_note": "encoder-matrix-v1 · 400 training queries · 200 reused validation queries · fixed attempt 500",
    },
    {
        "id": "05-weak-client", "title": "A one-step client contributes useful knowledge", "minimum_seconds": 12,
        "cues": [
            ["A one step verification client added eighteen percentage points in a matched inclusion comparison.", "One-step verification client: +18 percentage points in the matched inclusion comparison."],
            ["That client held four unique intents. This does not show that every weak device gives the same benefit.", "This client held four unique intents; the observed benefit is specific to this comparison."],
        ],
        "source_note": "encoder-matrix-v1 · all clients 95.5% vs omit client 4 at 77.5% · reused validation, three seeds",
    },
    {
        "id": "06-recovery", "title": "Dropout, quorum, and fresh rejoin", "minimum_seconds": 20,
        "cues": [
            ["The recorded T L S Flower recovery run accepted five, four, five, two, then five clients.", "Recorded TLS Flower recovery: accepted contributors 5 → 4 → 5 → 2 → 5."],
            ["On attempt four, quorum three was missed, so version three stayed unchanged.", "Attempt 4 missed quorum 3. The checkpoint remained at version 3."],
            ["Returning clients synchronized the current head. These were separate processes on one physical laptop.", "Returning clients synchronized the current head. Separate processes on ONE physical laptop."],
        ],
        "source_note": "encoder-isolated-five-v8 · authenticated Flower Deployment Runtime · recorded recovery evidence",
        "video": {"clips": [[33.0, 3.0], [48.0, 3.0], [284.0, 4.0], [345.0, 5.0], [375.0, 5.0]], "crop": [350, 180, 1020, 660], "box": [832, 236, 1012, 655]},
    },
    {
        "id": "07-boundaries", "title": "A compact update. Clear next gates.", "minimum_seconds": 12,
        "cues": [
            ["The head has sixty one thousand five hundred twenty raw F P thirty two bytes per client, per direction.", "61,520 raw FP32 bytes per client, per direction; protocol and transport overhead are additional."],
            ["Physical laptop recovery and fresh human queries remain pending.", "Pending: physical multi-laptop recovery and fresh independent human queries."],
        ],
        "source_note": "15,380 trained head parameters · no model, label, or threshold changed for this walkthrough",
    },
]


def run(args: list[str], **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value):
    # Byte hashes must survive Git's LF normalization on Windows checkouts.
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def font(size=36, bold=False):
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / ("segoeuib.ttf" if bold else "segoeui.ttf")), size)


def wrap(draw, text, size, max_width, bold=False):
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            test = f"{line} {word}".strip()
            if draw.textlength(test, font=font(size, bold)) > max_width and line:
                lines.append(line)
                line = word
            else:
                line = test
        lines.append(line)
    return lines


def txt(draw, xy, text, size=36, color=INK, bold=False, width=None, spacing=1.3):
    x, y = xy
    lines = wrap(draw, text, size, width, bold) if width else text.split("\n")
    for line in lines:
        if x + draw.textlength(line, font=font(size, bold)) > W - 50:
            raise ValueError(f"Text overflows: {line!r}")
        draw.text((x, y), line, font=font(size, bold), fill=color)
        y += int(size * spacing)
    return y


def panel(draw, box, fill="#1b3041", outline=None):
    draw.rounded_rectangle(box, radius=22, fill=fill, outline=outline, width=2)


def base(scene, index):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 8), fill=TEAL)
    txt(d, (76, 50), "FEDLORA-EDGE", 24, TEAL, True)
    txt(d, (76, 95), f"{index + 1:02d} / {len(SCENES):02d}", 22, MUTED)
    panel(d, (1435, 42, 1845, 107), fill="#28434d")
    d.ellipse((1457, 65, 1473, 81), fill=TEAL)
    txt(d, (1490, 53), "Recorded walkthrough", 29, INK, True)
    txt(d, (76, 163), scene["title"], 54, INK, True, width=1780)
    # The caption and provenance remain outside all screen captures.
    panel(d, (52, 908, 1868, 1035), fill="#071520")
    txt(d, (76, 1050), scene["source_note"], 18, MUTED)
    return im, d


def draw_scene(scene, index):
    im, d = base(scene, index)
    sid = scene["id"]
    if sid == "01-intro":
        txt(d, (76, 258), "Banking intent learning across uneven clients", 44, MUTED)
        for x, big, small in [(76, "20", "BANKING77 intents"), (677, "5", "logical clients"), (1278, "1", "physical laptop")]:
            panel(d, (x, 380, x + 566, 652))
            txt(d, (x + 32, 403), big, 120, TEAL, True)
            txt(d, (x + 32, 561), small, 31, INK)
        txt(d, (76, 715), "Confirmed demo", 25, TEAL, True)
        txt(d, (76, 755), "Frozen MPNet + federated linear head", 43, INK, True)
        txt(d, (76, 820), "Original frozen-A LoRA benchmark preserved as a separate model identity", 29, MUTED)
    elif sid == "02-query":
        # Move title to the left; the source pane is composed by ffmpeg.
        d.rectangle((76, 157, 1370, 314), fill=BG)
        txt(d, (76, 166), scene["title"], 55, INK, True, width=710)
        txt(d, (76, 355), "“My card payment\nis still pending.”", 49, TEAL, True, width=710)
        txt(d, (76, 528), "Pending card payment", 38, INK, True, width=710)
        txt(d, (76, 611), "Selected model, version 500\nLocal intent classification\nTeam-authored support response", 30, MUTED, width=700, spacing=1.7)
        panel(d, (820, 247, 1856, 860), fill="#28434d")
    elif sid == "03-confirmation":
        txt(d, (76, 278), "ALL 300 CONFIRMATION QUERIES", 25, TEAL, True)
        txt(d, (64, 322), "94%", 172, INK, True)
        txt(d, (76, 547), "282 / 300 correct", 51, INK, True)
        txt(d, (76, 659), "Model and rules locked before scoring", 29, MUTED, width=790)
        txt(d, (76, 713), "Scored set is now evaluation evidence", 29, MUTED, width=790)
        panel(d, (1040, 289, 1844, 836))
        txt(d, (1080, 326), "FIXED CLARIFICATION RULE", 25, TEAL, True)
        txt(d, (1080, 379), "98.65%", 88, INK, True)
        txt(d, (1080, 495), "accuracy on answered queries", 31, MUTED)
        txt(d, (1080, 575), "74.33%", 80, GOLD, True)
        txt(d, (1080, 681), "coverage of the same 300 queries", 30, MUTED)
        txt(d, (1080, 754), "Selective accuracy always needs coverage", 26, MUTED)
    elif sid == "04-comparison":
        txt(d, (76, 269), "33 development runs", 36, TEAL, True)
        txt(d, (76, 327), "Numerical federation · reused validation", 28, MUTED, width=700)
        for y, method, number, col in [(415, "Sample weighted", "62.50%", MUTED), (565, "Equal weights", "76.50%", INK), (715, "Equal + clipping", "78.17%", TEAL)]:
            txt(d, (76, y), method, 28, MUTED)
            txt(d, (76, y + 37), number, 66, col, True)
        txt(d, (418, 792), "± 0.29 pp SD", 27, MUTED)
        src = Image.open(OUT / "narrated-demo-source-comparisons.png").convert("RGB")
        src = src.crop((350, 190, 1370, 834)).resize((1012, 639), Image.Resampling.LANCZOS)
        im.paste(src, (832, 245))
    elif sid == "05-weak-client":
        txt(d, (76, 268), "MATCHED DEVELOPMENT COMPARISON", 25, TEAL, True)
        txt(d, (76, 302), "+18 pp", 160, TEAL, True)
        txt(d, (76, 520), "From the one-step verification client", 42, INK, True, width=1080)
        txt(d, (76, 615), "95.5% with all clients → 77.5% without client 4", 35, MUTED, width=1100)
        txt(d, (76, 707), "Matched inclusion comparison · seeds 42, 43, 44", 30, MUTED, width=1100)
        panel(d, (1260, 302, 1844, 841))
        txt(d, (1300, 342), "4", 160, GOLD, True)
        txt(d, (1300, 550), "unique intents", 42, INK, True)
        txt(d, (1300, 643), "held by this client", 31, MUTED)
        txt(d, (1300, 724), "Benefit is specific\nto this comparison", 29, MUTED)
    elif sid == "06-recovery":
        d.rectangle((76, 157, 1500, 245), fill=BG)
        txt(d, (76, 166), "Dropout, quorum,\nand fresh rejoin", 54, INK, True, width=710)
        txt(d, (76, 347), "5 → 4 → 5 → 2 → 5", 53, TEAL, True, width=730)
        txt(d, (76, 425), "accepted contributors", 29, MUTED)
        txt(d, (76, 531), "Quorum 3", 40, GOLD, True)
        txt(d, (76, 596), "Attempt 4 retains version 3", 35, INK, True, width=730)
        txt(d, (76, 692), "Fresh rejoin → current head\nTLS-authenticated Flower\nONE physical laptop", 30, MUTED, width=710, spacing=1.7)
        panel(d, (820, 224, 1856, 903), fill="#28434d")
    elif sid == "07-boundaries":
        txt(d, (76, 292), "61,520", 135, TEAL, True)
        txt(d, (76, 476), "raw FP32 bytes", 47, INK, True)
        txt(d, (76, 552), "per client, per direction", 36, MUTED)
        txt(d, (76, 647), "15,380 trained head parameters", 30, MUTED)
        txt(d, (76, 709), "Protocol and transport overhead additional", 28, MUTED, width=870)
        panel(d, (1050, 293, 1844, 849))
        txt(d, (1090, 332), "NEXT EVIDENCE GATES", 26, GOLD, True)
        txt(d, (1090, 419), "Physical multi-laptop\nrecovery", 42, INK, True)
        txt(d, (1090, 587), "Fresh independent\nhuman queries", 42, INK, True)
        txt(d, (1090, 758), "Both remain pending", 31, MUTED)
    im.save(WORK / f"{sid}.png")


def seconds(path):
    with wave.open(str(path)) as handle:
        return handle.getnframes() / handle.getframerate()


def synthesize():
    jobs = []
    for scene in SCENES:
        for i, (spoken, _caption) in enumerate(scene["cues"]):
            jobs.append({"text": spoken, "path": str(WORK / f"{scene['id']}-{i}.wav")})
    write_json(WORK / "speech-jobs.json", jobs)
    # System.Speech uses an installed synthetic voice and writes directly to WAV.
    # Text is read as data from JSON, never interpolated into PowerShell code.
    ps = WORK / "synthesize.ps1"
    ps.write_text("""param([string]$JobsFile, [string]$VoiceName)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$narrator = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $narrator.SelectVoice($VoiceName)
    $narrator.Rate = 1
    $narrator.Volume = 100
    $jobs = Get-Content -Raw -Encoding UTF8 -LiteralPath $JobsFile | ConvertFrom-Json
    foreach ($job in $jobs) {
        $narrator.SetOutputToWaveFile($job.path)
        $narrator.Speak($job.text)
        $narrator.SetOutputToNull()
    }
} finally { $narrator.Dispose() }
""", encoding="utf-8", newline="\n")
    run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps), "-JobsFile", str(WORK / "speech-jobs.json"), "-VoiceName", VOICE])


def ass_time(n):
    cs = round(n * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def vtt_time(n):
    ms = round(n * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def audio_and_captions(scene):
    cue_paths = [WORK / f"{scene['id']}-{i}.wav" for i in range(len(scene["cues"]))]
    with wave.open(str(cue_paths[0])) as handle:
        params = handle.getparams()
    rate, width, channels = params.framerate, params.sampwidth, params.nchannels
    silence = lambda length: bytes(round(length * rate) * width * channels)
    content = bytearray(silence(0.25))
    cues, elapsed = [], 0.25
    for i, path in enumerate(cue_paths):
        with wave.open(str(path)) as handle:
            if handle.getparams()[:3] != params[:3]:
                raise ValueError("Narration WAV parameters differ")
            raw = handle.readframes(handle.getnframes())
        length = seconds(path)
        cues.append({"start": elapsed, "end": elapsed + length, "spoken": scene["cues"][i][0], "caption": scene["cues"][i][1]})
        content.extend(raw)
        content.extend(silence(0.16))
        elapsed += length + 0.16
    duration = math.ceil(max(scene["minimum_seconds"], elapsed + 0.65) * FPS) / FPS
    content.extend(silence(duration - elapsed))
    with wave.open(str(WORK / f"{scene['id']}.wav"), "wb") as handle:
        handle.setparams(params)
        handle.writeframes(content)
    ass = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0", "", "[V4+ Styles]", "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding", "Style: Default,Segoe UI,38,&H00FFFFFF,&H00FFFFFF,&H00071520,&H00071520,0,0,0,0,100,100,0,0,1,1,0,2,95,95,57,1", "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for cue in cues:
        ass.append(f"Dialogue: 0,{ass_time(cue['start'])},{ass_time(cue['end'])},Default,,0,0,0,,{cue['caption']}")
    (WORK / f"{scene['id']}.ass").write_text("\n".join(ass) + "\n", encoding="utf-8-sig", newline="\n")
    prepared = {**scene, "duration_seconds": duration, "timed_cues": cues}
    if scene["id"] == "06-recovery":
        # Put the missed-quorum and fresh-rejoin source views under the matching
        # spoken sentences. The first sentence shows attempts 1, 2, and 3.
        first_part = cues[1]["start"] / 3
        prepared["video"] = {**scene["video"], "clips": [
            [33.0, first_part], [48.0, first_part], [284.0, first_part],
            [345.0, cues[2]["start"] - cues[1]["start"]],
            [375.0, duration - cues[2]["start"]],
        ]}
    return prepared


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    published_capture = OUT / "narrated-demo-source-comparisons.png"
    if not published_capture.exists():
        shutil.copyfile(ROOT / "runs/demo-desktop-comparisons-restarted.png", published_capture)
    for source in SOURCES:
        if not (ROOT / source).is_file():
            raise FileNotFoundError(source)
    evidence = json.loads((ROOT / "research/results/encoder-v2/confirmation-result.json").read_text())
    recovery = json.loads((ROOT / "research/results/remaining-work-v1/five-recovery.json").read_text())
    matrix = json.loads((ROOT / "research/results/encoder-matrix-v1/summary.json").read_text())
    assert (evidence["correct"], evidence["metrics"]["count"], evidence["metrics"]["accuracy"]) == (282, 300, 0.94)
    assert recovery["accepted_counts"] == [5, 4, 5, 2, 5]
    assert recovery["rounds"][3]["version_after"] == 3 and recovery["quorum"] == 3
    assert recovery["traffic"]["raw_head_bytes_per_direction_per_client"] == 61520
    assert matrix["completed"] == 33
    for i, scene in enumerate(SCENES):
        draw_scene(scene, i)
    synthesize()
    prepared = [audio_and_captions(scene) for scene in SCENES]
    total = sum(scene["duration_seconds"] for scene in prepared)
    if total > 120:
        raise ValueError(f"Narration is {total:.3f}s, exceeding the 120s limit; shorten script")
    write_json(WORK / "prepared.json", {"duration_seconds": total, "voice": VOICE, "rate": 1, "scenes": prepared})
    vtt, script, offset = ["WEBVTT", ""], ["# Recorded walkthrough narration", "", f"Voice: {VOICE}, installed Windows synthetic voice (System.Speech), rate +1.", "No voice cloning, external TTS, or paid service. Screens and training evidence are recorded.", ""], 0
    for scene in prepared:
        script.extend([f"## {vtt_time(offset)} — {scene['title'].replace(chr(10), ' ')}", ""])
        for cue in scene["timed_cues"]:
            vtt.extend([f"{vtt_time(offset + cue['start'])} --> {vtt_time(offset + cue['end'])}", cue["caption"], ""])
            script.extend([cue["spoken"], ""])
        script.extend([f"Source: {scene['source_note']}", ""])
        offset += scene["duration_seconds"]
    (OUT / "narrated-demo.vtt").write_text("\n".join(vtt), encoding="utf-8", newline="\n")
    (OUT / "narrated-demo-script.md").write_text("\n".join(script), encoding="utf-8", newline="\n")
    shutil.copyfile(WORK / "01-intro.png", OUT / "narrated-demo-poster.png")
    print(json.dumps({"prepared": True, "duration_seconds": total, "scene_seconds": [s["duration_seconds"] for s in prepared]}))


def encode():
    prepared = json.loads((WORK / "prepared.json").read_text(encoding="utf-8"))
    segments = []
    for scene in prepared["scenes"]:
        sid, duration = scene["id"], scene["duration_seconds"]
        args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-filter_complex_threads", "1", "-loop", "1", "-framerate", str(FPS), "-i", f"{sid}.png"]
        source = scene.get("video")
        filters, n = [], 1
        if source:
            # Each selected source interval plays at real time. Long idle spans
            # are removed with explicit cuts, and the last frame can be held.
            for start, length in source["clips"]:
                args.extend(["-ss", str(start), "-t", str(length), "-i", str(OUT / "demo-backup.webm")])
                filters.append(f"[{n}:v]setpts=PTS-STARTPTS,fps={FPS}[c{n}]")
                n += 1
            ins = "".join(f"[c{i}]" for i in range(1, n))
            x, y, cw, ch = source["crop"]
            bx, by, bw, bh = source["box"]
            filters.append(f"{ins}concat=n={n-1}:v=1:a=0,crop={cw}:{ch}:{x}:{y},scale={bw}:{bh},setsar=1,tpad=stop_mode=clone:stop_duration={duration}[screen]")
            filters.append(f"[0:v][screen]overlay={bx}:{by}:shortest=1,ass={sid}.ass,format=yuv420p[v]")
        else:
            filters.append(f"[0:v]ass={sid}.ass,format=yuv420p[v]")
        args.extend(["-filter_complex", ";".join(filters), "-map", "[v]", "-an", "-t", str(duration), "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-threads", "2", "-r", str(FPS), "-movflags", "+faststart", f"{sid}.mp4"])
        run(args, cwd=WORK)
        segments.append(f"file '{sid}.mp4'")
        print(f"Encoded {sid}: {duration:.3f}s", flush=True)
    (WORK / "concat.txt").write_text("\n".join(segments) + "\n", encoding="ascii", newline="\n")
    staged = WORK / "narrated-demo.mp4"
    # Join PCM before one AAC encode, avoiding encoder-delay drift at scene cuts.
    with wave.open(str(WORK / "narration.wav"), "wb") as output:
        for index, scene in enumerate(prepared["scenes"]):
            with wave.open(str(WORK / f"{scene['id']}.wav")) as part:
                if index == 0:
                    output.setparams(part.getparams())
                output.writeframes(part.readframes(part.getnframes()))
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt", "-i", "narration.wav", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", "loudnorm=I=-17:TP=-2:LRA=7", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-t", str(prepared["duration_seconds"]), "-movflags", "+faststart", "-metadata", "title=FedLoRA-Edge — Recorded walkthrough", "-metadata", f"comment=Recorded evidence; offline synthetic narration by {VOICE}. No new model evaluation.", str(staged)], cwd=WORK)
    probe = json.loads(run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(staged)], capture_output=True).stdout)
    actual = float(probe["format"]["duration"])
    assert actual <= 120, f"Encoded duration exceeds 120 seconds: {actual}"
    assert {s["codec_name"] for s in probe["streams"]} == {"h264", "aac"}
    # A full decode catches broken concatenation or damaged streams.
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(staged), "-f", "null", "-"], capture_output=True)
    # Every review frame lies inside a spoken caption, so inspection covers the
    # actual burned subtitles as well as the slide and captured browser panes.
    sheet = Image.new("RGB", (1920, 580 * 4), BG)
    sheet_draw, offset, review_frames = ImageDraw.Draw(sheet), 0.0, []
    for index, scene in enumerate(prepared["scenes"]):
        cue = scene["timed_cues"][min(1, len(scene["timed_cues"]) - 1)]
        at = offset + (cue["start"] + cue["end"]) / 2
        frame = WORK / f"review-{scene['id']}.png"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(at), "-i", str(staged), "-frames:v", "1", str(frame)], capture_output=True)
        thumb = Image.open(frame).convert("RGB").resize((960, 540), Image.Resampling.LANCZOS)
        x, y = index % 2 * 960, index // 2 * 580
        sheet_draw.text((x + 14, y + 5), f"{scene['id']} · {vtt_time(at)}", font=font(22), fill=TEAL)
        sheet.paste(thumb, (x, y + 38))
        review_frames.append({"path": frame.relative_to(ROOT).as_posix(), "at_seconds": at, "sha256": sha(frame)})
        offset += scene["duration_seconds"]
    sheet.save(WORK / "review-contact-sheet.png")
    write_json(WORK / "review-frames.json", review_frames)
    shutil.copyfile(staged, OUT / "narrated-demo.mp4")
    write_json(WORK / "ffprobe.json", probe)
    artifact_names = ["narrated-demo.mp4", "narrated-demo.vtt", "narrated-demo-script.md", "narrated-demo-poster.png", "narrated-demo-source-comparisons.png", "narrated-demo.md"]
    manifest = {
        "schema_version": 1, "artifact": "presentation/narrated-demo.mp4", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/build_narrated_demo.py", "generator_sha256": sha(Path(__file__)),
        "scope": "Edited recorded UI walkthrough and evidence cards; no live training or new evaluation",
        "duration_seconds": actual, "width": W, "height": H, "frames_per_second": FPS,
        "video_codec": "H.264 (libx264), yuv420p", "audio_codec": "AAC, mono, 48 kHz, nominal 128 kb/s", "full_decode_passed": True,
        "environment": {"python": sys.version, "pillow": PILLOW_VERSION, "platform": sys.platform, "ffmpeg": run(["ffmpeg", "-version"], capture_output=True).stdout.splitlines()[0], "ffprobe": run(["ffprobe", "-version"], capture_output=True).stdout.splitlines()[0]},
        "qa": {"contact_sheet": "runs/narrated-demo/review-contact-sheet.png", "review_frames": review_frames, "probe": "runs/narrated-demo/ffprobe.json", "note": "Captioned frames extracted from the fully decoded final video for visual inspection"},
        "narration": {"type": "synthetic", "engine": "Windows System.Speech / SAPI", "voice": VOICE, "rate": 1, "external_service": False, "voice_clone": False, "normalization": "FFmpeg loudnorm I=-17 LUFS, TP=-2 dBTP, LRA=7"},
        "screen_capture_provenance": {"raw_recording": "presentation/demo-backup.webm", "raw_recording_sha256": sha(OUT / "demo-backup.webm"), "comparison_capture_original": "runs/demo-desktop-comparisons-restarted.png", "comparison_capture_retained": "presentation/narrated-demo-source-comparisons.png", "source_delivery_receipt": "research/results/demo-delivery-v1/verification.json", "recorded_walkthrough_label": "Burned into every frame, outside captured screen pane", "edits": "Trimmed source intervals at original playback speed, cropped/scaled panes, last-frame holds, captions and evidence cards; source metrics unmodified"},
        "identities": {"confirmed_run": "encoder-flower-balanced-v2", "selected_version": 500, "kind": "frozen_mpnet_linear_head", "confirmation_lock": "81fedc0777a0c98f02d071b9dd3adaef862c52c538a165ccc75d2b4f01187681", "confirmation_head": "d85f8e14a62ebaa092c7a1e173e74e5c35a3c6dd59258c08252626f7d6cd7312", "recovery_run": "encoder-isolated-five-v8", "development_study": "encoder-matrix-v1"},
        "source_sha256": {s: sha(ROOT / s) for s in SOURCES},
        "artifacts_sha256": {f"presentation/{name}": sha(OUT / name) for name in artifact_names},
        "scenes": prepared["scenes"],
        "limitations": ["Five logical clients on one physical laptop.", "The confirmed frozen-MPNet head is not LoRA; original LoRA evidence is preserved separately.", "33 development runs use numerical federation and reused validation, not Flower runtime or new confirmation.", "98.65% is selective accuracy at 74.33% coverage, not all-query accuracy.", "The one-step client's 18-point inclusion benefit concerns four intents unique to that client.", "61,520 bytes is the raw FP32 head per client per direction, not total wire traffic.", "Physical multi-laptop recovery and fresh independent human queries remain pending."],
    }
    write_json(OUT / "narrated-demo-manifest.json", manifest)
    print(json.dumps({"artifact": str(OUT / "narrated-demo.mp4"), "duration_seconds": actual, "sha256": sha(OUT / "narrated-demo.mp4"), "full_decode_passed": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Render frames and synthesize narration without CPU-heavy encoding")
    mode.add_argument("--encode-only", action="store_true", help="Encode previously prepared assets")
    args = parser.parse_args()
    if not args.encode_only:
        prepare()
    if not args.prepare_only:
        encode()


if __name__ == "__main__":
    main()
