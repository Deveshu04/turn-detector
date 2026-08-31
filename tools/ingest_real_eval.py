"""Ingest phone/voice-recorder clips as the real-voice Hinglish eval set.

Workflow:
  1. `python -m tools.ingest_real_eval --sheet` writes data/real_eval/PROMPTS.txt.
     Open it on your phone and record one clip per prompt, in order.
  2. Put the 30 files (m4a/mp3/wav/ogg, or anything ffmpeg reads) into one folder,
     named so alphabetical order == prompt order (phone recorders' default
     "Recording 001..." naming already does this).
  3. `python -m tools.ingest_real_eval <folder>` converts each to 16 kHz mono
     FLAC under data/real_eval/audio/ and writes manifest.jsonl with labels
     taken from the prompt list in tools/record_real_eval.py.
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from record_real_eval import OUT, PROMPTS, SR  # noqa: E402

AUDIO = OUT / "audio"
MANIFEST = OUT / "manifest.jsonl"
EXTS = {".m4a", ".mp3", ".wav", ".ogg", ".opus", ".flac", ".aac", ".webm", ".3gp"}


def write_sheet():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["REAL-VOICE HINGLISH EVAL: record one clip per line, in order.",
             "Speak naturally. Stop recording ~1 second after you stop speaking.",
             ""]
    for i, p in enumerate(PROMPTS, 1):
        tag = "COMPLETE" if p["label"] == 1 else "INCOMPLETE"
        lines.append(f"{i:02d}. [{tag}] “{p['text']}”")
        lines.append(f"    how: {p['note']}")
        lines.append("")
    path = OUT / "PROMPTS.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}; open it on your phone and record in order")


def ingest(folder: str):
    files = sorted(
        (p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS),
        # numeric stems sort as numbers ("2" before "10"), others alphabetically
        key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem.lower()),
    )
    if not files:
        sys.exit(f"no audio files found in {folder}")
    if len(files) != len(PROMPTS):
        print(f"warning: {len(files)} files for {len(PROMPTS)} prompts, "
              f"mapping the first {min(len(files), len(PROMPTS))} in order")
    AUDIO.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (src, p) in enumerate(zip(files, PROMPTS)):
        rid = f"real{i:02d}"
        dst = AUDIO / f"{rid}.flac"
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src),
             "-ar", str(SR), "-ac", "1", "-sample_fmt", "s16", str(dst)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"ffmpeg failed on {src.name}: {proc.stderr.strip()}")
        import soundfile as sf
        info = sf.info(dst)
        rows.append({"id": rid, "path": f"audio/{rid}.flac", "label": p["label"],
                     "language": "hinglish", "source": "real_human",
                     "kind": p["kind"], "duration_s": round(info.duration, 2),
                     "text": p["text"], "split": "test",
                     "source_file": src.name})
        print(f"{rid}  {info.duration:5.1f}s  [{p['kind']:18s}]  <- {src.name}")
    with open(MANIFEST, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_c = sum(r["label"] for r in rows)
    print(f"\n{len(rows)} clips ingested ({n_c} complete / {len(rows)-n_c} "
          f"incomplete) -> {MANIFEST}")
    print("Listen-check a couple of FLACs in data/real_eval/audio/ before "
          "scoring.")


if __name__ == "__main__":
    if "--sheet" in sys.argv or len(sys.argv) == 1:
        write_sheet()
    else:
        ingest(sys.argv[1])
