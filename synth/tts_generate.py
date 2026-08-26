"""Generate Hinglish audio from corpus_plan.jsonl via Microsoft edge-tts.

Resumable: every clip has a deterministic ID; jobs whose outputs already exist
in the manifest are skipped, so rerunning after an interruption (network,
throttling, Ctrl+C) continues where it left off.

For "full" jobs, incomplete "cut" variants are derived from the SAME audio by
slicing at TTS word-boundary timestamps — acoustically natural mid-sentence
stops with zero extra TTS calls.

Run:  python -m synth.tts_generate [--limit N] [--concurrency 5]
Outputs under synth/output/: audio/<utt_id>[.cutK].flac, manifest.jsonl,
failures.jsonl, boundaries/<utt_id>.json
"""

import argparse
import asyncio
import io
import json
import random
import subprocess
from pathlib import Path

import edge_tts
import numpy as np
import soundfile as sf

SR = 16000
OUT = Path(__file__).parent / "output"
AUDIO_DIR = OUT / "audio"
BOUNDS_DIR = OUT / "boundaries"
MANIFEST = OUT / "manifest.jsonl"
FAILURES = OUT / "failures.jsonl"

MIN_CLIP_S = 0.6          # discard cuts shorter than this
MIN_REMOVED_S = 0.4       # a cut must remove at least this much speech


def mp3_to_pcm16k(mp3_bytes: bytes) -> np.ndarray:
    """Decode mp3 -> mono float32 @16k via ffmpeg pipes (no temp files)."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", "1", "pipe:1"],
        input=mp3_bytes, capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


async def synth_one(job: dict) -> tuple[np.ndarray, list[dict]]:
    tts = edge_tts.Communicate(
        job["text"], job["voice"], rate=job["rate"], pitch=job["pitch"],
        boundary="WordBoundary",
    )
    audio = io.BytesIO()
    boundaries = []
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            audio.write(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            boundaries.append(
                {"t_end": (chunk["offset"] + chunk["duration"]) / 1e7,
                 "text": chunk["text"]}
            )
    wav = mp3_to_pcm16k(audio.getvalue())
    return wav, boundaries


def derive_cuts(job: dict, wav: np.ndarray, bounds: list[dict]) -> list[dict]:
    """Slice complete audio at word boundaries -> incomplete variants."""
    rows = []
    if not bounds:
        return rows
    total_s = len(wav) / SR
    for k, frac in enumerate(job.get("cuts", [])):
        idx = round(frac * (len(bounds) - 1))
        idx = max(1, min(idx, len(bounds) - 2)) if len(bounds) >= 3 else None
        if idx is None:
            continue
        cut_t = bounds[idx]["t_end"] + 0.05
        if cut_t < MIN_CLIP_S or total_s - cut_t < MIN_REMOVED_S:
            continue
        cut_id = f"{job['utt_id']}_cut{k}"
        piece = wav[: int(cut_t * SR)]
        sf.write(AUDIO_DIR / f"{cut_id}.flac", piece, SR, subtype="PCM_16")
        rows.append({
            "id": cut_id, "path": f"audio/{cut_id}.flac", "label": 0,
            "language": "hinglish", "source": "synth", "kind": "cut",
            "voice": job["voice"], "script": job["script"],
            "domain": job["domain"], "sentence_id": job["sentence_id"],
            "duration_s": round(len(piece) / SR, 3),
            "text": " ".join(b["text"] for b in bounds[: idx + 1]),
        })
    return rows


async def process_job(job: dict, sem: asyncio.Semaphore, done_ids: set,
                      manifest_f, failures_f, lock: asyncio.Lock):
    if job["utt_id"] in done_ids:
        return "skipped"
    async with sem:
        for attempt in range(4):
            try:
                await asyncio.sleep(random.uniform(0.05, 0.3))
                wav, bounds = await synth_one(job)
                break
            except Exception as e:
                if attempt == 3:
                    async with lock:
                        failures_f.write(json.dumps(
                            {"utt_id": job["utt_id"], "error": repr(e)}) + "\n")
                        failures_f.flush()
                    return "failed"
                await asyncio.sleep(2 ** attempt + random.uniform(0, 1))

    if len(wav) < int(MIN_CLIP_S * SR):
        async with lock:
            failures_f.write(json.dumps(
                {"utt_id": job["utt_id"], "error": "too_short"}) + "\n")
            failures_f.flush()
        return "failed"

    sf.write(AUDIO_DIR / f"{job['utt_id']}.flac", wav, SR, subtype="PCM_16")
    (BOUNDS_DIR / f"{job['utt_id']}.json").write_text(
        json.dumps(bounds, ensure_ascii=False), encoding="utf-8")

    rows = [{
        "id": job["utt_id"], "path": f"audio/{job['utt_id']}.flac",
        "label": job["label"], "language": "hinglish", "source": "synth",
        "kind": job["kind"], "voice": job["voice"], "script": job["script"],
        "domain": job["domain"], "sentence_id": job["sentence_id"],
        "duration_s": round(len(wav) / SR, 3), "text": job["text"],
    }]
    rows += derive_cuts(job, wav, bounds)

    async with lock:
        for r in rows:
            manifest_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest_f.flush()
    return "ok"


async def main_async(limit: int | None, concurrency: int):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    BOUNDS_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [json.loads(l) for l in
            open(OUT / "corpus_plan.jsonl", encoding="utf-8")]
    if limit:
        jobs = jobs[:limit]

    done_ids = set()
    if MANIFEST.exists():
        for l in open(MANIFEST, encoding="utf-8"):
            try:
                done_ids.add(json.loads(l)["id"])
            except json.JSONDecodeError:
                pass  # partial last line from a crash; job will simply re-run

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    n = {"ok": 0, "skipped": 0, "failed": 0}
    with open(MANIFEST, "a", encoding="utf-8") as mf, \
         open(FAILURES, "a", encoding="utf-8") as ff:
        tasks = [process_job(j, sem, done_ids, mf, ff, lock) for j in jobs]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            n[await coro] += 1
            if i % 100 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] ok={n['ok']} "
                      f"skipped={n['skipped']} failed={n['failed']}", flush=True)
    print("done:", n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=5)
    a = ap.parse_args()
    asyncio.run(main_async(a.limit, a.concurrency))
