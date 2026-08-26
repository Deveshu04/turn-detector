"""Build the deterministic TTS job plan for the Hinglish synthetic dataset.

Reads sentences.py, expands slot templates, assigns voices/prosody/variants with
a fixed seed, and writes synth/output/corpus_plan.jsonl. Job IDs are stable
hashes of their content, so tts_generate.py can resume by skipping existing
outputs after an interruption.

Variant scheme per sentence instance:
  - N_FULL_RENDITIONS "full" jobs (complete turns). Each also declares 1-2
    audio-cut fractions; tts_generate.py derives incomplete "cut" clips from
    the same audio using TTS word-boundary timestamps (no extra TTS calls).
  - one "tail_conj" job (prefix + trailing conjunction) -> incomplete
  - one "tail_filler" job (prefix + hesitation filler) -> incomplete
  - for ~half of sentences, one "midfiller_full" job (filler inserted
    mid-sentence, still a complete turn) -> complete

Run: python -m synth.corpus
"""

import hashlib
import json
import math
import random
from pathlib import Path

from synth.sentences import (
    CONJ_TAILS,
    FILLER_TAILS,
    MID_FILLERS,
    SLOTS,
    TEMPLATES,
)

SEED = 20260826
N_SLOT_COMBOS = 8          # instances per slotted template
N_FULL_RENDITIONS = 5      # complete renditions per sentence instance
MIDFILLER_FRAC = 0.8

DEVA_VOICES = ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"]
LATIN_VOICES = ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural"]
RATES = ["-10%", "-5%", "+0%", "+8%", "+15%"]
PITCHES = ["-15Hz", "-5Hz", "+0Hz", "+10Hz", "+20Hz"]

QUESTION_WORDS = {
    "क्या", "कब", "कितना", "कहां", "कौन", "कहाँ",
    "kya", "kab", "kitna", "kitni", "kahan", "kaun",
}

OUT_DIR = Path(__file__).parent / "output"


def _slot_names(text: str) -> list[str]:
    return [s for s in SLOTS if "{" + s + "}" in text]


def _fill(text: str, script: str, combo: dict) -> str:
    for slot, value in combo.items():
        text = text.replace("{" + slot + "}", value[script])
    return text


def _is_question(latin_text: str) -> bool:
    words = latin_text.lower().split()
    return any(w in QUESTION_WORDS for w in words)


def _utt_id(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def expand_sentences(rng: random.Random) -> list[dict]:
    """Expand templates x slot combos into concrete sentence instances."""
    instances = []
    for t_idx, (deva, latin, domain) in enumerate(TEMPLATES):
        slots = _slot_names(deva)
        combos: list[dict]
        if slots:
            combos = []
            for _ in range(N_SLOT_COMBOS):
                combos.append({s: rng.choice(SLOTS[s]) for s in slots})
            # dedupe identical combos
            seen, unique = set(), []
            for c in combos:
                key = tuple(c[s]["latin"] for s in slots)
                if key not in seen:
                    seen.add(key)
                    unique.append(c)
            combos = unique
        else:
            combos = [{}]
        for c_idx, combo in enumerate(combos):
            instances.append({
                "sentence_id": f"s{t_idx:03d}_{c_idx}",
                "deva": _fill(deva, "deva", combo),
                "latin": _fill(latin, "latin", combo),
                "domain": domain,
                "is_question": _is_question(_fill(latin, "latin", combo)),
            })
    return instances


def _rendition(rng: random.Random, sent: dict) -> dict:
    script = rng.choice(["deva", "latin"])
    voice = rng.choice(DEVA_VOICES if script == "deva" else LATIN_VOICES)
    return {
        "script": script,
        "voice": voice,
        "rate": rng.choice(RATES),
        "pitch": rng.choice(PITCHES),
        "text_base": sent[script],
    }


def _prefix(text: str, frac: float) -> str:
    words = text.split()
    n = max(2, math.ceil(len(words) * frac))
    return " ".join(words[:min(n, len(words) - 1)])


def build_plan() -> list[dict]:
    rng = random.Random(SEED)
    jobs = []
    for sent in expand_sentences(rng):
        sid = sent["sentence_id"]

        # complete renditions with declared audio-cut fractions
        for r in range(N_FULL_RENDITIONS):
            ren = _rendition(rng, sent)
            n_words = len(ren["text_base"].split())
            cuts = []
            if n_words >= 6:
                n_cuts = 1 if rng.random() < 0.5 else 2
                cuts = [round(rng.uniform(0.45, 0.80), 3) for _ in range(n_cuts)]
            end_punct = "?" if sent["is_question"] else ("।" if ren["script"] == "deva" else ".")
            jobs.append({
                "utt_id": _utt_id(sid, "full", r, ren["voice"], ren["rate"], ren["pitch"]),
                "sentence_id": sid, "domain": sent["domain"], "kind": "full",
                "label": 1, "text": ren["text_base"] + end_punct,
                "voice": ren["voice"], "script": ren["script"],
                "rate": ren["rate"], "pitch": ren["pitch"], "cuts": cuts,
            })

        # trailing conjunction -> incomplete
        ren = _rendition(rng, sent)
        tail = rng.choice(CONJ_TAILS)[ren["script"] if ren["script"] in ("deva",) else "latin"]
        jobs.append({
            "utt_id": _utt_id(sid, "tail_conj", ren["voice"], ren["rate"], tail),
            "sentence_id": sid, "domain": sent["domain"], "kind": "tail_conj",
            "label": 0, "text": _prefix(ren["text_base"], 0.65) + " " + tail + "...",
            "voice": ren["voice"], "script": ren["script"],
            "rate": ren["rate"], "pitch": ren["pitch"], "cuts": [],
        })

        # trailing hesitation filler -> incomplete
        ren = _rendition(rng, sent)
        tail = rng.choice(FILLER_TAILS)[ren["script"] if ren["script"] in ("deva",) else "latin"]
        jobs.append({
            "utt_id": _utt_id(sid, "tail_filler", ren["voice"], ren["rate"], tail),
            "sentence_id": sid, "domain": sent["domain"], "kind": "tail_filler",
            "label": 0, "text": _prefix(ren["text_base"], 0.7) + " " + tail + "...",
            "voice": ren["voice"], "script": ren["script"],
            "rate": ren["rate"], "pitch": ren["pitch"], "cuts": [],
        })

        # mid-sentence filler, still complete
        if rng.random() < MIDFILLER_FRAC:
            ren = _rendition(rng, sent)
            words = ren["text_base"].split()
            if len(words) >= 5:
                pos = rng.randint(2, len(words) - 2)
                filler = rng.choice(MID_FILLERS)[ren["script"] if ren["script"] in ("deva",) else "latin"]
                text = " ".join(words[:pos]) + " " + filler + " " + " ".join(words[pos:])
                end_punct = "?" if sent["is_question"] else ("।" if ren["script"] == "deva" else ".")
                jobs.append({
                    "utt_id": _utt_id(sid, "midfiller", ren["voice"], ren["rate"], pos, filler),
                    "sentence_id": sid, "domain": sent["domain"], "kind": "midfiller_full",
                    "label": 1, "text": text + end_punct,
                    "voice": ren["voice"], "script": ren["script"],
                    "rate": ren["rate"], "pitch": ren["pitch"], "cuts": [],
                })
    return jobs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_plan()
    plan_path = OUT_DIR / "corpus_plan.jsonl"
    with open(plan_path, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    n_complete = sum(1 for j in jobs if j["label"] == 1)
    n_incomplete = len(jobs) - n_complete
    n_cut = sum(len(j["cuts"]) for j in jobs)
    sentence_ids = {j["sentence_id"] for j in jobs}
    print(f"plan: {len(jobs)} TTS jobs -> {plan_path}")
    print(f"  sentence instances : {len(sentence_ids)}")
    print(f"  complete clips     : {n_complete} (+{sum(1 for j in jobs if j['kind']=='midfiller_full')} midfiller)")
    print(f"  incomplete synth   : {n_incomplete}")
    print(f"  derived cut clips  : {n_cut} (incomplete, no extra TTS)")
    print(f"  total clips        : {len(jobs) + n_cut}")
    kinds = {}
    for j in jobs:
        kinds[j["kind"]] = kinds.get(j["kind"], 0) + 1
    print(f"  kinds              : {kinds}")


if __name__ == "__main__":
    main()
