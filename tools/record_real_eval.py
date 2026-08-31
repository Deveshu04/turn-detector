"""Guided recorder for the real-voice Hinglish evaluation set.

Run:  uv run python tools/record_real_eval.py   (opens a local Gradio page)

Shows 30 prompts one at a time; each recording is resampled to 16 kHz mono
FLAC under data/real_eval/audio/ with its label in manifest.jsonl. Re-running
resumes at the first unrecorded prompt; Redo overwrites the current one.

The sentences are written fresh for this eval: none reuse the synthetic
training templates in synth/sentences.py, so there is no train contamination.
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import gradio as gr  # noqa: E402

OUT = Path(__file__).parent.parent / "data" / "real_eval"
AUDIO = OUT / "audio"
MANIFEST = OUT / "manifest.jsonl"
SR = 16000

# label 1 = complete turn, 0 = still speaking. All texts are NEW sentences,
# deliberately disjoint from the synthetic training corpus.
PROMPTS = [
    # A: complete sentences; end them naturally, like talking to a friend
    dict(kind="complete", label=1, text="Yaar aaj ka weather ekdum mast hai, chalo bahar chalte hain.", note="End cleanly, normal pace."),
    dict(kind="complete", label=1, text="Mujhe subah gym jaana tha but neend hi nahi khuli.", note="End cleanly."),
    dict(kind="complete", label=1, text="Cricket match dekha kal raat ko? Last over pagal tha.", note="Question, natural rising tone."),
    dict(kind="complete", label=1, text="Maine nayi movie dekhi weekend pe, ending thodi weird thi.", note="End cleanly."),
    dict(kind="complete", label=1, text="Bhaiya thoda kam kar do na, do sau bahut zyada hai.", note="Bargaining tone, end cleanly."),
    dict(kind="complete", label=1, text="College mein aaj professor ne surprise test le liya.", note="End cleanly."),
    dict(kind="complete", label=1, text="Ghar pe sab theek hai, mummy papa dono mast hain.", note="End cleanly."),
    dict(kind="complete", label=1, text="Chai peeni hai toh jaldi aa jao, thandi ho rahi hai.", note="End cleanly."),
    dict(kind="complete", label=1, text="Is Sunday ko hum log beach jaane ka plan bana rahe hain.", note="End cleanly."),
    dict(kind="complete", label=1, text="Mera laptop finally repair ho gaya, ab kaam start kar sakta hoon.", note="End cleanly."),
    # B: complete WITH a mid-sentence filler; hesitate mid-way, but FINISH
    dict(kind="midfiller_complete", label=1, text="Maine usko bola ki... matlab... hum kal milte hain pakka.", note="Hesitate at the dots, then FINISH the sentence."),
    dict(kind="midfiller_complete", label=1, text="Wo restaurant... umm... haan Karim's, wahan ka khana zabardast hai.", note="Pause mid-way, then finish."),
    dict(kind="midfiller_complete", label=1, text="Meeting shayad... kya bolte hain... postpone ho gayi hai next week tak.", note="Hesitate, then finish."),
    dict(kind="midfiller_complete", label=1, text="Mujhe lagta hai... haan toh... hum train se hi chalenge, sasta padega.", note="Hesitate, then finish."),
    # C: incomplete; end ON the conjunction, as if more is coming
    dict(kind="trailing_conj", label=0, text="Main market gaya tha lekin...", note="Stop right after 'lekin', voice hanging."),
    dict(kind="trailing_conj", label=0, text="Khana toh bana liya hai aur...", note="Stop after 'aur', as if listing more."),
    dict(kind="trailing_conj", label=0, text="Wo aaj nahi aa payega kyunki...", note="Stop after 'kyunki', reason coming."),
    dict(kind="trailing_conj", label=0, text="Agar baarish nahi hui toh...", note="Stop after 'toh'."),
    dict(kind="trailing_conj", label=0, text="Paise toh maine bhej diye the par...", note="Stop after 'par'."),
    # D: incomplete; trail off on a filler, still thinking
    dict(kind="trailing_filler", label=0, text="Uska naam kya tha... umm...", note="Trail off, thinking hard."),
    dict(kind="trailing_filler", label=0, text="Main keh raha tha ki matlab...", note="Trail off on 'matlab'."),
    dict(kind="trailing_filler", label=0, text="Wo jagah... kya bolte hain usko...", note="Searching for a word."),
    dict(kind="trailing_filler", label=0, text="Haan toh phir humne socha ki... wo...", note="Trail off on 'wo'."),
    dict(kind="trailing_filler", label=0, text="Iska price tha shayad... umm...", note="Trying to remember, trail off."),
    # E: incomplete; STOP mid-sentence on a genuine thinking pause. Do not finish!
    dict(kind="midthought_stop", label=0, text="Kal main office ja raha tha aur raste mein--", note="STOP dead at the dash, as if lost in thought. Keep recording ~1s of silence."),
    dict(kind="midthought_stop", label=0, text="Mujhe na ek cheez samajh nahi aayi ki--", note="Stop at the dash. ~1s silence after."),
    dict(kind="midthought_stop", label=0, text="Agle mahine se main soch raha tha ki roz--", note="Stop at the dash. ~1s silence after."),
    dict(kind="midthought_stop", label=0, text="Uski shaadi mein hum sab log--", note="Stop at the dash. ~1s silence after."),
    dict(kind="midthought_stop", label=0, text="Ye recipe mein pehle onion daalna hai phir--", note="Stop at the dash. ~1s silence after."),
    dict(kind="midthought_stop", label=0, text="Boss ne bola tha ki agar client--", note="Stop at the dash. ~1s silence after."),
]


def to_16k_mono(sr: int, wav: np.ndarray) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.dtype.kind in "iu":
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    wav = wav.astype(np.float32)
    if sr != SR:
        x_new = np.linspace(0, len(wav) - 1, int(len(wav) * SR / sr))
        wav = np.interp(x_new, np.arange(len(wav)), wav).astype(np.float32)
    return wav


def recorded_ids() -> set:
    done = set()
    if MANIFEST.exists():
        for line in open(MANIFEST, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except json.JSONDecodeError:
                pass
    return done


def first_pending() -> int:
    done = recorded_ids()
    for i in range(len(PROMPTS)):
        if f"real{i:02d}" not in done:
            return i
    return len(PROMPTS)


def prompt_view(i: int) -> str:
    if i >= len(PROMPTS):
        return ("## ✅ All 30 clips recorded!\n\nScoring runs from "
                "`data/real_eval/`.")
    p = PROMPTS[i]
    lbl = "🟢 COMPLETE turn" if p["label"] == 1 else "🟠 INCOMPLETE (still speaking)"
    return (f"## Clip {i + 1} / {len(PROMPTS)}: {lbl}\n\n"
            f"# “{p['text']}”\n\n**How:** {p['note']}\n\n"
            f"*Speak naturally, don't read robotically. Stop the recording "
            f"about a second after you stop speaking.*")


def save(audio, i: int):
    if audio is None:
        return gr.update(), i, "Record something first."
    if i >= len(PROMPTS):
        return gr.update(value=None), i, "Already finished."
    sr, wav = audio
    wav = to_16k_mono(sr, wav)
    if len(wav) < SR // 2:
        return gr.update(), i, "Too short, try again."
    AUDIO.mkdir(parents=True, exist_ok=True)
    p = PROMPTS[i]
    rid = f"real{i:02d}"
    sf.write(AUDIO / f"{rid}.flac", wav, SR, subtype="PCM_16")
    rows = [r for r in (json.loads(l) for l in open(MANIFEST, encoding="utf-8"))
            if r["id"] != rid] if MANIFEST.exists() else []
    rows.append({"id": rid, "path": f"audio/{rid}.flac", "label": p["label"],
                 "language": "hinglish", "source": "real_human",
                 "kind": p["kind"], "duration_s": round(len(wav) / SR, 2),
                 "text": p["text"], "split": "test"})
    with open(MANIFEST, "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    nxt = first_pending()
    return gr.update(value=None), nxt, f"Saved {rid} ({len(wav)/SR:.1f}s). {len(rows)}/{len(PROMPTS)} done."


with gr.Blocks(title="Real Hinglish eval recorder") as app:
    idx = gr.State(first_pending())
    view = gr.Markdown(prompt_view(first_pending()))
    mic = gr.Audio(sources=["microphone"], type="numpy", label="Record")
    status = gr.Markdown()
    with gr.Row():
        save_btn = gr.Button("Save & next ▶", variant="primary")
        redo_btn = gr.Button("◀ Back one")

    def do_save(audio, i):
        upd, nxt, msg = save(audio, i)
        return upd, nxt, msg, prompt_view(nxt)

    def go_back(i):
        prev = max(0, i - 1)
        return prev, prompt_view(prev), "Re-record this one (Save overwrites)."

    save_btn.click(do_save, [mic, idx], [mic, idx, status, view])
    redo_btn.click(go_back, [idx], [idx, view, status])

if __name__ == "__main__":
    app.launch(inbrowser=True)
