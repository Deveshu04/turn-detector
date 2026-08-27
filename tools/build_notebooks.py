"""Generate self-contained Kaggle notebooks from the repo sources.

02_train.ipynb embeds src/turn_detector/*.py as %%writefile cells, so the
notebook always matches the tested library — single source of truth, no
Kaggle-side package management. Re-run this after editing src/ and re-upload.

Run: python -m tools.build_notebooks
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src" / "turn_detector"
OUT = ROOT / "notebooks" / "kaggle"

MODULES = ["__init__", "common", "features", "augment", "config", "model",
           "dataset", "train"]


# --------------------------------------------------------------------------
# 01_data_prep — CPU, internet ON
# --------------------------------------------------------------------------

PREP_MD = """\
# 01 · Data prep — smart-turn v3.2 → English + Hindi + a multilingual tail

**Kaggle settings:** CPU (no GPU needed) · **Internet ON** · takes ~2-3 h.

Streams `pipecat-ai/smart-turn-data-v3.2-train` (41 GB) and `-test`, resamples
to 16 kHz mono, keeps the last 8 s, writes FLAC + `manifest.parquet` to
`/kaggle/working/prep`.

**Composition** (this is the E6 "full data" prep, a superset of the E1-E4 one):

| stream | kept | cap |
|---|---|---|
| train, `eng` | all (effectively uncapped) | 33,000 / label |
| train, `hin` | all | — |
| train, everything else | a tail for multilingual robustness | 850 / (language, label) |
| test | `eng` + `hin` only | — |

English/Hindi are renamed to `english`/`hindi` (every downstream consumer masks
on the long names); other languages keep their **raw ISO-639-3 code** and are
all assigned `split="train"` — validation stays EN+HI so that best-checkpoint
selection is comparable with the earlier experiments, and the test stream is
untouched so the headline numbers keep meaning the same thing.

**Size:** expect **~16-17 GB**; Kaggle's `/kaggle/working` limit is ~19.6 GB.
The final cell prints the working-size total — that printout **is** the guard:
if it comes out near 19 GB, lower `OTHER_CAP_PER_LANG_LABEL` and re-run rather
than pushing a training job at a truncated dataset.

**When it finishes:** *Save Version* → after it completes, create a Kaggle
Dataset from this notebook's output named **`smart-turn-enhi-prep`**
(New Dataset → import from notebook output). Training runs attach that dataset.

**Interrupted?** A committed batch run starts from an empty `/kaggle/working`,
so there is nothing to resume from — just *Save & Run All* again. Prep normally
finishes well inside a single session. (The resume bookkeeping in the code only
helps when you re-run cells inside one live interactive session.)
"""

# datasets >= 4 routes Audio decoding through torchcodec (extra FFmpeg deps);
# 3.x decodes with soundfile, which is already available on Kaggle images.
PREP_PIP = '%pip install -q "datasets>=3.6,<4" soundfile polars'

PREP_MAIN = r'''
import hashlib, json, time
from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf
from datasets import Audio, load_dataset

OUT = Path("/kaggle/working/prep")
AUDIO_DIR = OUT / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT / "manifest.jsonl"

SR = 16000
MAX_SECONDS = 8.0
EN_TRAIN_CAP_PER_LABEL = 33000   # english train rows per label (~all of them)
OTHER_CAP_PER_LANG_LABEL = 850   # non-EN/HI train rows per (language, label)
VAL_PCT = 5                      # % of train-source rows held out as val

# smart-turn v3.2 tags language with ISO-639-3 codes; the manifest (and every
# downstream consumer) uses the long names for the two languages we report on,
# so normalise those once here and key EVERYTHING below -- rec, counters,
# resume rebuild -- on the normalised value. Every other language keeps its raw
# ISO code, which is what train.py's "multilingual_other" slice masks against.
LANG_MAP = {"eng": "english", "hin": "hindi"}
CORE = ("eng", "hin")

# a killed session can leave a half-written last line; drop it before appending
if MANIFEST.exists() and MANIFEST.stat().st_size:
    with open(MANIFEST, "rb+") as f:
        f.seek(-1, 2)
        if f.read(1) != b"\n":
            cut = MANIFEST.read_bytes().rfind(b"\n")
            f.truncate(cut + 1)
            print(f"truncated partial last line (kept {cut + 1} bytes)")

# ---- resume: rebuild done-set and counters from an existing manifest ----
done_ids, counts = set(), {}
if MANIFEST.exists():
    for line in open(MANIFEST):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        done_ids.add(r["id"])
        src = "test" if r["split"] == "test" else "train"
        counts[(r["language"], r["label"], src)] = \
            counts.get((r["language"], r["label"], src), 0) + 1
print(f"resuming with {len(done_ids)} rows, counts={counts}")

mf = open(MANIFEST, "a")

def process(row, src):
    # a null language tag would otherwise write "language": null into the
    # manifest and break every downstream string mask
    code = row["language"] or "unk"
    core = code in CORE
    # the test stream stays EN+HI: overall/test numbers must keep comparing
    # like with like across E1-E6.
    if src == "test" and not core:
        return 0
    lang = LANG_MAP.get(code, code)
    label = int(bool(row["endpoint_bool"]))
    rid = row["id"]
    if rid in done_ids:
        return 0
    if src == "train":
        if lang == "english" and \
                counts.get(("english", label, "train"), 0) >= EN_TRAIN_CAP_PER_LABEL:
            return 0
        if not core and \
                counts.get((lang, label, "train"), 0) >= OTHER_CAP_PER_LANG_LABEL:
            return 0
    wav = np.asarray(row["audio"]["array"], dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav[-int(MAX_SECONDS * SR):]
    if len(wav) < int(0.3 * SR):
        return 0
    sub = AUDIO_DIR / rid[:2]
    sub.mkdir(exist_ok=True)
    sf.write(sub / f"{rid}.flac", wav, SR, subtype="PCM_16")
    if src == "test":
        split = "test"
    elif not core:
        # multilingual tail is train-only on purpose: val must stay EN+HI so
        # best-checkpoint selection is comparable across every experiment.
        split = "train"
    else:
        split = "val" if int(hashlib.md5(rid.encode()).hexdigest(), 16) % 100 < VAL_PCT else "train"
    rec = {
        "id": rid, "path": f"audio/{rid[:2]}/{rid}.flac", "label": label,
        "language": lang,
        "midfiller": None if row["midfiller"] is None else bool(row["midfiller"]),
        "endfiller": None if row["endfiller"] is None else bool(row["endfiller"]),
        "synthetic": bool(row["synthetic"]),
        "dataset": row.get("dataset"), "duration_s": round(len(wav) / SR, 3),
        "split": split, "source": "real", "kind": "",
    }
    mf.write(json.dumps(rec) + "\n")
    done_ids.add(rid)
    counts[(lang, label, src)] = counts.get((lang, label, src), 0) + 1
    return 1

for src, name in [("train", "pipecat-ai/smart-turn-data-v3.2-train"),
                  ("test", "pipecat-ai/smart-turn-data-v3.2-test")]:
    dd = load_dataset(name, streaming=True)
    split_name = "train" if "train" in dd else list(dd.keys())[0]
    ds = dd[split_name].cast_column("audio", Audio(sampling_rate=SR))
    t0, n_scanned, n_kept = time.time(), 0, 0
    for row in ds:
        n_scanned += 1
        n_kept += process(row, src)
        if n_scanned % 5000 == 0:
            mf.flush()
            print(f"[{src}] scanned {n_scanned} kept {n_kept} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    mf.flush()
    print(f"[{src}] DONE: scanned {n_scanned}, kept {n_kept}")

mf.close()
'''

PREP_FINALIZE = r'''
import json
import polars as pl
from pathlib import Path

OUT = Path("/kaggle/working/prep")
seen, unique, bad = set(), [], 0
for line in open(OUT / "manifest.jsonl"):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        bad += 1
        continue
    if r["id"] not in seen:
        seen.add(r["id"])
        unique.append(r)
if bad:
    print(f"skipped {bad} unparseable manifest lines")
# infer_schema_length=None: midfiller/endfiller are null for long stretches
df = pl.DataFrame(unique, infer_schema_length=None)
df.write_parquet(OUT / "manifest.parquet")

print(f"total {df.height} clips, {df['duration_s'].sum()/3600:.1f} h")
print(df.group_by(["split", "language", "label"]).len().sort(["split", "language", "label"]))
print(df.group_by("split").agg(
    pl.col("midfiller").mean().alias("midfiller_rate"),
    pl.col("synthetic").mean().alias("synthetic_rate"),
))
import shutil
total_gb = sum(f.stat().st_size for f in OUT.rglob("*")) / 1e9
print(f"output size: {total_gb:.1f} GB (expect ~16-17 GB; must stay under ~19.6 GB)")
if total_gb > 19.0:
    print("!! too close to the /kaggle/working limit — lower OTHER_CAP_PER_LANG_LABEL")
'''


# --------------------------------------------------------------------------
# 02_train — GPU
# --------------------------------------------------------------------------

TRAIN_MD = """\
# 02 · Train turn-detection experiment (E1-E6)

**Kaggle settings:** GPU **T4 x2 or P100** · **Internet ON** (downloads
whisper-tiny once) · ~1-2 h per experiment.

**Attach as input datasets** (Add Input):
1. `smart-turn-enhi-prep` — output of notebook 01
2. `hinglish-synth` — the uploaded synthetic Hinglish dataset
3. *(only when resuming)* the previous version's output of THIS notebook

**Run an experiment:** set `EXPERIMENT` in the config cell to one of
`e1_baseline` · `e2_hinglish_aug` · `e3_tinymel_scratch` · `e4_no_pause_aug` ·
`e5_distill` · `e6_full_data`, then *Save Version → Save & Run All*. Repeat per
experiment (one per session).

**Distillation (E5):** `e5_distill` trains TinyMelNet against a frozen Whisper
teacher, so it additionally needs the teacher's `ckpt_best.pt`. Attach the
`turn-detect-ckpt` dataset and set `TEACHER_FROM` to the teacher's run folder
(e.g. `/kaggle/input/turn-detect-ckpt/run_e2_hinglish_aug`);
`python -m tools.push_kaggle train e5_distill --teacher e2_hinglish_aug` stages
that checkpoint and writes the path for you. The run raises immediately if the
teacher is missing rather than silently training without it.

**Time budget:** `TIME_BUDGET_MIN` (default 630 = 10.5 h) makes training stop
itself at the next checkpoint before Kaggle's 12 h session wall. This matters:
a commit run killed by the wall publishes **no output at all**, so a run that
would overrun must end early and leave `ckpt_last.pt` behind to resume from.

**Resume after a kill (or a time-budget stop):** attach the previous run's
output (or the `turn-detect-ckpt` dataset built by
`python -m tools.push_kaggle train <exp> --resume`), set `RESUME_FROM` to its
path (e.g. `/kaggle/input/turn-detect-ckpt/run_e2_hinglish_aug`), run again.
Training continues from the last checkpoint (≤500 steps lost). A missing
checkpoint or a config-hash mismatch now raises instead of silently
restarting from scratch.

**Afterwards:** download `run_<EXPERIMENT>/` (metrics.json, ckpt_best.pt,
model_fp32.onnx, model_int8.onnx) into the repo's `experiments/` folder.

The `turn_detector` package below is auto-generated from the tested repo
sources by `tools/build_notebooks.py` — edit the repo, not the cells.
"""

TRAIN_PIP = "%pip install -q onnx onnxruntime onnxscript polars soundfile"

TRAIN_SETUP = 'import os\nos.makedirs("turn_detector", exist_ok=True)'

TRAIN_CONFIG = r'''
EXPERIMENT = "e1_baseline"   # e1_baseline | e2_hinglish_aug | e3_tinymel_scratch | e4_no_pause_aug | e5_distill | e6_full_data
PREP = "/kaggle/input/smart-turn-enhi-prep/prep"
HINGLISH = "/kaggle/input/hinglish-synth"
RESUME_FROM = ""             # e.g. "/kaggle/input/turn-detect-ckpt/run_e2_hinglish_aug"
TEACHER_FROM = ""            # e5_distill only: run folder holding the teacher's ckpt_best.pt
TIME_BUDGET_MIN = 630        # stop cleanly at 10.5 h; Kaggle kills at 12 h with NO output
'''

TRAIN_RUN = r'''
import shutil, sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from turn_detector.config import EXPERIMENTS
from turn_detector.train import train

cfg = EXPERIMENTS[EXPERIMENT]
out_dir = Path("/kaggle/working") / f"run_{EXPERIMENT}"
out_dir.mkdir(parents=True, exist_ok=True)

# fail in seconds, not sessions: Kaggle's torch has no sm_60 (P100) kernels
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} (sm_{cap[0]}{cap[1]})")
    if cap[0] < 7:
        raise RuntimeError(
            f"{name} (sm_{cap[0]}{cap[1]}) is unsupported by this torch build — "
            f"session must use the T4 (machine_shape NvidiaTeslaT4). Re-push."
        )

if RESUME_FROM:
    # fail loudly: a silent fall-through here burns a whole GPU session
    # restarting from step 0 while the log still says "resuming"
    src_ckpt = Path(RESUME_FROM, "ckpt_last.pt")
    if not src_ckpt.exists():
        raise RuntimeError(
            f"RESUME_FROM={RESUME_FROM!r} has no ckpt_last.pt. Attach the right "
            f"input dataset/notebook output, or set RESUME_FROM = \"\" to start fresh."
        )
    for f in Path(RESUME_FROM).glob("*"):
        if not (out_dir / f.name).exists():
            shutil.copy(f, out_dir / f.name)
    head = torch.load(out_dir / "ckpt_last.pt", map_location="cpu", weights_only=False)
    if head["cfg_hash"] != cfg.config_hash():
        raise RuntimeError(
            f"checkpoint cfg_hash {head['cfg_hash']} != {cfg.config_hash()} for "
            f"{EXPERIMENT}: that checkpoint belongs to a different config. "
            f"Set RESUME_FROM = \"\" to start fresh."
        )
    print(f"resuming {EXPERIMENT} from {RESUME_FROM} @ step {head['step']}")

# Kaggle has used both flat (/kaggle/input/<slug>) and nested
# (/kaggle/input/{datasets,notebooks}/<user>/<slug>) mount layouts, and a
# source kernel's output takes minutes to publish after it completes — so
# resolve mounts by searching rather than trusting a hardcoded path.
import os

def resolve_mount(configured: str, slug: str, marker: str = "manifest.parquet") -> str:
    if Path(configured, marker).exists():
        return configured
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("audio", "__pycache__")]
        if root.count("/") > 8:
            dirs[:] = []
        if marker in files and slug in root:
            hits.append(root)
    if len(hits) == 1:
        print(f"mount for {slug}: {configured} -> {hits[0]}")
        return hits[0]
    listing = "\n".join(sorted(
        str(Path(r) / f) for r, _, fs in os.walk("/kaggle/input")
        for f in fs if r.count("/") <= 6
    )[:60])
    raise RuntimeError(
        f"could not resolve mount for {slug!r} (marker {marker}, hits={hits}).\n"
        f"/kaggle/input contains:\n{listing}\n"
        f"If the source kernel just finished, wait a few minutes and re-push."
    )

PREP = resolve_mount(PREP, "turn-detect-01-data-prep")
HINGLISH = resolve_mount(HINGLISH, "hinglish-synth")
if RESUME_FROM:
    RESUME_FROM = resolve_mount(RESUME_FROM, "run_" + EXPERIMENT,
                                marker="ckpt_last.pt")

# distillation: the teacher's ckpt_best.pt rides in on the turn-detect-ckpt
# dataset. Fail here rather than train a "distilled" student with no teacher.
teacher_dir = None
if getattr(cfg, "kd_teacher", ""):
    if not TEACHER_FROM:
        raise RuntimeError(
            f"{EXPERIMENT} distills from {cfg.kd_teacher!r} but TEACHER_FROM is "
            f"empty. Attach the turn-detect-ckpt dataset and set TEACHER_FROM = "
            f'"/kaggle/input/turn-detect-ckpt/run_{cfg.kd_teacher}", or push with '
            f"python -m tools.push_kaggle train {EXPERIMENT} --teacher {cfg.kd_teacher}"
        )
    teacher_dir = resolve_mount(TEACHER_FROM, "run_" + cfg.kd_teacher,
                                marker="ckpt_best.pt")
    print(f"teacher ({cfg.kd_teacher}): {teacher_dir}")

real = [(f"{PREP}/manifest.parquet", PREP)]
synth = [(f"{HINGLISH}/manifest.parquet", HINGLISH)]
train_sources = real + (synth if cfg.use_hinglish_synth else [])
sources = {
    "train": train_sources,
    "val": train_sources,
    "test": real + synth,   # always evaluate hinglish slice, even for e1
}

metrics = train(cfg, sources, str(out_dir), num_workers=3,
                time_budget_minutes=TIME_BUDGET_MIN, teacher_dir=teacher_dir)

if metrics.get("status") == "time_budget_reached":
    print(
        f"\nPARTIAL RUN: stopped at step {metrics['step']}/{metrics['total_steps']}.\n"
        f"ckpt_last.pt is in run_{EXPERIMENT}/ and this version WILL publish its output.\n"
        f"To continue:  python -m tools.push_kaggle train {EXPERIMENT} --resume\n"
        f"(or manually: attach this version's output, set\n"
        f" RESUME_FROM = \"/kaggle/input/turn-detect-ckpt/run_{EXPERIMENT}\", Save & Run All)\n"
        f"No metrics.json/ONNX yet — those are written by the final run."
    )
'''

TRAIN_SUMMARY = r'''
import json
from pathlib import Path

run_dir = Path("/kaggle/working") / f"run_{EXPERIMENT}"
mpath = run_dir / "metrics.json"
if not mpath.exists():
    print(f"no metrics.json in {run_dir} — partial run (see the cell above). "
          f"Save Version so ckpt_last.pt is published, then resume.")
    print("files:", sorted(p.name for p in run_dir.glob("*")))
else:
    m = json.loads(mpath.read_text())
    head = {k: m[k] for k in ("experiment", "params", "best_val_auc",
                              "threshold", "train_minutes") if k in m}
    print(json.dumps(head, indent=2))
    print(json.dumps(m.get("test", {}), indent=2))
    print(json.dumps(m.get("int8_subset", {}), indent=2))
    print("\nNow: Save Version, then download run_" + EXPERIMENT + "/ into the repo's experiments/ folder.")
'''


NB_METADATA = {
    "kernelspec": {"name": "python3", "display_name": "Python 3",
                   "language": "python"},
    "language_info": {"name": "python"},
}


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip("\n"))


def build_prep() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.update(NB_METADATA)
    nb.cells = [
        nbf.v4.new_markdown_cell(PREP_MD),
        code(PREP_PIP),
        code(PREP_MAIN),
        code(PREP_FINALIZE),
    ]
    return nb


def build_train() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.update(NB_METADATA)
    cells = [
        nbf.v4.new_markdown_cell(TRAIN_MD),
        code(TRAIN_PIP),
        code(TRAIN_SETUP),
    ]
    for mod in MODULES:
        src_text = (SRC / f"{mod}.py").read_text(encoding="utf-8")
        cells.append(code(f"%%writefile turn_detector/{mod}.py\n{src_text}"))
    cells += [code(TRAIN_CONFIG), code(TRAIN_RUN), code(TRAIN_SUMMARY)]
    nb.cells = cells
    return nb


def preserve_ids(nb: nbf.NotebookNode, path: Path) -> nbf.NotebookNode:
    """Reuse the existing file's cell ids, positionally.

    nbformat mints a random id per cell, so an otherwise no-op rebuild (the
    test suite runs one) would rewrite every cell header and bury the real
    change in id churn.
    """
    if not path.exists():
        return nb
    old = nbf.read(str(path), as_version=4)
    for cell, prev in zip(nb.cells, old.cells):
        if "id" in prev:
            cell["id"] = prev["id"]
    return nb


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, nb in [("01_data_prep", build_prep()), ("02_train", build_train())]:
        path = OUT / f"{name}.ipynb"
        nbf.write(preserve_ids(nb, path), str(path))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
