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

MODULES = ["__init__", "features", "augment", "config", "model", "dataset", "train"]


# --------------------------------------------------------------------------
# 01_data_prep — CPU, internet ON
# --------------------------------------------------------------------------

PREP_MD = """\
# 01 · Data prep — filter smart-turn v3.2 to English + Hindi FLAC shards

**Kaggle settings:** CPU (no GPU needed) · **Internet ON** · takes ~1-2 h.

Streams `pipecat-ai/smart-turn-data-v3.2-train` (41 GB, only EN/HI rows are
decoded) and `-test`, resamples to 16 kHz mono, keeps the last 8 s, writes
FLAC + `manifest.parquet` to `/kaggle/working/prep`.

**When it finishes:** *Save Version* → after it completes, create a Kaggle
Dataset from this notebook's output named **`smart-turn-enhi-prep`**
(New Dataset → import from notebook output). Training runs attach that dataset.

Interrupted? Just *Save & Run All* again — already-written clips are skipped.
"""

PREP_PIP = "%pip install -q -U datasets soundfile polars"

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
EN_TRAIN_CAP_PER_LABEL = 17500   # english train rows per label (35k total)
VAL_PCT = 5                      # % of train-source rows held out as val

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
    lang = row["language"]
    if lang not in ("english", "hindi"):
        return 0
    label = int(bool(row["endpoint_bool"]))
    rid = row["id"]
    if rid in done_ids:
        return 0
    if src == "train" and lang == "english" and \
            counts.get(("english", label, "train"), 0) >= EN_TRAIN_CAP_PER_LABEL:
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
rows = [json.loads(l) for l in open(OUT / "manifest.jsonl")]
seen, unique = set(), []
for r in rows:
    if r["id"] not in seen:
        seen.add(r["id"])
        unique.append(r)
df = pl.DataFrame(unique)
df.write_parquet(OUT / "manifest.parquet")

print(f"total {df.height} clips, {df['duration_s'].sum()/3600:.1f} h")
print(df.group_by(["split", "language", "label"]).len().sort(["split", "language", "label"]))
print(df.group_by("split").agg(
    pl.col("midfiller").mean().alias("midfiller_rate"),
    pl.col("synthetic").mean().alias("synthetic_rate"),
))
import shutil
total_gb = sum(f.stat().st_size for f in OUT.rglob("*")) / 1e9
print(f"output size: {total_gb:.1f} GB (must stay under ~19 GB)")
'''


# --------------------------------------------------------------------------
# 02_train — GPU
# --------------------------------------------------------------------------

TRAIN_MD = """\
# 02 · Train turn-detection experiment (E1-E4)

**Kaggle settings:** GPU **T4 x2 or P100** · **Internet ON** (downloads
whisper-tiny once) · ~1-2 h per experiment.

**Attach as input datasets** (Add Input):
1. `smart-turn-enhi-prep` — output of notebook 01
2. `hinglish-synth` — the uploaded synthetic Hinglish dataset
3. *(only when resuming)* the previous version's output of THIS notebook

**Run an experiment:** set `EXPERIMENT` in the config cell to one of
`e1_baseline` · `e2_hinglish_aug` · `e3_tinymel_scratch` · `e4_no_pause_aug`,
then *Save Version → Save & Run All*. Repeat per experiment (one per session).

**Resume after a kill:** attach the previous run's output, set `RESUME_FROM`
to its path (e.g. `/kaggle/input/02-train/run_e2_hinglish_aug`), run again.
Training continues from the last checkpoint (≤500 steps lost).

**Afterwards:** download `run_<EXPERIMENT>/` (metrics.json, ckpt_best.pt,
model_fp32.onnx, model_int8.onnx) into the repo's `experiments/` folder.

The `turn_detector` package below is auto-generated from the tested repo
sources by `tools/build_notebooks.py` — edit the repo, not the cells.
"""

TRAIN_PIP = "%pip install -q onnx onnxruntime onnxscript polars soundfile"

TRAIN_SETUP = 'import os\nos.makedirs("turn_detector", exist_ok=True)'

TRAIN_CONFIG = r'''
EXPERIMENT = "e1_baseline"   # e1_baseline | e2_hinglish_aug | e3_tinymel_scratch | e4_no_pause_aug
PREP = "/kaggle/input/smart-turn-enhi-prep/prep"
HINGLISH = "/kaggle/input/hinglish-synth"
RESUME_FROM = ""             # e.g. "/kaggle/input/02-train/run_e2_hinglish_aug"
'''

TRAIN_RUN = r'''
import shutil, sys
from pathlib import Path

sys.path.insert(0, ".")
from turn_detector.config import EXPERIMENTS
from turn_detector.train import train

cfg = EXPERIMENTS[EXPERIMENT]
out_dir = Path("/kaggle/working") / f"run_{EXPERIMENT}"
out_dir.mkdir(parents=True, exist_ok=True)

if RESUME_FROM and Path(RESUME_FROM, "ckpt_last.pt").exists():
    for f in Path(RESUME_FROM).glob("*"):
        if not (out_dir / f.name).exists():
            shutil.copy(f, out_dir / f.name)
    print(f"copied previous run from {RESUME_FROM}")

real = [(f"{PREP}/manifest.parquet", PREP)]
synth = [(f"{HINGLISH}/manifest.parquet", HINGLISH)]
train_sources = real + (synth if cfg.use_hinglish_synth else [])
sources = {
    "train": train_sources,
    "val": train_sources,
    "test": real + synth,   # always evaluate hinglish slice, even for e1
}

metrics = train(cfg, sources, str(out_dir), num_workers=3)
'''

TRAIN_SUMMARY = r'''
import json
from pathlib import Path

m = json.loads((Path("/kaggle/working") / f"run_{EXPERIMENT}" / "metrics.json").read_text())
print(json.dumps({k: m[k] for k in ("experiment", "params", "best_val_auc",
                                    "threshold", "train_minutes")}, indent=2))
print(json.dumps(m["test"], indent=2))
print(json.dumps(m.get("int8_subset", {}), indent=2))
print("\nNow: Save Version, then download run_" + EXPERIMENT + "/ into the repo's experiments/ folder.")
'''


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip("\n"))


def build_prep() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(PREP_MD),
        code(PREP_PIP),
        code(PREP_MAIN),
        code(PREP_FINALIZE),
    ]
    return nb


def build_train() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, nb in [("01_data_prep", build_prep()), ("02_train", build_train())]:
        path = OUT / f"{name}.ipynb"
        nbf.write(nb, str(path))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
