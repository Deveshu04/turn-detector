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
# 01 · Data prep — filter smart-turn v3.2 to English + Hindi FLAC shards

**Kaggle settings:** CPU (no GPU needed) · **Internet ON** · takes ~1-2 h.

Streams `pipecat-ai/smart-turn-data-v3.2-train` (41 GB, only EN/HI rows are
decoded) and `-test`, resamples to 16 kHz mono, keeps the last 8 s, writes
FLAC + `manifest.parquet` to `/kaggle/working/prep`.

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
EN_TRAIN_CAP_PER_LABEL = 17500   # english train rows per label (35k total)
VAL_PCT = 5                      # % of train-source rows held out as val

# smart-turn v3.2 tags language with ISO-639-3 codes; the manifest (and every
# downstream consumer) uses the long names, so normalise once here and key
# EVERYTHING below -- rec, counters, resume rebuild -- on the long name.
LANG_MAP = {"eng": "english", "hin": "hindi"}

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
    if row["language"] not in ("eng", "hin"):
        return 0
    lang = LANG_MAP[row["language"]]
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
EXPERIMENT = "e1_baseline"   # e1_baseline | e2_hinglish_aug | e3_tinymel_scratch | e4_no_pause_aug
PREP = "/kaggle/input/smart-turn-enhi-prep/prep"
HINGLISH = "/kaggle/input/hinglish-synth"
RESUME_FROM = ""             # e.g. "/kaggle/input/turn-detect-ckpt/run_e2_hinglish_aug"
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

real = [(f"{PREP}/manifest.parquet", PREP)]
synth = [(f"{HINGLISH}/manifest.parquet", HINGLISH)]
train_sources = real + (synth if cfg.use_hinglish_synth else [])
sources = {
    "train": train_sources,
    "val": train_sources,
    "test": real + synth,   # always evaluate hinglish slice, even for e1
}

metrics = train(cfg, sources, str(out_dir), num_workers=3,
                time_budget_minutes=TIME_BUDGET_MIN)

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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, nb in [("01_data_prep", build_prep()), ("02_train", build_train())]:
        path = OUT / f"{name}.ipynb"
        nbf.write(nb, str(path))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
