# Kaggle Runbook — exact steps for every manual action

You need a **phone-verified Kaggle account** (Settings → Phone verification) to
use GPUs and Internet-enabled notebooks.

## Step 1 — Upload the Hinglish synthetic dataset (once, ~5 min)

1. Go to kaggle.com → **Create → New Dataset**.
2. Drag in `synth/output/hinglish-synth.zip` (122 MB). Kaggle auto-extracts zips.
3. Title: **hinglish-synth** (URL slug must be `hinglish-synth`). → **Create**.
4. Once created, open the dataset page and confirm you see `manifest.parquet`
   and an `audio/` folder at the top level (not nested inside another folder).
   If they're nested one level deep, note the folder name — you'll adjust the
   `HINGLISH` path in the training notebook config cell accordingly.

## Step 2 — Run data prep (once, CPU, ~1–2 h)

1. **Create → New Notebook** → File → Import Notebook → upload
   `notebooks/kaggle/01_data_prep.ipynb`.
2. Right panel → **Session options**: Accelerator = **None**,
   Internet = **ON**.
3. **Save Version → Save & Run All (Commit)**. You can close the tab; it runs
   in the background (check *Your Work → notebook → Logs*).
4. When finished: open the notebook's latest version → **Output** tab →
   **New Dataset** (create dataset from output). Name it exactly
   **`smart-turn-enhi-prep`**.
   - If it was interrupted (rare): just Save & Run All again — it resumes,
     already-written clips are skipped.

## Step 3 — Training runs (GPU, ~1–2 h each, run sequentially)

For each experiment `e1_baseline` → `e2_hinglish_aug` → `e3_tinymel_scratch`
→ `e4_no_pause_aug`:

1. **Create → New Notebook** → import `notebooks/kaggle/02_train.ipynb`
   (first time only; afterwards just edit the same notebook).
2. Right panel → **Add Input** → *Datasets → Your Work* → attach
   `smart-turn-enhi-prep` **and** `hinglish-synth`.
3. Session options: Accelerator = **GPU T4 x2** (or P100), Internet = **ON**.
4. In the **config cell**, set `EXPERIMENT = "<experiment id>"`.
5. **Save Version → Save & Run All (Commit)**.
6. When done: version page → **Output** tab → download `run_<experiment>/`
   (contains `metrics.json`, `ckpt_best.pt`, `model_fp32.onnx`,
   `model_int8.onnx`) and put it under `experiments/` in this repo, e.g.
   `experiments/run_e1_baseline/metrics.json`.

### If a GPU session dies mid-run

1. Add Input → *Your Work → Notebooks* → attach the previous version's output
   of the training notebook.
2. Set `RESUME_FROM = "/kaggle/input/<notebook-slug>/run_<experiment>"` in the
   config cell, Save & Run All. It continues from the last checkpoint
   (≤500 training steps lost). Set `RESUME_FROM = ""` again for fresh runs.

## GPU budget

~1–2 h per experiment × 4 experiments ≈ 6–8 GPU hours total, well inside the
~30 h/week quota. Sessions auto-terminate at 12 h — one experiment always fits.

## What to report back

After each run, drop the `run_<experiment>/` folder into `experiments/` and
say which experiments are done — analysis (Phase 3) starts from
`experiments/run_*/metrics.json`.
