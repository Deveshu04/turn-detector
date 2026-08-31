# Kaggle Runbook: CLI workflow

Everything below is driven from the repo with `tools/push_kaggle.py`. The
manual click-path still works and is kept as an appendix at the bottom.

## Prerequisites (once)

- A **phone-verified Kaggle account** (Settings → Phone verification), required
  for GPU and Internet-enabled notebooks.
- API token: Kaggle → Settings → **Create New Token** → save `kaggle.json` to
  `%USERPROFILE%\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json`.
- `pip install kaggle` in the venv.

Notebooks are generated from the repo sources, so **always** regenerate before
pushing (`push_kaggle` does this for you):

```
.venv/Scripts/python.exe -m tools.build_notebooks
```

## Step 1: Upload the Hinglish synthetic dataset (once, ~5 min)

Build it, then create the dataset from the extracted folder:

```
.venv/Scripts/python.exe -m synth.package_kaggle       # -> synth/output/hinglish-synth.zip
kaggle datasets create -p synth/output/kaggle_upload
```

`synth/output/kaggle_upload/` holds `dataset-metadata.json`
(`"id": "deveshupathak/hinglish-synth"`) next to a copy of `hinglish-synth.zip`;
Kaggle auto-extracts the zip on upload. Later refreshes use
`kaggle datasets version -p synth/output/kaggle_upload -m "<msg>"`.

Confirm on the dataset page that `manifest.parquet` and `audio/` sit at the top
level. If they end up nested one level deeper, adjust `HINGLISH` in
`tools/push_kaggle.py` (`set_config_cell`).

## Step 2: Data prep (once, CPU, internet ON, ~1–2 h)

```
.venv/Scripts/python.exe -m tools.push_kaggle prep
.venv/Scripts/python.exe -m tools.push_kaggle status prep     # poll
```

`push` uploads `notebooks/kaggle/01_data_prep.ipynb` as the kernel
`turn-detect-01-data-prep` and starts a commit run. The train kernel mounts this
kernel's latest output directly via `kernel_sources`, so **no dataset needs to be
created from it**.

Prep writes its ~122k clips into 64 ZIP shards (`prep/shards/shard_NN.zip`)
rather than loose files, because Kaggle's kernel-output publishing silently
fails past ~100k files (the version "succeeds" with an 845-byte empty
`_output_.zip`); the train notebook extracts the shards to `/tmp/prep_audio`
and uses that as the audio root, while `manifest.parquet` still comes from the
mount. The final prep cell prints the shard/entry counts and asserts every
manifest row has a shard entry.

If the prep run is interrupted, just push it again. A committed batch run starts
from an empty `/kaggle/working`, so there is nothing to resume from; prep
normally completes inside a single session.

## Step 3: Training runs (GPU, sequential)

For each experiment `e1_baseline` → `e2_hinglish_aug` → `e3_tinymel_scratch`
→ `e4_no_pause_aug`:

```
.venv/Scripts/python.exe -m tools.push_kaggle train e2_hinglish_aug
.venv/Scripts/python.exe -m tools.push_kaggle status train
.venv/Scripts/python.exe -m tools.push_kaggle pull train experiments/run_e2_hinglish_aug
```

`push_kaggle train` regenerates the notebook, rewrites the config cell
(`EXPERIMENT`, `PREP`, `HINGLISH`, `RESUME_FROM`, `TIME_BUDGET_MIN`), attaches
`hinglish-synth` + the prep kernel, and commits with GPU + internet on.

`pull` downloads `run_<experiment>/` (`metrics.json`, `ckpt_best.pt`,
`model_fp32.onnx`, `model_int8.onnx`); put it under `experiments/`; Phase 3
analysis reads `experiments/run_*/metrics.json`.

### Time budget and resume

Kaggle kills a commit run at the 12 h wall and publishes **no output at all**,
which would lose the entire session. So `train()` takes
`time_budget_minutes` (`TIME_BUDGET_MIN = 630`, i.e. 10.5 h): at the next
checkpoint past the budget it saves `ckpt_last.pt`, prints
`TIME BUDGET REACHED at step X/Y`, and returns
`{"status": "time_budget_reached", ...}` **without** running the final eval or
ONNX export. The version still completes, so its output is published.

To continue such a run (or one that died for any other reason):

```
.venv/Scripts/python.exe -m tools.push_kaggle train e2_hinglish_aug --resume
```

That does the checkpoint round trip a kernel cannot do for itself:

1. `kaggle kernels output turn-detect-02-train -p notebooks/push/ckpt_stage/`
2. publishes that folder as the dataset `deveshupathak/turn-detect-ckpt`
   (`datasets create` the first time, `datasets version -m` afterwards)
3. pushes the train kernel with `turn-detect-ckpt` in `dataset_sources` and
   `RESUME_FROM = "/kaggle/input/turn-detect-ckpt/run_<experiment>"`

Training picks up from `ckpt_last.pt` (≤500 steps lost). The notebook now
**hard-fails** if `RESUME_FROM` has no `ckpt_last.pt`, or if the checkpoint's
`cfg_hash` doesn't match the experiment config. Previously it silently
restarted from step 0 while the log claimed it was resuming. `config_hash()`
ignores `notes` and `checkpoint_every_steps`, so cosmetic edits don't invalidate
a checkpoint.

Run `stage-ckpt <experiment>` alone if you only want the dataset refreshed:

```
.venv/Scripts/python.exe -m tools.push_kaggle stage-ckpt e2_hinglish_aug
```

## GPU budget

~1–2 h per experiment × 4 experiments ≈ 6–8 GPU hours, well inside the ~30 h/week
quota. Sessions auto-terminate at 12 h and the 10.5 h budget keeps a partial run
recoverable.

## What to report back

After each run, drop `run_<experiment>/` into `experiments/` and say which
experiments are done.

---

## Appendix: manual fallback (no CLI)

If the API token or CLI is unavailable:

1. **Hinglish dataset:** kaggle.com → Create → New Dataset → drag in
   `synth/output/hinglish-synth.zip` → title **hinglish-synth**.
2. **Prep:** Create → New Notebook → File → Import Notebook →
   `notebooks/kaggle/01_data_prep.ipynb`. Session options: Accelerator = None,
   Internet = ON. **Save Version → Save & Run All (Commit)**. When it finishes,
   Output tab → New Dataset → name it **`smart-turn-enhi-prep`**, and set
   `PREP = "/kaggle/input/smart-turn-enhi-prep/prep"` in the train notebook.
3. **Train:** import `notebooks/kaggle/02_train.ipynb`, Add Input →
   `smart-turn-enhi-prep` + `hinglish-synth`, Accelerator = GPU T4 x2 (or P100),
   Internet = ON, set `EXPERIMENT` in the config cell, Save & Run All.
4. **Resume manually:** Add Input → Your Work → Notebooks → attach the previous
   version's output, set
   `RESUME_FROM = "/kaggle/input/<notebook-slug>/run_<experiment>"`,
   Save & Run All. Set it back to `""` for fresh runs.
