"""Push/monitor/pull Kaggle kernels via the CLI (automated mode).

Usage:
  python -m tools.push_kaggle prep                      # push + run data prep
  python -m tools.push_kaggle train e2_hinglish_aug     # push + run experiment
  python -m tools.push_kaggle train e2_hinglish_aug --resume
  python -m tools.push_kaggle stage-ckpt e2_hinglish_aug  # ckpt round-trip only
  python -m tools.push_kaggle status <prep|train>
  python -m tools.push_kaggle pull <prep|train> [dest]  # download outputs

The train kernel reads its prep data from the prep kernel's latest output
(kernel_sources) and hinglish-synth from dataset_sources.

Resume, concretely: a kernel cannot mount its own output, so the checkpoint
makes a round trip through a dataset. `--resume` (= `stage-ckpt` then push):

  1. `kaggle kernels output turn-detect-02-train` -> notebooks/push/ckpt_stage/
     (which contains run_<experiment>/ckpt_last.pt from the partial run)
  2. write dataset-metadata.json for deveshupathak/turn-detect-ckpt and
     `kaggle datasets create` it (first time) or `version -m` it (afterwards)
  3. push the train kernel with turn-detect-ckpt added to dataset_sources and
     RESUME_FROM = /kaggle/input/turn-detect-ckpt/run_<experiment>

Partial runs happen by design: the notebook's TIME_BUDGET_MIN stops training
at ~10.5 h so the commit publishes its checkpoint instead of being killed at
the 12 h wall with no output at all.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat

USER = "deveshupathak"
ROOT = Path(__file__).parent.parent
NB_DIR = ROOT / "notebooks" / "kaggle"
PUSH_DIR = ROOT / "notebooks" / "push"

KERNELS = {
    "prep": {
        "slug": "turn-detect-01-data-prep",
        "notebook": "01_data_prep.ipynb",
        "metadata": {
            "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": False, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        },
    },
    "train": {
        "slug": "turn-detect-02-train",
        "notebook": "02_train.ipynb",
        "metadata": {
            "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_internet": True,
            # Kaggle's torch build ships sm_70+ kernels only; the P100 default
            # (sm_60) dies with "no kernel image". Pin the T4 (sm_75).
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [f"{USER}/hinglish-synth"],
            "competition_sources": [],
            "kernel_sources": [f"{USER}/turn-detect-01-data-prep"],
        },
    },
}

PREP_MOUNT = "/kaggle/input/notebooks/deveshupathak/turn-detect-01-data-prep/prep"
CKPT_SLUG = "turn-detect-ckpt"
CKPT_DATASET = f"{USER}/{CKPT_SLUG}"
CKPT_STAGE = PUSH_DIR / "ckpt_stage"
CKPT_MOUNT = f"/kaggle/input/datasets/deveshupathak/{CKPT_SLUG}"
# mirrors TIME_BUDGET_MIN in tools/build_notebooks.py's TRAIN_CONFIG cell
TIME_BUDGET_MIN = 630


def run(cmd: list[str]) -> str:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = (result.stdout or "") + (result.stderr or "")
    print(out.strip())
    if result.returncode != 0:
        sys.exit(f"command failed ({result.returncode})")
    return out


def run_ok(cmd: list[str]) -> tuple[bool, str]:
    """Like run(), but a non-zero exit is a legitimate answer, not a failure."""
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = (result.stdout or "") + (result.stderr or "")
    print(out.strip())
    return result.returncode == 0, out


def set_config_cell(nb_path: Path, experiment: str, resume: bool,
                    time_budget_min: int = TIME_BUDGET_MIN):
    nb = nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.source.startswith("EXPERIMENT"):
            resume_path = f"{CKPT_MOUNT}/run_{experiment}" if resume else ""
            cell.source = (
                f'EXPERIMENT = "{experiment}"\n'
                f'PREP = "{PREP_MOUNT}"\n'
                f'HINGLISH = "/kaggle/input/datasets/deveshupathak/hinglish-synth"\n'
                f'RESUME_FROM = "{resume_path}"\n'
                f'TIME_BUDGET_MIN = {time_budget_min}'
            )
            break
    nbformat.write(nb, nb_path)


def stage_ckpt(experiment: str) -> Path:
    """Round-trip the train kernel's latest checkpoint into a Kaggle dataset.

    A kernel cannot mount its own output, so `run_<experiment>/ckpt_last.pt` is
    pulled locally and (re)published as `turn-detect-ckpt`, which the next train
    push attaches as a normal input dataset.
    """
    if CKPT_STAGE.exists():
        shutil.rmtree(CKPT_STAGE)
    CKPT_STAGE.mkdir(parents=True, exist_ok=True)

    run(["kaggle", "kernels", "output", f"{USER}/{KERNELS['train']['slug']}",
         "-p", str(CKPT_STAGE)])

    run_dir = CKPT_STAGE / f"run_{experiment}"
    if not (run_dir / "ckpt_last.pt").exists():
        sys.exit(f"no ckpt_last.pt under {run_dir} — the previous run published "
                 f"no checkpoint, so there is nothing to resume from")

    # keep the upload to the checkpoints; logs/onnx would just bloat it
    for p in list(CKPT_STAGE.rglob("*")):
        if p.is_file() and p.name not in ("ckpt_last.pt", "ckpt_best.pt"):
            p.unlink()
    for p in sorted(CKPT_STAGE.rglob("*"), key=lambda q: -len(q.parts)):
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()

    (CKPT_STAGE / "dataset-metadata.json").write_text(json.dumps({
        "title": CKPT_SLUG,
        "id": CKPT_DATASET,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2), encoding="utf-8")

    exists, out = run_ok(["kaggle", "datasets", "status", CKPT_DATASET])
    missing = (not exists) or any(
        s in out for s in ("403", "404", "not found", "Not Found")
    )
    # --dir-mode zip: Kaggle unpacks it back into run_<experiment>/ on mount
    if missing:
        run(["kaggle", "datasets", "create", "-p", str(CKPT_STAGE),
             "--dir-mode", "zip"])
    else:
        run(["kaggle", "datasets", "version", "-p", str(CKPT_STAGE),
             "-m", f"resume checkpoint for {experiment}", "--dir-mode", "zip"])
    print(f"staged -> {CKPT_DATASET} (mounts at {CKPT_MOUNT}/run_{experiment})")
    return run_dir


def push(kind: str, experiment: str | None = None, resume: bool = False):
    spec = KERNELS[kind]
    if resume:
        assert kind == "train", "--resume only applies to train"
        assert experiment, "train needs an experiment id"
        stage_ckpt(experiment)
    stage = PUSH_DIR / kind
    stage.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "tools.build_notebooks"], check=True)
    nb_src = NB_DIR / spec["notebook"]
    nb_dst = stage / spec["notebook"]
    nb_dst.write_text(nb_src.read_text(encoding="utf-8"), encoding="utf-8")

    meta = dict(spec["metadata"])
    meta["id"] = f"{USER}/{spec['slug']}"
    meta["title"] = spec["slug"]
    meta["code_file"] = spec["notebook"]
    if kind == "train":
        assert experiment, "train needs an experiment id"
        set_config_cell(nb_dst, experiment, resume)
        if resume:
            meta["dataset_sources"] = meta["dataset_sources"] + [CKPT_DATASET]
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    run(["kaggle", "kernels", "push", "-p", str(stage)])
    print(f"\npushed -> https://www.kaggle.com/code/{USER}/{spec['slug']}")


def status(kind: str):
    run(["kaggle", "kernels", "status", f"{USER}/{KERNELS[kind]['slug']}"])


def pull(kind: str, dest: str | None = None):
    slug = KERNELS[kind]["slug"]
    dest = dest or str(ROOT / "experiments" / "kaggle_output" / kind)
    Path(dest).mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", f"{USER}/{slug}", "-p", dest])
    print(f"outputs -> {dest}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    cmd = args[0]
    if cmd == "prep":
        push("prep")
    elif cmd == "train":
        push("train", experiment=args[1], resume="--resume" in args)
    elif cmd == "stage-ckpt":
        stage_ckpt(args[1])
    elif cmd == "status":
        status(args[1])
    elif cmd == "pull":
        pull(args[1], args[2] if len(args) > 2 else None)
    else:
        sys.exit(f"unknown command {cmd}")
