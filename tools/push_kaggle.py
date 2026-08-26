"""Push/monitor/pull Kaggle kernels via the CLI (automated mode).

Usage:
  python -m tools.push_kaggle prep                      # push + run data prep
  python -m tools.push_kaggle train e2_hinglish_aug     # push + run experiment
  python -m tools.push_kaggle train e2_hinglish_aug --resume
  python -m tools.push_kaggle status <prep|train>
  python -m tools.push_kaggle pull <prep|train> [dest]  # download outputs

The train kernel reads its prep data from the prep kernel's latest output
(kernel_sources) and hinglish-synth from dataset_sources. --resume points
RESUME_FROM at the train kernel's own previous output (attached via
kernel_sources self-reference is not allowed, so we attach the previous
output by downloading is unnecessary — Kaggle mounts the named kernel's
LATEST successful output; for resume we instead re-attach our own last
version via dataset produced from it — simpler: the checkpoint copy happens
from /kaggle/input/<self-slug> when Kaggle allows self-source; if not, pull
outputs locally and re-upload as dataset turn-detect-ckpt).
"""

import json
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
            "dataset_sources": [f"{USER}/hinglish-synth"],
            "competition_sources": [],
            "kernel_sources": [f"{USER}/turn-detect-01-data-prep"],
        },
    },
}

PREP_MOUNT = "/kaggle/input/turn-detect-01-data-prep/prep"
CKPT_DATASET = f"{USER}/turn-detect-ckpt"


def run(cmd: list[str]) -> str:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = (result.stdout or "") + (result.stderr or "")
    print(out.strip())
    if result.returncode != 0:
        sys.exit(f"command failed ({result.returncode})")
    return out


def set_config_cell(nb_path: Path, experiment: str, resume: bool):
    nb = nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.source.startswith("EXPERIMENT"):
            resume_path = f"/kaggle/input/turn-detect-ckpt/run_{experiment}" if resume else ""
            cell.source = (
                f'EXPERIMENT = "{experiment}"\n'
                f'PREP = "{PREP_MOUNT}"\n'
                f'HINGLISH = "/kaggle/input/hinglish-synth"\n'
                f'RESUME_FROM = "{resume_path}"'
            )
            break
    nbformat.write(nb, nb_path)


def push(kind: str, experiment: str | None = None, resume: bool = False):
    spec = KERNELS[kind]
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
    elif cmd == "status":
        status(args[1])
    elif cmd == "pull":
        pull(args[1], args[2] if len(args) > 2 else None)
    else:
        sys.exit(f"unknown command {cmd}")
