"""Push/monitor/pull Kaggle kernels via the CLI (automated mode).

Usage:
  python -m tools.push_kaggle prep                      # push + run data prep
  python -m tools.push_kaggle train e2_hinglish_aug     # push + run experiment
  python -m tools.push_kaggle train e2_hinglish_aug --resume
  python -m tools.push_kaggle train e5_distill --teacher e2_hinglish_aug
  python -m tools.push_kaggle train e5_distill --teacher e2_hinglish_aug --no-stage
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

`--teacher <experiment>` (for E5 distillation) rides the same round trip: it
publishes that experiment's `ckpt_best.pt` to turn-detect-ckpt and points the
notebook's TEACHER_FROM at `/kaggle/input/turn-detect-ckpt/run_<teacher>`.

  IMPORTANT WRINKLE: `kaggle kernels output` only ever returns the *latest
  completed version* of the train kernel. So `--teacher X` works when that
  latest version is the X run (stage the teacher right after X finishes, before
  running anything else) — otherwise X's checkpoint is simply not in the pull
  and staging fails. If X's checkpoint is already inside turn-detect-ckpt from
  an earlier staging, skip the round trip with `--no-stage`: the notebook is
  still wired to the teacher and the dataset is still attached.
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
                    teacher: str | None = None,
                    time_budget_min: int = TIME_BUDGET_MIN):
    nb = nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.source.startswith("EXPERIMENT"):
            resume_path = f"{CKPT_MOUNT}/run_{experiment}" if resume else ""
            teacher_path = f"{CKPT_MOUNT}/run_{teacher}" if teacher else ""
            cell.source = (
                f'EXPERIMENT = "{experiment}"\n'
                f'PREP = "{PREP_MOUNT}"\n'
                f'HINGLISH = "/kaggle/input/datasets/deveshupathak/hinglish-synth"\n'
                f'RESUME_FROM = "{resume_path}"\n'
                f'TEACHER_FROM = "{teacher_path}"\n'
                f'TIME_BUDGET_MIN = {time_budget_min}'
            )
            break
    nbformat.write(nb, nb_path)


def stage_ckpt(experiment: str, need: str = "ckpt_last.pt",
               extra: dict[str, str] | None = None) -> Path:
    """Round-trip the train kernel's latest checkpoints into a Kaggle dataset.

    A kernel cannot mount its own output, so `run_<experiment>/<need>` is pulled
    locally and (re)published as `turn-detect-ckpt`, which the next train push
    attaches as a normal input dataset.

    Every `run_*/ckpt_*.pt` present in the pulled output is kept — a resume push
    that also carries a distillation teacher needs two different runs in the
    same dataset. Only the *requested* runs (`experiment` plus anything in
    `extra`, as {run_name: required file}) are required to be there.
    """
    required = {experiment: need}
    required.update(extra or {})

    if CKPT_STAGE.exists():
        shutil.rmtree(CKPT_STAGE)
    CKPT_STAGE.mkdir(parents=True, exist_ok=True)

    run(["kaggle", "kernels", "output", f"{USER}/{KERNELS['train']['slug']}",
         "-p", str(CKPT_STAGE)])

    for name, marker in required.items():
        if not (CKPT_STAGE / f"run_{name}" / marker).exists():
            found = sorted(p.parent.name + "/" + p.name
                           for p in CKPT_STAGE.rglob("ckpt_*.pt"))
            sys.exit(
                f"no {marker} under run_{name}/ in the train kernel's latest "
                f"output (found: {found or 'nothing'}).\n"
                f"`kaggle kernels output` only returns the LATEST version, so "
                f"stage a run's checkpoint while that run is the latest one. "
                f"If it is already inside {CKPT_DATASET} from an earlier "
                f"staging, re-push with --no-stage."
            )
    run_dir = CKPT_STAGE / f"run_{experiment}"

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

    kept = sorted(p.relative_to(CKPT_STAGE).as_posix()
                  for p in CKPT_STAGE.rglob("ckpt_*.pt"))
    print("staging:", ", ".join(kept))

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


def push(kind: str, experiment: str | None = None, resume: bool = False,
         teacher: str | None = None, stage_teacher: bool = True):
    spec = KERNELS[kind]
    if resume or teacher:
        assert kind == "train", "--resume/--teacher only apply to train"
        assert experiment, "train needs an experiment id"
    if teacher:
        # the notebook resolves the mount by cfg.kd_teacher, so a mismatch here
        # trains against one teacher and records the other in metrics.json
        sys.path.insert(0, str(ROOT / "src"))
        from turn_detector.config import EXPERIMENTS
        declared = EXPERIMENTS[experiment].kd_teacher
        if declared != teacher:
            sys.exit(
                f"--teacher {teacher} but {experiment}'s kd_teacher is "
                f"{declared!r}. Set kd_teacher in src/turn_detector/config.py "
                f"(this changes the config hash — correct, it is a different "
                f"experiment) and re-push."
            )
    if resume:
        # one round trip carries both: the student's own ckpt_last and,
        # if distilling, the teacher's ckpt_best
        stage_ckpt(experiment,
                   extra={teacher: "ckpt_best.pt"} if teacher else None)
    elif teacher and stage_teacher:
        stage_ckpt(teacher, need="ckpt_best.pt")
    elif teacher:
        print(f"--no-stage: assuming run_{teacher}/ckpt_best.pt is already in "
              f"{CKPT_DATASET}")
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
        set_config_cell(nb_dst, experiment, resume, teacher)
        if resume or teacher:
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
        teacher = None
        if "--teacher" in args:
            i = args.index("--teacher") + 1
            if i >= len(args) or args[i].startswith("--"):
                sys.exit("--teacher needs an experiment name, e.g. "
                         "--teacher e2_hinglish_aug")
            teacher = args[i]
        push("train", experiment=args[1], resume="--resume" in args,
             teacher=teacher, stage_teacher="--no-stage" not in args)
    elif cmd == "stage-ckpt":
        stage_ckpt(args[1])
    elif cmd == "status":
        status(args[1])
    elif cmd == "pull":
        pull(args[1], args[2] if len(args) > 2 else None)
    else:
        sys.exit(f"unknown command {cmd}")
