"""Qualitative / error analysis for a trained turn-detection run.

Uses ONLY local data: the synthetic Hinglish set in synth/output (manifest.jsonl
+ the template-hashed split from synth.package_kaggle.split_of) and the six demo
clips in demo/examples.

Torch-free: onnxruntime + numpy + soundfile + matplotlib (Agg) only.

Writes into <run_dir>/analysis/:
  worst_errors.md     20 worst misclassifications + per-kind accuracy
  prob_curves.png     sliding P(complete) for the 6 demo clips
  threshold_sweep.png accuracy + F1 vs decision threshold on the test split

Run: python -m tools.error_analysis --run models
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from synth.package_kaggle import split_of  # noqa: E402
from turn_detector.common import SAMPLE_RATE  # noqa: E402
from turn_detector.infer import TurnDetector  # noqa: E402

SYNTH_OUT = ROOT / "synth" / "output"
EXAMPLES = ROOT / "demo" / "examples"
KINDS = ["full", "cut", "tail_conj", "tail_filler", "midfiller_full"]
SWEEP = np.round(np.arange(0.05, 0.9501, 0.05), 2)

# ---- palette (light surface). Slots 1+2 pass the lightness band, the chroma
# floor, CVD ΔE 24.7, normal ΔE 33.6 and contrast >= 3:1 against #fcfcfb ------
SURFACE = "#fcfcfb"
SERIES_1 = "#2a78d6"   # blue:   accuracy / expected-complete
SERIES_2 = "#eb6834"   # orange: F1 / expected-incomplete
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_test_rows() -> list[dict]:
    """Deduped, existing-on-disk hinglish clips whose sentence hashes to test."""
    manifest = SYNTH_OUT / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"no local synth manifest at {manifest}")
    rows, seen = [], set()
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["id"] in seen or not (SYNTH_OUT / r["path"]).exists():
                continue
            seen.add(r["id"])
            if split_of(r["sentence_id"]) == "test":
                rows.append(r)
    rows.sort(key=lambda r: r["id"])           # deterministic ordering
    return rows


def load_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:                       # linear resample; torch-free
        n = int(len(wav) * SAMPLE_RATE / sr)
        wav = np.interp(np.linspace(0, len(wav) - 1, n),
                        np.arange(len(wav)), wav).astype(np.float32)
    return np.ascontiguousarray(wav, dtype=np.float32)


# --------------------------------------------------------------------------
# metrics (no sklearn in this env, so small numpy implementations)
# --------------------------------------------------------------------------
def roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Mann-Whitney U statistic with mid-ranks for ties."""
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(probs, kind="mergesort")
    ranks = np.empty(len(probs), dtype=np.float64)
    s = probs[order]
    i = 0
    while i < len(s):                           # average ranks within each tie
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def metrics_at(labels: np.ndarray, probs: np.ndarray, thr: float) -> dict:
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "acc": float((pred == labels).mean()) if len(labels) else float("nan"),
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "precision": prec, "recall": rec,
    }


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------
def md_cell(text: str, limit: int = 60) -> str:
    t = " ".join(str(text).split())
    if len(t) > limit:
        t = t[: limit - 1] + "…"
    return t.replace("|", "\\|")


def write_worst_errors(rows: list[dict], thr: float, overall: dict, auc: float,
                       out: Path, run_dir: Path, thr_src: str) -> None:
    labels = np.array([r["label"] for r in rows])
    probs = np.array([r["prob"] for r in rows])

    # |prob - label|; ties broken by id so the table is reproducible
    worst = sorted(rows, key=lambda r: (-abs(r["prob"] - r["label"]), r["id"]))[:20]

    L = [
        "# Worst errors: local Hinglish test split",
        "",
        f"- run: `{run_dir.as_posix()}`",
        f"- model: `{(run_dir / 'model_int8.onnx').as_posix()}`",
        f"- threshold: **{thr:.2f}** (from {thr_src})",
        f"- clips: **{len(rows)}** "
        f"(complete={int(labels.sum())}, incomplete={int((1 - labels).sum())})",
        f"- accuracy **{overall['acc']:.3f}** · AUC **{auc:.3f}** · "
        f"F1 **{overall['f1']:.3f}** "
        f"(P {overall['precision']:.3f} / R {overall['recall']:.3f})",
        "",
        "## 20 worst misclassifications",
        "",
        "Ranked by |prob − label|. `label` 1 = turn complete, 0 = still speaking.",
        "",
        "| # | id | kind | voice | dur (s) | label | prob | text |",
        "|---|----|------|-------|--------:|------:|-----:|------|",
    ]
    for i, r in enumerate(worst, 1):
        L.append(
            f"| {i} | `{r['id']}` | {r['kind']} | {r['voice']} | "
            f"{r['duration_s']:.2f} | {r['label']} | {r['prob']:.3f} | "
            f"{md_cell(r.get('text', ''))} |"
        )

    L += [
        "",
        "## Per-kind accuracy",
        "",
        f"At threshold {thr:.2f}.",
        "",
        "| kind | n | label | accuracy | mean prob | errors |",
        "|------|--:|------:|---------:|----------:|-------:|",
    ]
    present = [k for k in KINDS if any(r["kind"] == k for r in rows)]
    present += sorted({r["kind"] for r in rows} - set(KINDS))
    for kind in present:
        sub = [r for r in rows if r["kind"] == kind]
        kl = np.array([r["label"] for r in sub])
        kp = np.array([r["prob"] for r in sub])
        acc = float(((kp >= thr).astype(int) == kl).mean())
        lab = str(kl[0]) if len(set(kl.tolist())) == 1 else "mixed"
        L.append(f"| {kind} | {len(sub)} | {lab} | {acc:.3f} | {kp.mean():.3f} "
                 f"| {int(round((1 - acc) * len(sub)))} |")
    L.append(f"| **all** | {len(rows)} | mixed | {overall['acc']:.3f} | "
             f"{probs.mean():.3f} | "
             f"{int(round((1 - overall['acc']) * len(rows)))} |")
    L.append("")
    out.write_text("\n".join(L), encoding="utf-8")


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)


def plot_prob_curves(det: TurnDetector, thr: float, out: Path) -> int:
    files = sorted(EXAMPLES.glob("*.flac"))
    if not files:
        print(f"[warn] no clips in {EXAMPLES} - skipping prob_curves.png")
        return 0

    curves = []
    for path in files:
        pts = det.sliding_probs(load_wav(path), step_s=0.24)
        curves.append(([p["t"] for p in pts], [p["prob"] for p in pts]))
    # small multiples share one x scale so panels stay comparable
    x_max = max(ts[-1] for ts, _ in curves) * 1.22   # room for the end labels

    ncols = 3
    nrows = int(np.ceil(len(files) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.7 * nrows),
                             dpi=140, sharey=True, sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for ax, path, (ts, ps) in zip(axes.ravel(), files, curves):
        # expected class read off the filename - identity, not rank
        color = SERIES_1 if path.stem.startswith("complete") else SERIES_2

        style_axes(ax)
        ax.axhline(thr, linestyle="--", linewidth=1.0, color=MUTED, zorder=1)
        ax.plot(ts, ps, color=color, linewidth=2.0, solid_capstyle="round",
                solid_joinstyle="round", zorder=3)
        ax.plot(ts[-1:], ps[-1:], marker="o", markersize=5.5, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        # one direct label per panel: the endpoint
        ax.annotate(f"{ps[-1]:.2f}", (ts[-1], ps[-1]), textcoords="offset points",
                    xytext=(7, 0), va="center", fontsize=8, color=TEXT_SECONDARY)
        ax.set_title(path.name, loc="left", fontsize=9.5, color=TEXT_PRIMARY,
                     pad=6)
        ax.set_ylim(-0.04, 1.08)
        ax.set_xlim(0, x_max)

    for ax in axes.ravel()[len(files):]:
        ax.set_visible(False)
    # x labels go on the bottom-most *visible* panel of each column (sharex
    # would otherwise hide them when the last row is partly empty)
    for col in range(ncols):
        visible = [axes[r][col] for r in range(nrows) if axes[r][col].get_visible()]
        if visible:
            ax = visible[-1]
            ax.set_xlabel("time heard so far (s)", fontsize=8.5,
                          color=TEXT_SECONDARY)
            ax.tick_params(labelbottom=True)
    for row in axes:
        row[0].set_ylabel("P(complete)", fontsize=8.5, color=TEXT_SECONDARY)

    handles = [
        plt.Line2D([], [], color=SERIES_1, linewidth=2.0,
                   label="expected: complete"),
        plt.Line2D([], [], color=SERIES_2, linewidth=2.0,
                   label="expected: incomplete"),
        plt.Line2D([], [], color=MUTED, linewidth=1.0, linestyle="--",
                   label=f"threshold {thr:.2f}"),
    ]
    top = 1.0 - 0.62 / fig.get_figheight()      # reserve a fixed title band
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.suptitle("P(turn complete) as each demo clip streams in",
                 x=0.01, ha="left", y=1.0 - 0.14 / fig.get_figheight(),
                 va="top", fontsize=12.5, color=TEXT_PRIMARY)
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False,
               fontsize=8.5, labelcolor=TEXT_SECONDARY,
               bbox_to_anchor=(0.01, 1.0 - 0.42 / fig.get_figheight()))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return len(files)


def _ha(x: float) -> str:
    """Keep an annotation inside the 0..1 threshold axis."""
    return "left" if x < 0.22 else ("right" if x > 0.78 else "center")


def plot_threshold_sweep(rows: list[dict], thr: float, out: Path) -> None:
    labels = np.array([r["label"] for r in rows])
    probs = np.array([r["prob"] for r in rows])
    accs = np.array([metrics_at(labels, probs, t)["acc"] for t in SWEEP])
    f1s = np.array([metrics_at(labels, probs, t)["f1"] for t in SWEEP])

    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    ax.axvline(thr, linestyle="--", linewidth=1.0, color=MUTED, zorder=1)
    ax.annotate(f"run threshold {thr:.2f}", (thr, 1.055), fontsize=8.5,
                color=TEXT_SECONDARY, ha=_ha(thr), va="top")

    peaks = []
    for series, color, name in ((accs, SERIES_1, "accuracy"),
                                (f1s, SERIES_2, "F1 (complete)")):
        ax.plot(SWEEP, series, color=color, linewidth=2.0,
                solid_capstyle="round", label=name, zorder=3)
        # direct-label the extreme only
        i = int(np.argmax(series))
        x, y = float(SWEEP[i]), float(series[i])
        ax.plot([x], [y], marker="o", markersize=5.5, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        # nudge the second label below its point when the two peaks collide
        below = any(abs(x - px) < 0.16 and abs(y - py) < 0.08 for px, py in peaks)
        ax.annotate(f"{name} max {y:.3f} @ {x:.2f}", (x, y),
                    textcoords="offset points", xytext=(0, -14 if below else 11),
                    ha=_ha(x), va="top" if below else "bottom", fontsize=8.5,
                    color=TEXT_SECONDARY)
        peaks.append((x, y))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.10)
    ax.set_xticks(np.round(np.arange(0.0, 1.01, 0.1), 1))
    ax.set_yticks(np.round(np.arange(0.0, 1.01, 0.2), 1))
    ax.set_xlabel("decision threshold", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("score", fontsize=9, color=TEXT_SECONDARY)
    ax.set_title(f"Threshold sweep: local Hinglish test split (n={len(rows)})",
                 loc="left", fontsize=12, color=TEXT_PRIMARY, pad=10)
    ax.legend(loc="lower left", frameon=False, fontsize=9,
              labelcolor=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
def resolve_threshold(run_dir: Path, override: float | None) -> tuple[float, str]:
    if override is not None:
        return float(override), "--threshold"
    path = run_dir / "metrics.json"
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("threshold")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value), "metrics.json"
            print(f"[warn] {path} has no numeric 'threshold' - using 0.5")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] could not read {path} ({e}) - using 0.5")
    else:
        print(f"[warn] no metrics.json in {run_dir} - using default threshold 0.5")
    return 0.5, "default 0.5"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="models",
                    help="run dir holding model_int8.onnx (default: models)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the threshold from metrics.json")
    ap.add_argument("--threads", type=int, default=2, help="ORT intra-op threads")
    args = ap.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = ROOT / args.run
    model = run_dir / "model_int8.onnx"
    if not model.exists():
        raise SystemExit(f"no model_int8.onnx in {run_dir}")

    thr, thr_src = resolve_threshold(run_dir, args.threshold)
    analysis = run_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    rows = load_test_rows()
    det = TurnDetector(model, threshold=thr, num_threads=args.threads)

    for i, r in enumerate(rows, 1):
        r["prob"] = det.predict(load_wav(SYNTH_OUT / r["path"]))["prob_complete"]
        if i % 100 == 0 or i == len(rows):
            print(f"  scored {i}/{len(rows)} clips", file=sys.stderr)

    labels = np.array([r["label"] for r in rows])
    probs = np.array([r["prob"] for r in rows])
    overall = metrics_at(labels, probs, thr)
    auc = roc_auc(labels, probs)

    write_worst_errors(rows, thr, overall, auc, analysis / "worst_errors.md",
                       run_dir, thr_src)
    n_curves = plot_prob_curves(det, thr, analysis / "prob_curves.png")
    plot_threshold_sweep(rows, thr, analysis / "threshold_sweep.png")

    best_i = int(np.argmax([metrics_at(labels, probs, t)["acc"] for t in SWEEP]))
    print(
        f"\nerror analysis - {run_dir.as_posix()}\n"
        f"  local hinglish test clips : {len(rows)} "
        f"({int(labels.sum())} complete / {int((1 - labels).sum())} incomplete)\n"
        f"  threshold                 : {thr:.2f} ({thr_src})\n"
        f"  accuracy @ threshold      : {overall['acc']:.3f}\n"
        f"  F1 @ threshold            : {overall['f1']:.3f} "
        f"(P {overall['precision']:.3f} / R {overall['recall']:.3f})\n"
        f"  AUC                       : {auc:.3f}\n"
        f"  best sweep accuracy       : {max(metrics_at(labels, probs, t)['acc'] for t in SWEEP):.3f} "
        f"@ {SWEEP[best_i]:.2f}\n"
        f"  wrote                     : {(analysis / 'worst_errors.md').as_posix()}, "
        f"prob_curves.png ({n_curves} clips), threshold_sweep.png"
    )


if __name__ == "__main__":
    main()
