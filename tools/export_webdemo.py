"""Stage 1 of the browser demo: export waveform-in ONNX models.

The shipped models take log-mel (1, 80, 800), which forces the client to
reimplement Whisper's feature math. These exports wrap each trained checkpoint
in melgraph.FullTurnModel, so the graph itself starts at raw 16 kHz samples
(1, 128000) and the browser does zero DSP.

Per model: fp32 export -> dynamic int8 quantization -> three gates, all printed.
  a) composed fp32 ONNX vs torch LogMel + torch model, max |dprob| over 8 clips
  b) decision-matched int8 threshold on held-in synth clips (no test labels)
  c) composed int8 acc/AUC on the local Hinglish TEST split vs the shipped
     mel-input int8 artifact measured on the same clips

Writes webdemo/models/{turn_*_fp32.onnx, turn_*_int8.onnx, config.json}.

Run: python -m tools.export_webdemo
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from synth.package_kaggle import split_of                          # noqa: E402
from tools.error_analysis import (                                 # noqa: E402
    SYNTH_OUT, load_test_rows, load_wav, metrics_at, roc_auc,
)
from turn_detector.common import (                                 # noqa: E402
    N_SAMPLES, SAMPLE_RATE, WINDOW_SECONDS, right_align,
)
from turn_detector.features import LogMel                          # noqa: E402
from turn_detector.melgraph import FullTurnModel                   # noqa: E402
from turn_detector.model import (                                  # noqa: E402
    TinyMelNet, WhisperTinyTurn, count_params,
)

OUT_DIR = ROOT / "webdemo" / "models"
EXAMPLES = ROOT / "demo" / "examples"
MATCH_SWEEP = np.round(np.arange(0.05, 0.9501, 0.01), 2)

SPECS = [
    {
        "id": "whisper",
        "label": "Whisper-Tiny encoder (E2)",
        "arch": "whisper",
        "ckpt": ROOT / "experiments" / "run_e2_hinglish_aug" / "ckpt_best.pt",
        "fp32_threshold": 0.63,
        "shipped_int8": ROOT / "models" / "model_int8.onnx",
        "shipped_threshold": 0.50,      # decision-matched, models/metrics.json
    },
    {
        "id": "tinymel",
        "label": "TinyMelNet, distilled (E5)",
        "arch": "tinymel",
        "ckpt": ROOT / "experiments" / "run_e5_distill" / "ckpt_best.pt",
        "fp32_threshold": 0.57,
        "shipped_int8": ROOT / "models" / "model_tinymel_int8.onnx",
        "shipped_threshold": 0.57,
    },
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def frozen(module: torch.nn.Module) -> torch.nn.Module:
    module.train(False)          # BatchNorm/Dropout must be off before export
    return module


def build_model(spec: dict) -> torch.nn.Module:
    model = (WhisperTinyTurn.from_pretrained() if spec["arch"] == "whisper"
             else TinyMelNet())
    state = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    return frozen(model)


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def mel_frontend_nodes(onnx_path: Path) -> list[str]:
    """Nodes consuming MelDFT's fixed buffers.

    Dynamic quantization covers Conv as well as MatMul, which would drop the
    DFT twiddles and the mel filterbank to uint8 and wreck the low-energy bins
    the log floor exists to preserve. Excluding them costs ~0.5 MB, no accuracy.
    """
    import onnx
    graph = onnx.load(str(onnx_path)).graph
    tags = ("cos_kernel", "sin_kernel", "mel_filters")
    buffers = {i.name for i in graph.initializer if any(t in i.name for t in tags)}
    if not buffers:
        raise RuntimeError(f"{onnx_path.name}: MelDFT buffers not found by name")
    return [n.name for n in graph.node if buffers.intersection(n.input)]


def export_pair(spec: dict, model: torch.nn.Module) -> tuple[Path, Path]:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    full = frozen(FullTurnModel(model))
    fp32_path = OUT_DIR / f"turn_{spec['id']}_fp32.onnx"
    int8_path = OUT_DIR / f"turn_{spec['id']}_int8.onnx"
    torch.onnx.export(
        full, (torch.zeros(1, N_SAMPLES),), str(fp32_path),
        input_names=["wav"], output_names=["logit"],
        opset_version=17, dynamo=False,
    )
    excluded = mel_frontend_nodes(fp32_path)
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8,
                     nodes_to_exclude=excluded)
    print(f"  exported {fp32_path.name} ({fp32_path.stat().st_size / 1e6:.2f} MB) "
          f"-> {int8_path.name} ({int8_path.stat().st_size / 1e6:.2f} MB), "
          f"{len(excluded)} mel node(s) kept fp32")
    return fp32_path, int8_path


# --------------------------------------------------------------------------
# clips
# --------------------------------------------------------------------------
def held_in_rows(limit: int) -> list[dict]:
    """Train/val synth clips, for threshold calibration only (no test labels)."""
    rows, seen = [], set()
    with open(SYNTH_OUT / "manifest.jsonl", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["id"] in seen or not (SYNTH_OUT / r["path"]).exists():
                continue
            seen.add(r["id"])
            if split_of(r["sentence_id"]) in ("train", "val"):
                rows.append(r)
    rows.sort(key=lambda r: r["id"])
    idx = np.random.default_rng(0).permutation(len(rows))[:limit]
    return [rows[int(i)] for i in sorted(idx)]


def parity_clips(n: int = 8) -> list[np.ndarray]:
    wavs = [right_align(load_wav(p)) for p in sorted(EXAMPLES.glob("*.flac"))][:n]
    for row in load_test_rows():
        if len(wavs) >= n:
            break
        wavs.append(right_align(load_wav(SYNTH_OUT / row["path"])))
    return wavs


def session(path: Path, threads: int):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=opts,
                                providers=["CPUExecutionProvider"])


def probs_over(sess, wavs, key: str = "wav") -> np.ndarray:
    return sigmoid([sess.run(None, {key: w[None]})[0].ravel()[0] for w in wavs])


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
def check_fp32_parity(spec, model, sess, wavs) -> float:
    mel_fn = LogMel()
    with torch.no_grad():
        ref = sigmoid([model(mel_fn(torch.from_numpy(w))).item() for w in wavs])
    worst = float(np.abs(ref - probs_over(sess, wavs)).max())
    print(f"  [a] fp32 composed vs torch LogMel+model, {len(wavs)} clips: "
          f"max |dprob| = {worst:.2e} ({'ok' if worst < 1e-3 else 'FAIL'})")
    if worst >= 1e-3:
        raise SystemExit(f"{spec['id']}: fp32 parity {worst:.2e} exceeds 1e-3")
    return worst


def decision_matched_threshold(spec, fp32_probs, int8_probs) -> tuple[float, float]:
    ref = fp32_probs >= spec["fp32_threshold"]
    agree = np.array([((int8_probs >= t) == ref).mean() for t in MATCH_SWEEP])
    best = float(agree.max())
    # ties break toward the fp32 threshold, keeping the calibration shift minimal
    cands = MATCH_SWEEP[agree >= best - 1e-12]
    thr = float(cands[int(np.argmin(np.abs(cands - spec["fp32_threshold"])))])
    print(f"  [b] decision-matched int8 threshold: {thr:.2f} "
          f"(agrees with fp32@{spec['fp32_threshold']:.2f} on {best * 100:.1f}% "
          f"of {len(ref)} held-in clips)")
    return thr, best


def report_test(name, labels, probs, thr) -> dict:
    m = metrics_at(labels, probs, thr)
    auc = roc_auc(labels, probs)
    print(f"      {name:<26} thr {thr:.2f}  acc {m['acc']:.4f}  AUC {auc:.4f}")
    return {"acc": round(float(m["acc"]), 4), "auc": round(float(auc), 4)}


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calib-clips", type=int, default=600,
                    help="held-in synth clips for threshold matching")
    ap.add_argument("--threads", type=int, default=4, help="ORT intra-op threads")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    test_rows = load_test_rows()
    test_wavs = [right_align(load_wav(SYNTH_OUT / r["path"])) for r in test_rows]
    test_labels = np.array([r["label"] for r in test_rows])
    calib_wavs = [right_align(load_wav(SYNTH_OUT / r["path"]))
                  for r in held_in_rows(args.calib_clips)]
    pclips = parity_clips()
    print(f"clips: {len(test_wavs)} hinglish test, {len(calib_wavs)} held-in "
          f"calibration, {len(pclips)} parity\n")

    from turn_detector.infer import NumpyLogMel
    np_mel = NumpyLogMel()
    test_mels = [np_mel(w)[None] for w in test_wavs]

    entries, failures = [], []
    for spec in SPECS:
        print(f"[{spec['id']}] {spec['label']}")
        model = build_model(spec)
        fp32_path, int8_path = export_pair(spec, model)

        fp32_sess = session(fp32_path, args.threads)
        int8_sess = session(int8_path, args.threads)
        check_fp32_parity(spec, model, fp32_sess, pclips)

        thr, agree = decision_matched_threshold(
            spec,
            probs_over(fp32_sess, calib_wavs),
            probs_over(int8_sess, calib_wavs),
        )

        print(f"  [c] local Hinglish TEST split (n={len(test_wavs)}):")
        new = report_test("composed int8 (waveform)", test_labels,
                          probs_over(int8_sess, test_wavs), thr)
        ship_sess = session(spec["shipped_int8"], args.threads)
        ship_probs = sigmoid([ship_sess.run(None, {"mel": m})[0].ravel()[0]
                              for m in test_mels])
        old = report_test("shipped int8 (log-mel)", test_labels, ship_probs,
                          spec["shipped_threshold"])

        drop = old["acc"] - new["acc"]
        print(f"      delta acc {new['acc'] - old['acc']:+.4f}, "
              f"delta AUC {new['auc'] - old['auc']:+.4f}"
              f"{'' if drop <= 0.02 else '   [warn] outside the 0.02 band'}")
        if drop > 0.03:
            failures.append(f"{spec['id']}: acc dropped {drop:.4f} (> 0.03)")

        entries.append({
            "id": spec["id"],
            "file": int8_path.name,
            "label": spec["label"],
            "size_mb": round(int8_path.stat().st_size / 1e6, 2),
            "threshold": thr,
            "params": count_params(model),
            "fp32_threshold": spec["fp32_threshold"],
            "decision_agreement": round(agree, 4),
            "test_acc": new["acc"],
            "test_auc": new["auc"],
        })
        print()

    (OUT_DIR / "config.json").write_text(json.dumps({
        "models": entries,
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
    }, indent=2), encoding="utf-8")

    print("webdemo/models/")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name:<26} {f.stat().st_size / 1e6:8.2f} MB")
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min")
    if failures:
        raise SystemExit("QUALITY GATE FAILED: " + "; ".join(failures))


if __name__ == "__main__":
    main()
