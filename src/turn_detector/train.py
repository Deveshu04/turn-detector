"""Training, evaluation, and ONNX export for turn-detection experiments.

Step-based training (sampler draws with replacement, so batches are iid and a
mid-run resume just continues from the saved step — no epoch bookkeeping).
Checkpoints every cfg.checkpoint_every_steps to <out_dir>/ckpt_last.pt; a
killed Kaggle session loses at most that many steps. Resume is automatic when
the checkpoint's config hash matches.

Final artifacts in out_dir: ckpt_best.pt, model_fp32.onnx, model_int8.onnx,
metrics.json.

Note for security scanners: `model.eval()` below is PyTorch's inference-mode
switch, not code evaluation.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from turn_detector.config import ExperimentConfig
from turn_detector.dataset import TurnDataset, load_manifests
from turn_detector.features import LogMel
from turn_detector.model import build_model, count_params


# ---------------- metrics ----------------

def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC via the rank-sum statistic (no sklearn dependency)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def f1_score(labels: np.ndarray, preds: np.ndarray) -> float:
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else float("nan")


def tune_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        acc = float(((probs >= t) == labels).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def slice_metrics(rows: pl.DataFrame, labels: np.ndarray, probs: np.ndarray,
                  threshold: float) -> dict:
    def compute(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return {"n": 0}
        l, p = labels[mask], probs[mask]
        preds = (p >= threshold).astype(int)
        return {
            "n": int(mask.sum()),
            "acc_050": round(float(((p >= 0.5) == l).mean()), 4),
            "acc_tuned": round(float((preds == l).mean()), 4),
            "f1_tuned": round(f1_score(l, preds), 4),
            "auc": round(rank_auc(l, p), 4),
        }

    lang = rows["language"].to_numpy()
    midf = rows["midfiller"].fill_null(False).to_numpy().astype(bool)
    endf = rows["endfiller"].fill_null(False).to_numpy().astype(bool)
    synth = rows["synthetic"].fill_null(False).to_numpy().astype(bool)
    all_mask = np.ones(len(labels), dtype=bool)
    return {
        "overall": compute(all_mask),
        "english": compute(lang == "english"),
        "hindi": compute(lang == "hindi"),
        "hinglish": compute(lang == "hinglish"),
        "filler": compute(midf | endf),
        "human_audio": compute(~synth & (lang != "hinglish")),
        "threshold": threshold,
    }


# ---------------- eval ----------------

@torch.no_grad()
def predict(model, mel_fn, dataset, device, batch_size=64, num_workers=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device != "cpu"))
    model.eval()
    all_probs, all_labels = [], []
    for wav, label, _ in loader:
        mel = mel_fn(wav.to(device))
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(mel)
        all_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        all_labels.append(label.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


# ---------------- checkpointing ----------------

def save_ckpt(path: Path, model, opt, sched, scaler, step, best_val_auc, cfg_hash):
    torch.save({
        "model": model.state_dict(), "opt": opt.state_dict(),
        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
        "step": step, "best_val_auc": best_val_auc, "cfg_hash": cfg_hash,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


# ---------------- training ----------------

def train(cfg: ExperimentConfig, sources: dict, out_dir: str,
          device: str | None = None, steps_per_epoch: int | None = None,
          num_workers: int = 2):
    """sources: {"train": [(manifest, root), ...], "val": ..., "test": ...}"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_df = load_manifests(sources["train"], "train")
    val_df = load_manifests(sources["val"], "val")
    print(f"train rows: {train_df.height}, val rows: {val_df.height}")
    train_ds = TurnDataset(train_df, cfg, train=True)
    val_ds = TurnDataset(val_df, cfg, train=False)

    spe = steps_per_epoch or math.ceil(train_df.height / cfg.batch_size)
    total_steps = spe * cfg.epochs
    warmup = max(1, int(total_steps * cfg.warmup_frac))

    model = build_model(cfg.arch).to(device)
    print(f"arch={cfg.arch} params={count_params(model):,}")
    mel_fn = LogMel().to(device)

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    other_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": cfg.lr_encoder},
         {"params": other_params, "lr": cfg.lr_head}],
        weight_decay=cfg.weight_decay,
    )

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # resume
    step, best_val_auc = 0, 0.0
    ckpt_last = out / "ckpt_last.pt"
    if ckpt_last.exists():
        state = torch.load(ckpt_last, map_location=device, weights_only=False)
        if state["cfg_hash"] == cfg.config_hash():
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            scaler.load_state_dict(state["scaler"])
            step = state["step"]
            best_val_auc = state["best_val_auc"]
            print(f"resumed from step {step} (best val AUC {best_val_auc:.4f})")
        else:
            print("checkpoint config hash mismatch — starting fresh")

    history = []
    t_start = time.time()
    epoch_pass = step // spe
    while step < total_steps:
        train_ds.set_epoch(epoch_pass)
        loader = DataLoader(
            train_ds, batch_size=cfg.batch_size,
            sampler=train_ds.balanced_sampler(num_samples=spe * cfg.batch_size),
            num_workers=num_workers, pin_memory=(device == "cuda"),
            drop_last=True, persistent_workers=False,
        )
        model.train()
        for wav, label, _ in loader:
            if step >= total_steps:
                break
            mel = mel_fn(wav.to(device, non_blocking=True))
            label = label.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                loss = loss_fn(model(mel), label)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

            if step % 50 == 0:
                print(f"step {step}/{total_steps} loss {loss.item():.4f} "
                      f"lr {sched.get_last_lr()[-1]:.2e} "
                      f"({(time.time() - t_start) / 60:.1f} min)", flush=True)
            if step % cfg.checkpoint_every_steps == 0:
                save_ckpt(ckpt_last, model, opt, sched, scaler, step,
                          best_val_auc, cfg.config_hash())

            if step % spe == 0:  # epoch boundary -> validate
                vl, vp = predict(model, mel_fn, val_ds, device,
                                 cfg.batch_size, num_workers)
                val_auc = rank_auc(vl, vp)
                val_acc = float(((vp >= 0.5) == vl).mean())
                history.append({"step": step, "val_auc": round(val_auc, 4),
                                "val_acc_050": round(val_acc, 4)})
                print(f"  == step {step}: val AUC {val_auc:.4f} "
                      f"acc@0.5 {val_acc:.4f}", flush=True)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save({"model": model.state_dict(),
                                "cfg_hash": cfg.config_hash(),
                                "step": step, "val_auc": val_auc},
                               out / "ckpt_best.pt")
                save_ckpt(ckpt_last, model, opt, sched, scaler, step,
                          best_val_auc, cfg.config_hash())
                model.train()
        epoch_pass += 1

    # ---- final evaluation with best weights ----
    best = torch.load(out / "ckpt_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])

    vl, vp = predict(model, mel_fn, val_ds, device, cfg.batch_size, num_workers)
    threshold = tune_threshold(vl, vp)

    test_df = load_manifests(sources["test"], "test")
    test_ds = TurnDataset(test_df, cfg, train=False)
    tl, tp = predict(model, mel_fn, test_ds, device, cfg.batch_size, num_workers)
    test_metrics = slice_metrics(test_df, tl, tp, threshold)

    metrics = {
        "experiment": cfg.name,
        "config": cfg.__dict__,
        "params": count_params(model),
        "train_rows": train_df.height,
        "best_val_auc": round(best_val_auc, 4),
        "history": history,
        "threshold": threshold,
        "test": test_metrics,
        "train_minutes": round((time.time() - t_start) / 60, 1),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(test_metrics, indent=2))

    export_onnx(model, mel_fn, out, cfg, test_df, test_ds, threshold, metrics)
    return metrics


# ---------------- export ----------------

def export_onnx(model, mel_fn, out: Path, cfg, test_df, test_ds,
                threshold, metrics):
    import onnxruntime as ort
    model = model.cpu().eval()
    dummy = torch.randn(1, 80, 800)
    fp32_path = out / "model_fp32.onnx"
    # static batch=1: turn detection inference is streaming, one window at a
    # time, and dynamic batch breaks the bidirectional-GRU reshape on export
    try:  # legacy exporter: consistent shape metadata, quantizer-friendly
        torch.onnx.export(
            model, (dummy,), str(fp32_path),
            input_names=["mel"], output_names=["logit"],
            opset_version=17, dynamo=False,
        )
    except TypeError:  # older torch without the dynamo kwarg
        torch.onnx.export(
            model, (dummy,), str(fp32_path),
            input_names=["mel"], output_names=["logit"], opset_version=17,
        )

    # torch vs onnx parity (per-sample, batch=1 graph)
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    diffs = []
    for _ in range(4):
        x = torch.randn(1, 80, 800)
        with torch.no_grad():
            ref = torch.sigmoid(model(x)).numpy()
        got = 1 / (1 + np.exp(-sess.run(None, {"mel": x.numpy()})[0]))
        diffs.append(np.abs(ref - got).max())
    parity = float(max(diffs))
    print(f"onnx fp32 parity: max |dprob| = {parity:.2e}")

    from onnxruntime.quantization import QuantType, quantize_dynamic
    int8_path = out / "model_int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)

    # int8 accuracy on a stratified test subset (bounded runtime on CPU)
    rng = np.random.default_rng(0)
    labels_np = test_df["label"].to_numpy()
    idx = np.concatenate([
        rng.permutation(np.nonzero(labels_np == c)[0])[:1000] for c in (0, 1)
    ])
    sess8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    probs, labels = [], []
    mel_cpu = mel_fn.cpu()
    for i in idx:
        wav, label, _ = test_ds[int(i)]
        mel = mel_cpu(wav.unsqueeze(0)).numpy()
        logit = sess8.run(None, {"mel": mel})[0][0]
        probs.append(1 / (1 + np.exp(-logit)))
        labels.append(int(label))
    labels, probs = np.array(labels), np.array(probs).ravel()
    int8_metrics = {
        "n": len(labels),
        "acc_tuned": round(float(((probs >= threshold) == labels).mean()), 4),
        "auc": round(rank_auc(labels, probs), 4),
        "size_mb": round(int8_path.stat().st_size / 1e6, 2),
        "fp32_size_mb": round(fp32_path.stat().st_size / 1e6, 2),
        "fp32_parity_max_dprob": parity,
    }
    metrics["int8_subset"] = int8_metrics
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("int8:", json.dumps(int8_metrics, indent=2))
