#!/usr/bin/env python3
"""Read-only VAE-GAN diagnostics: fixed checkpoint, no model selection/training."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.transforms import functional as TF
from anomaly_detection.config import load_config, resolve_path
from anomaly_detection.data import evaluation_samples, training_paths
from anomaly_detection.torch_train import device_for, load_checkpoint


def scores(error):
    flat = error.reshape(-1)
    return {"mean": float(flat.mean()),
            "max": float(flat.max()),
            "top1pct": float(np.mean(np.sort(flat)[-max(1, int(np.ceil(flat.size * .01))):])),
            "top5pct": float(np.mean(np.sort(flat)[-max(1, int(np.ceil(flat.size * .05))):]))}


def load_tensor(path, size=128):
    with Image.open(path) as img:
        return TF.to_tensor(img.convert("RGB").resize((size, size),
                            Image.Resampling.BILINEAR)).unsqueeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", help="Separate diagnostic directory, never modifies model checkpoint")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--train-samples", type=int, default=32)
    p.add_argument("--sampled-draws", type=int, default=3)
    a = p.parse_args()
    if a.train_samples < 1 or a.sampled_draws < 1:
        p.error("--train-samples and --sampled-draws must be >= 1")
    cfg = load_config(a.config)
    root = resolve_path(cfg, cfg["dataset_root"])
    device = device_for(a.device)
    model, state = load_checkpoint(cfg, Path(a.checkpoint), device)
    model.eval()
    out = Path(a.output) if a.output else resolve_path(cfg, cfg.get("output_dir", "outputs")) / "pytorch" / "diagnostics" / "paper_aligned"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    normal_train_paths = training_paths(root, cfg["category"])
    rng = np.random.default_rng(42)
    subset = rng.choice(len(normal_train_paths), min(len(normal_train_paths), a.train_samples), replace=False)
    samples = [(normal_train_paths[int(i)], 0, "train_normal") for i in subset]
    samples += [(path, label, "test_anomaly" if label else "test_normal")
                for path, label, _mask in evaluation_samples(root, cfg["category"])]
    with torch.inference_mode():
        for path, label, split in samples:
            x = load_tensor(path).to(device)
            mu, logvar, _ = model.encoder(x, sample=False)
            recon = model.decoder(mu)
            error = (x - recon).abs().mean(dim=1)[0].cpu().numpy()
            row = {"path": str(path), "split": split, "label": int(label)}
            row.update({"det_" + key: value for key, value in scores(error).items()})
            row["det_rgb_mae"] = float((x - recon).abs().mean())
            sampled = []
            for _ in range(a.sampled_draws):
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                pred = model.decoder(z)
                sampled.append(float((x - pred).abs().mean()))
            row["sampled_rgb_mae"] = float(np.mean(sampled))
            rows.append(row)
    with (out / "per_image_scores.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    train = [r for r in rows if r["split"] == "train_normal"]
    test = [r for r in rows if r["split"].startswith("test_")]
    y = np.asarray([r["label"] for r in test])
    summary = {
        "checkpoint": str(a.checkpoint), "epoch": int(state.get("epoch", -1)),
        "num_test_images": len(test), "num_train_normal_samples": len(train),
        "note": "Test AUROC is diagnostic, not validation-based checkpoint selection.",
        "image_auroc_by_score": {
            key: float(roc_auc_score(y, [r["det_" + key] for r in test]))
            if len(np.unique(y)) == 2 else None
            for key in ("max", "mean", "top1pct", "top5pct")
        },
        "mean_rgb_mae_by_split": {
            split: {"posterior_mean": float(np.mean([r["det_rgb_mae"] for r in rows if r["split"] == split])),
                    "sampled": float(np.mean([r["sampled_rgb_mae"] for r in rows if r["split"] == split]))}
            for split in ("train_normal", "test_normal", "test_anomaly")
            if any(r["split"] == split for r in rows)
        },
        "error_scores_by_split": {
            split: {key: float(np.mean([r["det_" + key] for r in rows if r["split"] == split]))
                    for key in ("max", "mean", "top1pct", "top5pct")}
            for split in ("train_normal", "test_normal", "test_anomaly")
            if any(r["split"] == split for r in rows)
        }
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[diagnose] wrote {out}/summary.json and per_image_scores.csv")


if __name__ == "__main__":
    main()
