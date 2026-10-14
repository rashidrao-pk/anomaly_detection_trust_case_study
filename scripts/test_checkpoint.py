#!/usr/bin/env python3
"""Inspect reconstruction quality for one test sample per MVTec defect type."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
from anomaly_detection.config import load_config, resolve_path
from anomaly_detection.data import evaluation_samples, load_mask, load_rgb
from anomaly_detection.inference import load_model, reconstruct


def choose_samples(samples, index):
    """Select one sorted example from each category, normal included."""
    if index < 0:
        raise ValueError("--index must be nonnegative")
    grouped = {}
    for item in samples:
        grouped.setdefault(item[0].parent.name, []).append(item)
    if not grouped:
        raise ValueError("No evaluation samples found")
    chosen = []
    for kind in sorted(grouped, key=lambda name: (name != "good", name)):
        if index >= len(grouped[kind]):
            raise IndexError(
                f"Category {kind!r} has {len(grouped[kind])} images, "
                f"but --index={index} was requested"
            )
        chosen.append((kind, *grouped[kind][index]))
    return chosen


def evaluate_reconstructions(cfg, checkpoint, index=0, output=None):
    """Compute measurements and save an input/reconstruction/residual/GT grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.metrics import structural_similarity

    root = resolve_path(cfg, cfg["dataset_root"])
    selected = choose_samples(evaluation_samples(root, cfg["category"]), index)
    checkpoint = str(checkpoint)
    if not Path(checkpoint + ".index").exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}.index")
    model = load_model(cfg, checkpoint)
    image_size = int(cfg.get("image_size", 128))
    destination = Path(output) if output else (
        resolve_path(cfg, cfg.get("output_dir", "outputs"))
        / "checkpoint_tests" / Path(checkpoint).name
    )
    destination.mkdir(parents=True, exist_ok=True)
    images = np.stack([load_rgb(path, image_size) for _, path, _, _ in selected])
    recon = reconstruct(model, images, deterministic=True)
    errors = np.abs(images - recon)
    error_maps = np.mean(errors, axis=-1)
    mse_maps = np.mean(np.square(images - recon), axis=-1)
    vmax = max(float(error_maps.max()), 1e-8)

    fig, axes = plt.subplots(
        len(selected), 5, figsize=(16, 3 * len(selected)), squeeze=False
    )
    rows = []
    for i, (kind, path, label, mask_path) in enumerate(selected):
        inp, rec = images[i], np.clip(recon[i], 0, 1)
        error = error_maps[i]
        gt = (np.zeros((image_size, image_size), dtype=bool)
              if mask_path is None else load_mask(mask_path, image_size))
        mae = float(np.mean(errors[i]))
        mse = float(np.mean(mse_maps[i]))
        psnr = None if mse == 0 else float(10 * np.log10(1.0 / mse))
        ssim = float(structural_similarity(inp, rec, channel_axis=-1,
                                           data_range=1.0))
        row = {
            "category": kind, "label": int(label), "image": str(path),
            "mae": mae, "mse": mse, "psnr_db": psnr, "ssim": ssim,
            "max_pixel_mae": float(error.max()),
            "mean_error_on_gt": float(error[gt].mean()) if gt.any() else None,
            "mean_error_outside_gt": float(error[~gt].mean()) if (~gt).any() else None,
        }
        rows.append(row)
        panels = (inp, rec, error, gt, np.where(gt, error, np.nan))
        titles = ("Input", "Reconstruction", "Mean absolute error",
                  "Ground truth", "Error within GT")
        for j, (panel, title) in enumerate(zip(panels, titles)):
            ax = axes[i, j]
            if j in (2, 4):
                ax.imshow(panel, cmap="magma", vmin=0, vmax=vmax)
            elif j == 3:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(panel)
            if i == 0:
                ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])
        axes[i, 0].set_ylabel(
            f"{kind}\nMAE {mae:.4f} | SSIM {ssim:.3f}",
            rotation=0, labelpad=85, va="center", fontsize=9
        )
        print(f"[checkpoint] {kind}: MAE={mae:.6f} MSE={mse:.6f} "
              f"PSNR={psnr if psnr is not None else 'infinite'} SSIM={ssim:.4f} ({path.name})")

    fig.subplots_adjust(left=0.19, wspace=0.08, hspace=0.18)
    fig.savefig(destination / "reconstruction_grid.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(destination / "reconstruction_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    with (destination / "reconstruction_metrics.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": checkpoint, "category": cfg["category"],
        "selection": "one per category, sorted image order",
        "index": index, "samples": len(rows), "metrics": rows,
        "notes": "Reconstruction fidelity is not anomaly-detection AUROC; "
                 "GT error uses original defective pixel masks."
    }
    (destination / "reconstruction_metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False,
                   default=lambda x: str(x))
    )
    print(f"Saved to: {destination}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", help="Checkpoint prefix; defaults to config's archived checkpoint")
    p.add_argument("--epochs", type=int, help="Use a newly trained checkpoint named by epoch count")
    p.add_argument("--index", type=int, default=0, help="Zero-based index within each anomaly category")
    p.add_argument("--output-dir", help="Override output directory")
    args = p.parse_args()
    if args.epochs is not None and args.checkpoint is not None:
        p.error("Choose either --epochs or --checkpoint, not both")
    if args.epochs is not None and args.epochs <= 0:
        p.error("--epochs must be positive")
    cfg = load_config(args.config)
    if args.epochs is not None:
        checkpoint = (
            resolve_path(cfg, cfg.get("output_dir", "outputs"))
            / "trained_models"
            / f"{cfg['category']}_VAE_GAN_{args.epochs}_model"
        )
    else:
        checkpoint = (args.checkpoint if args.checkpoint is not None
                      else resolve_path(cfg, cfg["checkpoint"]))
    evaluate_reconstructions(cfg, checkpoint, args.index, args.output_dir)


if __name__ == "__main__":
    main()
