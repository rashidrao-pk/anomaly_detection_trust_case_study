#!/usr/bin/env python3
"""Explain one or all MVTec test images with TensorFlow VAE-GAN.

Batch outputs retain defect category and original filename to prevent
overwrites (for example, test/good/000.png and test/crack/000.png).
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tf_scripts.core import restore_models, load_rgb, reconstruct, residual_map
from tf_scripts.config import apply_config, require_args, resolve_path


def test_images(dataset_root, category, anomaly_type=None):
    base = Path(dataset_root) / category / "test"
    if not base.is_dir():
        raise FileNotFoundError(f"MVTec test directory does not exist: {base}")
    if anomaly_type:
        if anomaly_type not in [p.name for p in base.iterdir() if p.is_dir()]:
            raise ValueError(f"No defect category {anomaly_type!r} under {base}")
        kinds = [base / anomaly_type]
    else:
        kinds = sorted(p for p in base.iterdir() if p.is_dir())
    files = [p for folder in kinds for p in sorted(folder.iterdir())
             if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    if not files:
        raise ValueError(f"No test images found under {base}")
    return files


def explain_image(model, path, method, num_samples, target, lime_batch_size):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    image = load_rgb(path)
    recon = reconstruct(model, image[None])[0]
    err = residual_map(image, recon)
    panels = [("Input", image), ("Reconstruction", recon), ("Error map", err)]
    if method == "lime":
        from lime import lime_image
        # LIME's image API expects two-class probabilities, not raw
        # reconstruction errors. Use a monotonic bounded surrogate.
        # Reference is the unperturbed sample's MSE, not a learned
        # calibration threshold or a claim of real anomaly probability.
        reference_score = float(np.square(image - recon).sum()) + 1e-8

        def predict(batch):
            batch = np.asarray(batch, dtype=np.float32)
            scores = []
            for i in range(0, len(batch), lime_batch_size):
                part = batch[i:i + lime_batch_size]
                recovered = reconstruct(model, part)
                scores.append(np.square(part - recovered).sum(axis=(1, 2, 3)))
            scores = np.concatenate(scores)
            p_anomaly = scores / (scores + reference_score)
            return np.column_stack([1.0 - p_anomaly, p_anomaly])

        # labels=(1,) guarantees we explain anomaly score even when the
        # unperturbed sample receives a normal-leaning surrogate score.
        explanation = lime_image.LimeImageExplainer(random_state=42).explain_instance(
            image, predict, labels=(1,), hide_color=0,
            num_samples=num_samples, batch_size=lime_batch_size)
        # Preserve real signed local coefficients instead of binarizing all
        # selected superpixels into uniformly saturated red/blue regions.
        from tf_scripts.lime_viz import signed_attribution
        lime_map = signed_attribution(explanation.segments, explanation.local_exp[1])
        panels.append(("LIME attribution (+red / -blue)", (image, lime_map)))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (title, arr) in zip(axes, panels):
        if isinstance(arr, tuple):
            original, attribution = arr
            ax.imshow(original)
            # Per-image normalization makes relative coefficients visible.
            # A fixed 0.15 display threshold suppresses weak contributions.
            scale = float(np.max(np.abs(attribution)))
            normalized = attribution / scale if scale > 0 else attribution
            overlay = np.ma.masked_where(np.abs(normalized) < 0.15, normalized)
            ax.imshow(overlay, cmap="bwr", vmin=-1, vmax=1,
                      alpha=0.65, interpolation="nearest")
        elif arr.ndim == 2:
            ax.imshow(arr, cmap="Reds" if title == "Error map" else "gray")
        else:
            ax.imshow(np.clip(arr, 0, 1))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200)
    plt.close(fig)
    return float(err.max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="YAML configuration from configs/")
    p.add_argument("--dataset-root", help="Override dataset root from YAML")
    p.add_argument("--checkpoint-dir")
    p.add_argument("--category", choices=["screw", "hazelnut"])
    p.add_argument("--checkpoint-epochs", type=int)
    p.add_argument("--latent-dim", type=int)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--image", help="Explain a single image")
    mode.add_argument("--all", action="store_true", help="Explain every MVTec test image")
    p.add_argument("--anomaly-type", help="When using --all, restrict to one test folder (including good)")
    p.add_argument("--method", choices=["residual", "lime"])
    p.add_argument("--num-samples", type=int)
    p.add_argument("--lime-batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, help="Limit batch to first N files for a smoke test")
    p.add_argument("--skip-existing", action="store_true", help="Skip existing PNGs")
    p.add_argument("--output")
    a = p.parse_args()
    a = apply_config(a, p, "explain")
    required = ("category", "checkpoint_dir", "dataset_root") if a.all else (
        "category", "checkpoint_dir", "image")
    require_args(a, p, required)
    if a.anomaly_type and not a.all:
        p.error("--anomaly-type requires --all")
    if a.limit is not None and (not a.all or a.limit < 1):
        p.error("--limit requires --all and a positive integer")
    if a.num_samples is not None and a.num_samples < 2:
        p.error("--num-samples must be at least 2")
    if a.lime_batch_size < 1:
        p.error("--lime-batch-size must be positive")

    a.checkpoint_epochs = 30000 if a.checkpoint_epochs is None else int(a.checkpoint_epochs)
    a.latent_dim = 32 if a.latent_dim is None else int(a.latent_dim)
    a.method = a.method or "residual"
    a.num_samples = 1000 if a.num_samples is None else int(a.num_samples)
    out = resolve_path(a.output or "outputs/tensorflow/xai")
    model = restore_models(resolve_path(a.checkpoint_dir), a.category,
                           a.checkpoint_epochs, a.latent_dim)

    if a.all:
        root = resolve_path(a.dataset_root)
        images = test_images(root, a.category, a.anomaly_type)
        if a.limit is not None:
            images = images[:a.limit]
        output_root = out / "batch" / a.category
    else:
        images = [resolve_path(a.image)]
        output_root = out

    rows = []
    processed = skipped = 0
    for number, image_path in enumerate(images, 1):
        category_name = image_path.parent.name
        output_file = (output_root / category_name / f"{image_path.stem}_{a.method}.png"
                       if a.all else output_root / f"{image_path.stem}_{a.method}.png")
        status = "skipped" if a.skip_existing and output_file.is_file() else "generated"
        max_error = ""
        if status == "generated":
            max_error = explain_image(model, image_path, a.method, a.num_samples,
                                      output_file, a.lime_batch_size)
            processed += 1
        else:
            skipped += 1
        rows.append({
            "image_path": str(image_path), "defect_type": category_name,
            "label": 0 if category_name == "good" else 1,
            "method": a.method, "figure": str(output_file),
            "status": status, "max_residual": max_error,
        })
        print(f"[{number}/{len(images)}] {status}: {output_file}", flush=True)

    if a.all:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = output_root / f"manifest_{a.method}.csv"
        with manifest.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Batch finished: generated={processed}, skipped={skipped}, manifest={manifest}")
    else:
        print(rows[0]["figure"])


if __name__ == "__main__":
    main()
