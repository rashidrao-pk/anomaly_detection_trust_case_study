#!/usr/bin/env python3
"""Single-image and batch XAI for the original TensorFlow VAE-GAN."""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tf_scripts.core import restore_models, load_rgb, reconstruct, residual_map
from tf_scripts.config import apply_config, require_args, resolve_path
from tf_scripts.xai_utils import (test_images, ground_truth_mask, attribution_map,
                                  display_overlay, anomaly_score_predictor,
                                  shap_spatial_map)


def explain_image(model, path, dataset_root, category, method, num_samples,
                  shap_evals, target, lime_batch_size):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = load_rgb(path)
    recon = reconstruct(model, image[None])[0]
    error = residual_map(image, recon)
    gt = ground_truth_mask(dataset_root, category, path, image.shape[:2])

    panels = [("Input", image, "image"),
              ("Input + GT" if gt is not None else "Input + GT (normal)",
               (image, gt), "gt"),
              ("Reconstruction", recon, "image"),
              ("Error map", error, "error")]

    # Shared bounded score surrogate; NOT a calibrated anomaly probability.
    # SHAP and LIME must explain exactly the same target.
    prediction = anomaly_score_predictor(model, image, recon,
                                         lime_batch_size)
    if method in ("lime", "all"):
        from lime import lime_image
        from skimage.segmentation import slic
        explainer = lime_image.LimeImageExplainer(random_state=42)
        explanation = explainer.explain_instance(
            image, prediction, labels=(1,), num_samples=num_samples,
            hide_color=0, batch_size=lime_batch_size,
            segmentation_fn=lambda x: slic(x, n_segments=100,
                                           compactness=10, sigma=1,
                                           start_label=0, channel_axis=-1))
        weights = explanation.local_exp.get(1, [])
        lime_map = attribution_map(explanation.segments, weights)
        panels.append(("LIME weights (+red / -blue)", (image, lime_map),
                       "attribution"))
    if method in ("shap", "all"):
        import shap
        masker = shap.maskers.Image("blur(16,16)", image.shape)
        explainer = shap.Explainer(prediction, masker, algorithm="partition",
                                  output_names=["reference", "anomaly surrogate"])
        result = explainer(image[None], max_evals=shap_evals,
                           batch_size=lime_batch_size)
        shap_map = shap_spatial_map(result.values, image.shape)
        panels.append(("SHAP weights (+red / -blue)", (image, shap_map),
                       "attribution"))

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(4.0 * len(panels), 4.6),
                             squeeze=False)
    for ax, (title, payload, kind) in zip(axes[0], panels):
        if kind == "image":
            ax.imshow(np.clip(payload, 0, 1))
        elif kind == "gt":
            source, mask = payload
            ax.imshow(source)
            if mask is not None and mask.any():
                ax.contour(mask.astype(float), levels=[0.5],
                           colors=["cyan"], linewidths=1.5)
        elif kind == "error":
            ax.imshow(payload, cmap="Reds", vmin=0)
        else:
            source, weights = payload
            display_overlay(ax, source, weights)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"{category} | class: {path.parent.name} | image ID: {path.stem}",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return float(error.max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="YAML configuration from configs/")
    p.add_argument("--dataset-root", help="Override dataset root from YAML")
    p.add_argument("--checkpoint-dir")
    p.add_argument("--category", choices=["screw", "hazelnut"])
    p.add_argument("--checkpoint-epochs", type=int)
    p.add_argument("--latent-dim", type=int)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--image")
    mode.add_argument("--all", action="store_true")
    p.add_argument("--anomaly-type", help="Restrict --all to one test category")
    p.add_argument("--method", choices=["residual", "lime", "shap", "all"])
    p.add_argument("--num-samples", type=int, help="LIME perturbations")
    p.add_argument("--shap-samples", type=int, default=200,
                   help="SHAP model evaluation budget (>= 6)")
    p.add_argument("--lime-batch-size", type=int, default=16)
    p.add_argument("--limit", type=int)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--output")
    a = p.parse_args()
    a = apply_config(a, p, "explain")
    require_args(a, p, ("category", "checkpoint_dir", "dataset_root") if a.all else
                 ("category", "checkpoint_dir", "dataset_root", "image"))
    if a.anomaly_type and not a.all:
        p.error("--anomaly-type requires --all")
    if a.limit is not None and (not a.all or a.limit < 1):
        p.error("--limit requires --all and positive N")
    if a.shap_samples < 6 or a.lime_batch_size < 1:
        p.error("--shap-samples must be >=6 and --lime-batch-size >=1")
    if a.num_samples is not None and a.num_samples < 2:
        p.error("--num-samples must be >=2")

    a.checkpoint_epochs = 30000 if a.checkpoint_epochs is None else int(a.checkpoint_epochs)
    a.latent_dim = 32 if a.latent_dim is None else int(a.latent_dim)
    a.method = a.method or "residual"
    a.num_samples = 1000 if a.num_samples is None else int(a.num_samples)
    dataset_root = resolve_path(a.dataset_root)
    output_dir = resolve_path(a.output or "outputs/tensorflow/xai")
    model = restore_models(resolve_path(a.checkpoint_dir), a.category,
                           a.checkpoint_epochs, a.latent_dim)
    if a.all:
        images = test_images(dataset_root, a.category, a.anomaly_type)
        images = images[:a.limit] if a.limit else images
        output_root = output_dir / "batch" / a.category
    else:
        images = [resolve_path(a.image)]
        output_root = output_dir

    rows = []
    for index, image_path in enumerate(images, start=1):
        target = ((output_root / image_path.parent.name /
                   f"{image_path.stem}_{a.method}.png") if a.all else
                  output_root / f"{image_path.stem}_{a.method}.png")
        existing = a.skip_existing and target.is_file()
        score = ("" if existing else explain_image(
            model, image_path, dataset_root, a.category, a.method,
            a.num_samples, a.shap_samples, target, a.lime_batch_size))
        rows.append(dict(image_path=str(image_path),
                         defect_type=image_path.parent.name,
                         image_id=image_path.stem,
                         label=int(image_path.parent.name != "good"),
                         method=a.method, figure=str(target),
                         status="skipped" if existing else "generated",
                         max_residual=score))
        print(f"[{index}/{len(images)}] {rows[-1]['status']}: {target}", flush=True)

    if a.all:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = output_root / f"manifest_{a.method}.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Manifest: {manifest}")
    else:
        print(rows[0]["figure"])


if __name__ == "__main__":
    main()
