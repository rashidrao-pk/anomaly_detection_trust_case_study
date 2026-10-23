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
                                  reference_heatmap, anomaly_score_predictor,
                                  shap_spatial_map, paper_mse_predictor,
                                  paper_segments, paper_shap_spatial_map,
                                  optimal_jaccard, paper_jaccard_visualization)


def explain_image(model, path, dataset_root, category, method, num_samples,
                  shap_evals, target, lime_batch_size, target_segments=100, fontsize=15,
                  evaluate_explanations=False):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = load_rgb(path)
    recon = reconstruct(model, image[None])[0]
    error = residual_map(image, recon)
    gt = ground_truth_mask(dataset_root, category, path, image.shape[:2])

    panels = [(f"Input", image, "image"),
              ("Input + GT" if gt is not None else "Input + GT (normal)",
               (image, gt), "gt"),
              ("Reconstruction", recon, "image"),
              ("Error map", error, "error")]

    # XAI-World 2024: raw, single-output reconstruction MSE.
    prediction = paper_mse_predictor(model, lime_batch_size)
    attribution_maps = {}
    if method in ("lime", "all"):
        from lime import lime_image
        blend = (image + recon) * 0.5
        segments = paper_segments(blend, target_segments)
        explainer = lime_image.LimeImageExplainer(random_state=1234)
        explanation = explainer.explain_instance(
            image, prediction, top_labels=1, num_samples=num_samples,
            hide_color=recon, batch_size=lime_batch_size,
            segmentation_fn=lambda ignored: segments)
        label = explanation.top_labels[0]
        weights = explanation.local_exp[label]
        lime_map = attribution_map(segments, weights)
        panels.append(("LIME heatmap", lime_map, "attribution"))
        attribution_maps["lime"] = lime_map
    if method in ("shap", "all"):
        import shap
        masker = shap.maskers.Image(recon, recon.shape)
        explainer = shap.Explainer(prediction, masker,
                                  output_names=["Anomaly"])
        result = explainer(image[None], max_evals=shap_evals,
                           batch_size=lime_batch_size)
        shap_map = paper_shap_spatial_map(result.values, image.shape)
        panels.append(("SHAP heatmap", shap_map, "attribution"))
        attribution_maps["shap"] = shap_map

    evaluation = {}
    if evaluate_explanations:
        for name, values in attribution_maps.items():
            # Match the paper figure: Jaccard notation as the panel title,
            # numeric IoU inside the upper-left of the image.
            panel_title = (r"J($\gamma,\gamma^\prime_L$)" if name == "lime"
                           else r"J($\gamma,\gamma^\prime_S$)")
            if gt is None or not gt.any():
                evaluation[name + "_best_iou"] = float("nan")
                evaluation[name + "_threshold"] = float("nan")
                image_panel = np.full((*image.shape[:2], 3), 255, dtype=np.uint8)
                overlay_text = "N/A"
            else:
                best_iou, cutoff, predicted = optimal_jaccard(gt, values)
                evaluation[name + "_best_iou"] = best_iou
                evaluation[name + "_threshold"] = cutoff
                image_panel = paper_jaccard_visualization(gt, predicted)
                overlay_text = f"{best_iou:.4f}"
            panels.append((panel_title, image_panel, "evaluation", overlay_text))
    # Allocate colorbars their own columns: Matplotlib's fig.colorbar(ax=ax)
    # otherwise shrinks the LIME/SHAP image panels.
    fontsize = 19 if evaluate_explanations else 15
    panel_widths = []
    for panel in panels:
        panel_widths.append(1.0)
        if panel[2] == "attribution":
            panel_widths.append(0.07)

    fig = plt.figure(figsize=(3.35 * len(panels) + 0.35 * len(attribution_maps),
                              3.95))
    grid = fig.add_gridspec(
        1, len(panel_widths), width_ratios=panel_widths,
        left=0.025, right=0.985, bottom=0.08, top=0.87, wspace=0.15)

    # Same absolute color limits for BOTH methods within this figure.
    # These are raw attribution units, not separately normalized overlays.
    shared_max = max(
        (float(np.max(np.abs(values))) for values in attribution_maps.values()),
        default=1e-8)
    shared_max = max(shared_max, 1e-8)
    column = 0
    first_ax = None
    for panel in panels:
        title, payload, kind = panel[:3]
        overlay_text = panel[3] if len(panel) > 3 else None
        ax = fig.add_subplot(grid[0, column])
        if first_ax is None:
            first_ax = ax
        column += 1
        if kind == "image":
            ax.imshow(np.clip(payload, 0, 1))
        elif kind == "gt":
            source, mask = payload
            ax.imshow(source)
            if mask is not None and mask.any():
                ax.contour(mask.astype(float), levels=[0.5],
                           colors=["cyan"], linewidths=1.5)
        elif kind == "evaluation":
            ax.imshow(payload)
            if overlay_text is not None:
                ax.text(0.03, 0.95, overlay_text, transform=ax.transAxes,
                        ha="left", va="top", fontsize=fontsize, color="black")
        elif kind == "error":
            ax.imshow(payload, cmap="Reds", vmin=0)
        else:
            heatmap_artist = ax.imshow(
                payload, cmap="bwr", vmin=-shared_max, vmax=shared_max)
            cax = fig.add_subplot(grid[0, column])
            column += 1
            fig.colorbar(heatmap_artist, cax=cax)
            cax.tick_params(labelsize=max(8, fontsize - 8))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=fontsize)
        ax.set_xticks([])
        ax.set_yticks([])

    first_ax.set_ylabel(f"{category} | {path.parent.name} - {path.stem}",
                        fontsize=fontsize)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return float(error.max()), evaluation


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
    p.add_argument("--include-good", action="store_true", help="Include normal samples in --all")
    p.add_argument("--method", choices=["residual", "lime", "shap", "all"])
    p.add_argument("--num-samples", type=int, help="LIME perturbations")
    p.add_argument("--shap-samples", type=int, default=5000,
                   help="SHAP model evaluation budget (>= 6)")
    p.add_argument("--lime-batch-size", type=int, default=200)
    p.add_argument("--target-segments", type=int, default=100)
    p.add_argument("--limit", type=int)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--evaluate-explanations", action="store_true",
                   help="Add paper-style optimal Jaccard/GT panels and CSV metrics")
    p.add_argument("--output")
    a = p.parse_args()
    # The publication defaults take precedence over older YAML demonstration
    # num_samples: 100, unless overridden explicitly by --num-samples.
    if a.num_samples is None:
        a.num_samples = 5000
    a = apply_config(a, p, "explain")
    require_args(a, p, ("category", "checkpoint_dir", "dataset_root") if a.all else
                 ("category", "checkpoint_dir", "dataset_root", "image"))
    if a.evaluate_explanations and a.method == "residual":
        p.error("--evaluate-explanations requires --method lime, shap, or all")
    if a.include_good and not a.all:
        p.error("--include-good requires --all")
    if a.anomaly_type and not a.all:
        p.error("--anomaly-type requires --all")
    if a.limit is not None and (not a.all or a.limit < 1):
        p.error("--limit requires --all and positive N")
    if a.target_segments < 1:
        p.error("--target-segments must be positive")
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
        images = test_images(dataset_root, a.category, a.anomaly_type, include_good=a.include_good)
        images = images[:a.limit] if a.limit else images
        output_root = output_dir / "batch" / a.category
    else:
        images = [resolve_path(a.image)]
        output_root = output_dir

    rows = []
    suffix = "_eval" if a.evaluate_explanations else ""
    for index, image_path in enumerate(images, start=1):
        target = ((output_root / image_path.parent.name /
                   f"{image_path.stem}_{a.method}{suffix}.png") if a.all else
                  output_root / f"{image_path.parent.name}_{image_path.stem}_{a.method}{suffix}.png")
        existing = a.skip_existing and target.is_file()
        if existing:
            score, evaluation = "", {}
        else:
            score, evaluation = explain_image(
                model, image_path, dataset_root, a.category, a.method,
                a.num_samples, a.shap_samples, target, a.lime_batch_size,
                a.target_segments, evaluate_explanations=a.evaluate_explanations)
        rows.append(dict(
            image_path=str(image_path),
            defect_type=image_path.parent.name,
            image_id=image_path.stem,
            label=int(image_path.parent.name != "good"),
            method=a.method, figure=str(target),
            status="skipped" if existing else "generated",
            max_residual=score,
            lime_best_iou=evaluation.get("lime_best_iou", ""),
            lime_threshold=evaluation.get("lime_threshold", ""),
            shap_best_iou=evaluation.get("shap_best_iou", ""),
            shap_threshold=evaluation.get("shap_threshold", "")))
        print(f"[{index}/{len(images)}] {rows[-1]['status']}: {target}", flush=True)

    if a.all:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = output_root / f"manifest_{a.method}{suffix}.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Manifest: {manifest}")
    else:
        print(rows[0]["figure"])
        if a.evaluate_explanations:
            print({k:v for k,v in rows[0].items() if "iou" in k or "threshold" in k})


if __name__ == "__main__":
    main()
