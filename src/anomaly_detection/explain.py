"""Faithful LIME port of the XAI 2024 screw notebook (cells 34, 37, 40, 42).

Target: sum of squared RGB reconstruction errors from the encoder posterior mean.
Mask: deterministic reconstruction. Segmentation: Quickshift of 50/50 blend.
The original source is preserved on the xai-2024-paper branch.
"""
from pathlib import Path
import numpy as np
from .data import load_rgb
from .inference import reconstruct, predict_scores
from .visualization import save_explanation

def reconstruction_mse(model, batch):
    """Notebook lime_predict_loss: reduce_sum((decoder(z_mean)-input)^2)."""
    batch = np.asarray(batch, dtype=np.float32)
    recon = reconstruct(model, batch, deterministic=True)
    return np.sum(np.square(recon - batch), axis=(1, 2, 3))

def quickshift_segments(image, target=100, seed=1234):
    """Port notebook search_segment_number, with bounded binary search."""
    from skimage.segmentation import quickshift
    def segment(distance):
        # scikit-image 0.22 uses rng; earlier versions used random_seed.
        try:
            seg = quickshift(image, kernel_size=2, max_dist=distance,
                             ratio=0.95, rng=seed, sigma=0.05)
        except TypeError:
            seg = quickshift(image, kernel_size=2, max_dist=distance,
                             ratio=0.95, random_seed=seed, sigma=0.05)
        return len(np.unique(seg)), seg
    left, right = 0.0, 100.0
    left_count, _ = segment(left)
    right_count, _ = segment(right)
    iterations = 0
    while iterations < 40 and right_count != target:
        iterations += 1
        midpoint = (left + right) / 2
        midpoint_count, _ = segment(midpoint)
        if midpoint_count <= target <= left_count:
            right, right_count = midpoint, midpoint_count
        else:
            left, left_count = midpoint, midpoint_count
    return segment(right)[1]

def explain_image(cfg, model, image_path, method="lime", num_samples=None, scale=None, output=None):
    image = load_rgb(image_path, int(cfg.get("image_size", 128)))
    reconstruction = reconstruct(model, image[None])[0]
    settings = cfg.get("explanation", {})
    samples = int(num_samples if num_samples is not None else settings.get("num_samples", 5000))
    seed = int(settings.get("random_state", 1234))
    if method == "lime":
        from lime.lime_image import LimeImageExplainer
        blend = (image + reconstruction) * 0.5
        segments = quickshift_segments(blend, int(settings.get("target_segments", 100)), seed)
        print(f"[xai-2024] Quickshift segments: {len(np.unique(segments))}")
        explainer = LimeImageExplainer(random_state=seed)
        # LIME expects 2-D [n_samples, n_outputs]; a single continuous
        # regression target is handled here as the sole output (index 0).
        def predictor(batch):
            return reconstruction_mse(model, batch)[:, None]
        explanation = explainer.explain_instance(
            image,
            predictor,
            labels=(0,),
            segmentation_fn=lambda _img: segments,
            hide_color=reconstruction,
            batch_size=int(settings.get("batch_size", 200)),
            num_samples=samples,
        )
        weights = dict(explanation.local_exp[0])
        heat = np.zeros(segments.shape, dtype=np.float32)
        for index, value in weights.items():
            heat[segments == index] = value
        print(f"[xai-2024] reconstruction MSE: {reconstruction_mse(model, image[None])[0]:.6g}")
        print(f"[xai-2024] LIME beta: min={heat.min():.6g}, max={heat.max():.6g}, nonzero={np.count_nonzero(heat)}")
    elif method == "shap":
        import shap
        # Original notebook's reconstruction-image masker and scalar loss.
        masker = shap.maskers.Image(reconstruction, reconstruction.shape)
        def predictor(batch):
            return reconstruction_mse(model, batch)[:, None]
        explainer = shap.Explainer(predictor, masker, output_names=["Anomaly"])
        required = 2 * image.size + 1
        result = explainer(image[None], max_evals=max(samples, required))
        values = np.asarray(result.values)[0]
        if values.ndim == 4:
            values = values[..., 0]
        heat = np.mean(values, axis=-1) if values.ndim == 3 else values
    else:
        raise ValueError(f"Unsupported method: {method}")
    if output is not None:
        _, maps, _ = predict_scores(model, image[None])
        save_explanation(image, reconstruction, maps[0], heat, output, signed=True)
        np.save(str(Path(output).with_suffix(".npy")), heat)
    return heat
