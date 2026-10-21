"""Shared, testable explanation visualization and dataset helpers."""
from pathlib import Path
import numpy as np
from PIL import Image


def test_images(dataset_root, category, anomaly_type=None):
    base = Path(dataset_root) / category / "test"
    if not base.is_dir():
        raise FileNotFoundError(base)
    folders = ([base / anomaly_type] if anomaly_type else
               sorted(f for f in base.iterdir() if f.is_dir()))
    result = []
    for folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        result.extend(sorted(f for f in folder.iterdir()
                             if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
                             and f.is_file()))
    if not result:
        raise ValueError(f"No images in {base}")
    return result


def ground_truth_mask(dataset_root, category, path, shape):
    path = Path(path)
    if path.parent.name == "good":
        return None
    mask_path = (Path(dataset_root) / category / "ground_truth" /
                 path.parent.name / f"{path.stem}_mask.png")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing GT for {path}: {mask_path}")
    with Image.open(mask_path) as mask:
        array = np.asarray(mask.convert("L").resize(
            (shape[1], shape[0]), Image.Resampling.NEAREST))
    return array > 0


def attribution_map(segments, weights):
    seg = np.asarray(segments)
    if seg.ndim != 2:
        raise ValueError("Expected 2D superpixels")
    result = np.zeros(seg.shape, dtype=np.float32)
    for index, weight in weights:
        result[seg == int(index)] = float(weight)
    return result


def display_overlay(ax, image, weights, cutoff=0.2):
    """Visual-only opacity proportional to magnitude, not binary coverage."""
    from matplotlib.colors import ListedColormap
    arr = np.asarray(weights, dtype=np.float32)
    strength = float(np.max(np.abs(arr)))
    ax.imshow(image)
    if strength <= 1e-12:
        return
    norm = arr / strength
    colors = np.zeros((*norm.shape, 4), dtype=np.float32)
    colors[norm > 0, 0] = 1.0
    colors[norm < 0, 2] = 1.0
    colors[..., 3] = np.where(np.abs(norm) >= cutoff,
                              0.8 * np.abs(norm), 0.0)
    ax.imshow(colors, interpolation="nearest")


def anomaly_score_predictor(model, image, baseline_recon, batch_size):
    """LIME/SHAP shared bounded reconstruction-error score.

    Pseudo-probability: monotonic error ratio, *not* a trained classifier.
    """
    from tf_scripts.core import reconstruct
    reference = max(float(np.square(image - baseline_recon).sum()), 1e-8)
    def predict(batch):
        samples = np.asarray(batch, dtype=np.float32)
        scores = []
        for i in range(0, len(samples), batch_size):
            subset = samples[i:i + batch_size]
            reconstruction = reconstruct(model, subset)
            scores.append(np.square(subset - reconstruction).sum(axis=(1, 2, 3)))
        value = np.concatenate(scores)
        anomaly = value / (value + reference)
        return np.column_stack([1 - anomaly, anomaly]).astype(np.float32)
    return predict


def shap_spatial_map(values, image_shape):
    """Extract class-1 SHAP values, average across RGB channels."""
    arr = np.asarray(values)
    h, w, c = image_shape
    if arr.ndim == 5 and arr.shape[0] == 1 and arr.shape[1:4] == (h, w, c):
        if arr.shape[-1] < 2:
            raise ValueError(f"Expected 2 SHAP output columns, got {arr.shape}")
        return arr[0, ..., 1].mean(axis=-1)
    if arr.ndim == 4 and arr.shape == (1, h, w, c):
        return arr[0].mean(axis=-1)
    raise ValueError(f"Unexpected SHAP result shape: {arr.shape}")


def reference_heatmap(ax, values):
    """Plot signed weights in the XAI-2024 notebook style.

    Both paper notebooks use opaque Matplotlib bwr and a symmetric color
    limit of +/- max(abs(weight)), plus a vertical colorbar. Do not display
    a transparent overlay: it distorts the apparent color of signed weights.
    """
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    maxval = float(np.max(np.abs(finite))) if finite.size else 0.0
    maxval = max(maxval, 1e-8)
    return ax.imshow(array, cmap="bwr", vmin=-maxval, vmax=maxval)


def paper_mse_predictor(model, batch_size=200):
    """Original notebook's single-output, unnormalized reconstruction MSE."""
    from tf_scripts.core import reconstruct

    def predict(data):
        data = np.asarray(data, dtype=np.float32)
        scores = []
        for start in range(0, len(data), batch_size):
            chunk = data[start:start + batch_size]
            rec = reconstruct(model, chunk, deterministic=True)
            scores.append(np.square(rec - chunk).sum(axis=(1, 2, 3)))
        return np.concatenate(scores).reshape(-1, 1)
    return predict


def paper_quickshift(image, max_dist):
    """Original XAI-2024 notebook Quickshift arguments."""
    from skimage.segmentation import quickshift
    args = dict(kernel_size=2, max_dist=max_dist, ratio=0.95, sigma=0.05)
    try:
        return quickshift(image, rng=1234, **args)
    except TypeError as err:
        if "rng" not in str(err):
            raise
        return quickshift(image, random_seed=1234, **args)


def paper_segments(image, target_segments=100, initial_max_dist=100):
    """Binary-search Quickshift distance as in the paper's search_segment_number."""
    if target_segments < 1:
        raise ValueError("target_segments must be positive")
    left, right = 0.0, float(initial_max_dist)
    left_count = len(np.unique(paper_quickshift(image, left)))
    right_segments = paper_quickshift(image, right)
    right_count = len(np.unique(right_segments))
    iteration = 0
    while iteration < 40 and right_count != target_segments:
        iteration += 1
        middle = (left + right) / 2
        count = len(np.unique(paper_quickshift(image, middle)))
        if count <= target_segments <= left_count:
            right, right_count = middle, count
        else:
            left, left_count = middle, count
    return paper_quickshift(image, right)


def paper_shap_spatial_map(values, image_shape):
    """Match original notebook's first channel and single-output indexing."""
    arr = np.asarray(values)
    h, w, channels = image_shape
    expected = (1, h, w, channels, 1)
    if arr.shape != expected:
        raise ValueError(f"Expected paper SHAP values shape {expected}, got {arr.shape}")
    return arr[0, :, :, 0, 0]
