"""Monitoring utilities for PyTorch VAE-GAN training."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _short_index(stem: str) -> str:
    try:
        return str(int(stem))
    except ValueError:
        return stem


def _sample_name(path: Path, split: str) -> str:
    if split == "train_normal":
        return f"normal_{_short_index(path.stem)}"
    defect = path.parent.name
    return f"{defect}_{_short_index(path.stem)}"


def _mask_path_for(root: Path, category: str, image_path: Path) -> Path | None:
    defect = image_path.parent.name
    if defect == "good":
        return None
    mask = root / category / "ground_truth" / defect / f"{image_path.stem}_mask.png"
    return mask if mask.exists() else None


def _load_rgb(path: Path, image_size: int = 128) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def _load_mask(path: Path | None, image_size: int = 128) -> np.ndarray:
    if path is None or not Path(path).exists():
        return np.zeros((image_size, image_size), dtype=np.float32)
    with Image.open(path) as image:
        image = image.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
        arr = np.asarray(image, dtype=np.float32)
    return (arr > 127).astype(np.float32)


def choose_pair(train_paths, root, category, seed=42, n_normal=2, n_anomaly=2):
    """
    Returns a list of monitoring samples:
      [(name, image_path, mask_path, split), ...]
    with:
      - 2 normal train samples
      - 2 anomalous test samples (preferably from different anomaly folders)
    """
    rng = random.Random(seed)
    root = Path(root)

    # ---------- normal training samples ----------
    train_paths = sorted(Path(p) for p in train_paths)
    if len(train_paths) < n_normal:
        raise ValueError(f"Need at least {n_normal} train/good samples, found {len(train_paths)}")
    chosen_normals = rng.sample(train_paths, n_normal)

    samples = []
    for p in chosen_normals:
        samples.append((_sample_name(p, "train_normal"), p, None, "train_normal"))

    # ---------- anomalous test samples ----------
    test_root = root / category / "test"
    anomalies_by_type = defaultdict(list)

    for defect_dir in sorted(test_root.iterdir()):
        if not defect_dir.is_dir():
            continue
        if defect_dir.name == "good":
            continue
        for p in sorted(defect_dir.glob("*.png")):
            anomalies_by_type[defect_dir.name].append(p)

    if not anomalies_by_type:
        raise ValueError(f"No anomalous test images found under {test_root}")

    # Prefer different anomaly folders/types
    defect_types = list(anomalies_by_type.keys())
    rng.shuffle(defect_types)

    chosen_anomalies = []
    for defect in defect_types:
        if len(chosen_anomalies) >= n_anomaly:
            break
        chosen_anomalies.append(rng.choice(anomalies_by_type[defect]))

    # If still short, fill from remaining anomaly images
    if len(chosen_anomalies) < n_anomaly:
        all_anomalies = []
        for defect, items in anomalies_by_type.items():
            all_anomalies.extend(items)
        remaining = [p for p in all_anomalies if p not in chosen_anomalies]
        need = n_anomaly - len(chosen_anomalies)
        if len(remaining) < need:
            raise ValueError(f"Not enough anomalous samples to choose {n_anomaly}")
        chosen_anomalies.extend(rng.sample(remaining, need))

    for p in chosen_anomalies:
        samples.append((_sample_name(p, "anomalous_test"), p, _mask_path_for(root, category, p), "anomalous_test"))

    return samples


def write_pair_metadata(samples, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": name,
            "image": str(image_path),
            "mask": None if mask_path is None else str(mask_path),
            "split": split,
        }
        for name, image_path, mask_path, split in samples
    ]
    (output_dir / "monitor_samples.json").write_text(json.dumps(payload, indent=2))


@torch.inference_mode()
def save_monitor_image(model, samples, epoch, device, output_dir, image_size=128):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(samples) != 4:
        raise ValueError(
            f"Expected exactly 4 monitor samples (2 normal + 2 anomaly), got {len(samples)}"
        )

    # Expected order from choose_pair:
    # [normal1, normal2, anomaly1, anomaly2]
    normal_samples = samples[:2]
    anomaly_samples = samples[2:]
    grouped = [("Normal train", normal_samples), ("Anomalous test", anomaly_samples)]

    # Layout:
    # 2 rows total
    # each row has 2 samples
    # each sample has 3 panels: Input(+GT), Reconstruction, Anomaly map
    nrows = 2
    ncols = 6

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(18, 8),
        squeeze=False
    )

    model.eval()

    for row_idx, (row_name, row_samples) in enumerate(grouped):
        for sample_idx, (name, image_path, mask_path, split) in enumerate(row_samples):
            col0 = sample_idx * 3

            image_np = _load_rgb(Path(image_path), image_size=image_size)
            x = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

            recon = model.reconstruct(x).clamp(0, 1)
            recon_np = recon[0].permute(1, 2, 0).cpu().numpy()

            anomaly_map = (x - recon).abs().mean(dim=1)[0].cpu().numpy()
            gt_mask = _load_mask(mask_path, image_size=image_size)

            # Convert to grayscale for display
            input_gray = image_np.mean(axis=2)
            recon_gray = recon_np.mean(axis=2)

            # -------- Input + GT overlay --------
            ax = axes[row_idx, col0]
            ax.imshow(input_gray, cmap="gray", vmin=0, vmax=1)

            # Overlay GT mask only if present
            if gt_mask.sum() > 0:
                masked = np.ma.masked_where(gt_mask <= 0, gt_mask)
                ax.imshow(masked, cmap="Blues", alpha=0.35, vmin=0, vmax=1)
                ax.contour(gt_mask, levels=[0.5], colors="cyan", linewidths=1.2)

            ax.set_title(f"{name}\nInput + GT", fontsize=12)
            ax.axis("off")

            # -------- Reconstruction --------
            ax = axes[row_idx, col0 + 1]
            ax.imshow(recon_gray, cmap="gray", vmin=0, vmax=1)
            ax.set_title("Reconstruction", fontsize=12)
            ax.axis("off")

            # -------- Anomaly map --------
            ax = axes[row_idx, col0 + 2]
            ax.imshow(anomaly_map, cmap="Reds")
            ax.set_title("Anomaly map", fontsize=12)
            ax.axis("off")

        # Row label on the left
        axes[row_idx, 0].text(
            -0.32,
            0.5,
            row_name,
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=14,
            fontweight="bold",
            rotation=90,
        )

    fig.suptitle(f"Monitoring samples — epoch {epoch}", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = output_dir / f"epoch_{epoch:06d}.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path