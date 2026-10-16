"""Monitoring utilities for PyTorch VAE-GAN training."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from .data import evaluation_samples


def _short_index(stem: str) -> str:
    try:
        return str(int(stem))
    except ValueError:
        return stem


def _sample_name(path: Path, split: str) -> str:
    if split in ("train_normal", "test_normal"):
        return f"normal_{_short_index(path.stem)}"
    defect = path.parent.name
    return f"{defect}_{_short_index(path.stem)}"


def validate_monitor_sample(image_path, mask_path, split, root, category):
    """Reject incorrect split/category or missing/misaligned anomaly masks."""
    root = Path(root).resolve()
    path = Path(image_path).resolve()
    base = (root / category).resolve()
    if split == "train_normal":
        expected = base / "train" / "good"
    elif split == "test_normal":
        expected = base / "test" / "good"
    elif split == "anomalous_test":
        expected = base / "test" / path.parent.name
        if path.parent.name == "good":
            raise ValueError(f"Anomalous sample cannot come from test/good: {path}")
    else:
        raise ValueError(f"Unrecognized monitor split: {split}")
    if path.parent != expected or not path.is_file():
        raise ValueError(f"Monitor image not in {expected}: {path}")

    if split != "anomalous_test":
        if mask_path is not None:
            raise ValueError(f"Normal image must not carry a GT mask: {path}")
        return

    expected_mask = base / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
    if mask_path is None or Path(mask_path).resolve() != expected_mask or not expected_mask.is_file():
        raise FileNotFoundError(f"Expected GT mask for {path}: {expected_mask}")
    with Image.open(path) as img, Image.open(expected_mask) as mask:
        if img.size != mask.size:
            raise ValueError(f"Image/mask shape mismatch: {path} {img.size} vs {expected_mask} {mask.size}")
        if not np.any(np.asarray(mask.convert("L")) > 0):
            raise ValueError(f"Empty GT mask for anomalous sample: {expected_mask}")



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
      - 1 normal train sample and 1 normal test sample
      - 2 anomalous test samples (preferably from different anomaly folders)
    """
    rng = random.Random(seed)
    root = Path(root)
    if n_normal != 2:
        raise ValueError("Monitoring requires 2 normal samples: one train and one test")

    # ---------- normal training samples ----------
    train_paths = sorted(Path(p) for p in train_paths)
    if not train_paths:
        raise ValueError("Need at least one train/good sample")
    test_normals = sorted((root / category / "test" / "good").glob("*.png"))
    if not test_normals:
        raise ValueError(f"No normal test images found under {root / category / 'test' / 'good'}")

    samples = []
    for split, paths in (("train_normal", train_paths), ("test_normal", test_normals)):
        p = rng.choice(paths)
        samples.append((_sample_name(p, split), p, None, split))

    # ---------- anomalous test samples ----------
    # Use the same canonical dataset manifest as the evaluation pipeline.
    # This prevents independent monitor/evaluator sample enumeration from
    # silently disagreeing about defect labels or mask filenames.
    anomalies_by_type = defaultdict(list)
    for image_path, label, mask_path in evaluation_samples(root, category):
        if label:
            anomalies_by_type[Path(image_path).parent.name].append(
                (Path(image_path), Path(mask_path)))

    if len(anomalies_by_type) < 2 and n_anomaly > 1:
        raise ValueError("Need at least two anomaly categories for distinct monitoring examples")
    rng_types = sorted(anomalies_by_type)
    rng.shuffle(rng_types)
    chosen = [rng.choice(anomalies_by_type[kind]) for kind in rng_types[:n_anomaly]]
    if len(chosen) < n_anomaly:
        raise ValueError(f"Need {n_anomaly} distinct anomalous categories, found {len(chosen)}")
    for image_path, mask_path in chosen:
        samples.append((_sample_name(image_path, "anomalous_test"),
                        image_path, mask_path, "anomalous_test"))

    for _name, image_path, mask_path, split in samples:
        validate_monitor_sample(image_path, mask_path, split, root, category)

    return samples


def write_pair_metadata(samples, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": name,
        "defect_category": (Path(image_path).parent.name if split == "anomalous_test" else "good"),
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
    # [train_normal, test_normal, anomaly1, anomaly2]
    normal_samples = samples[:2]
    anomaly_samples = samples[2:]
    grouped = [("Normal", normal_samples), ("Anomalous test", anomaly_samples)]
    split_labels = {"train_normal": "Normal train", "test_normal": "Normal test",
                    "anomalous_test": "Anomalous test"}

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
            # input_gray = image_np.mean(axis=2)
            # recon_gray = recon_np.mean(axis=2)
            

            # -------- Input + GT overlay --------
            ax = axes[row_idx, col0]
            # ax.imshow(input_gray, cmap="gray", vmin=0, vmax=1)
            ax.imshow(image_np)  # Input panel

            # Overlay GT mask only if present
            if gt_mask.sum() > 0:
                masked = np.ma.masked_where(gt_mask <= 0, gt_mask)
                ax.imshow(masked, cmap="Blues", alpha=0.35, vmin=0, vmax=1)
                ax.contour(gt_mask, levels=[0.5], colors="cyan", linewidths=1.2)

            source = Path(image_path)\n            ax.set_title(f"{split_labels[split]} — {name}\\n{source.parent.parent.name}/{source.parent.name}/{source.name}\\nInput + GT", fontsize=10)
            ax.axis("off")

            # -------- Reconstruction --------
            ax = axes[row_idx, col0 + 1]
            # ax.imshow(recon_gray, cmap="gray", vmin=0, vmax=1)
            ax.imshow(recon_np)  # Reconstruction panel
            ax.set_title("Reconstruction", fontsize=12)
            ax.axis("off")

            # -------- Anomaly map --------
            ax = axes[row_idx, col0 + 2]
            ax.imshow(anomaly_map, cmap="Reds")
            ax.set_title("Anomaly map", fontsize=12)
            ax.axis("off")

        # Row label on the l    eft
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
