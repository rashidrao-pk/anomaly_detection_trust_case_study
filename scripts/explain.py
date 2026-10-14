#!/usr/bin/env python3
"""Explain a configured MVTec AD test sample with LIME or SHAP."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from anomaly_detection.config import load_config, resolve_path
from anomaly_detection.data import image_files
from anomaly_detection.inference import load_model
from anomaly_detection.explain import explain_image

def choose_image(cfg, image=None, anomaly_type=None, index=None):
    """Choose a test image from dataset_root/category/test using YAML defaults."""
    if image is not None:
        candidate = Path(image).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"Image not found: {candidate}")
        return candidate
    settings = cfg.get("explanation", {})
    anomaly_type = anomaly_type if anomaly_type is not None else settings.get("anomaly_type", "good")
    index = index if index is not None else int(settings.get("image_index", 0))
    if index < 0:
        raise ValueError("image_index must be non-negative")
    dataset_root = resolve_path(cfg, cfg["dataset_root"])
    folder = dataset_root / cfg["category"] / "test" / anomaly_type
    images = image_files(folder)
    if not images:
        raise ValueError(f"No images found in {folder}")
    if index >= len(images):
        raise IndexError(f"Image index {index} out of range for {folder}: {len(images)} files")
    return images[index]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--image", help="Optional explicit image path; overrides config selection")
    p.add_argument("--anomaly-type", help="Override explanation.anomaly_type in config")
    p.add_argument("--index", type=int, help="Override explanation.image_index in config (zero-based)")
    p.add_argument("--method", choices=["lime", "shap"], default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--output", help="Optional output filename; defaults to configured output_dir")
    p.add_argument("--checkpoint")
    args = p.parse_args()
    cfg = load_config(args.config)
    settings = cfg.get("explanation", {})
    image = choose_image(cfg, args.image, args.anomaly_type, args.index)
    method = args.method or settings.get("method", "lime")
    samples = args.samples if args.samples is not None else int(settings.get("num_samples", 100))
    scale = args.scale if args.scale is not None else float(settings.get("score_scale", 0.1))
    output = args.output
    if output is None:
        directory = resolve_path(cfg, cfg.get("output_dir", "outputs")) / "explanations"
        directory.mkdir(parents=True, exist_ok=True)
        output = str(directory / f"{image.parent.name}_{image.stem}_{method}.png")
    checkpoint = args.checkpoint or resolve_path(cfg, cfg["checkpoint"])
    print(f"Selected image: {image}")
    model = load_model(cfg, checkpoint)
    explain_image(cfg, model, image, method=method, num_samples=samples, scale=scale, output=output)
    print(f"Saved explanation: {output}")

if __name__ == "__main__":
    main()
