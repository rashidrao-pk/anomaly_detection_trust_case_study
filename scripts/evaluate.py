#!/usr/bin/env python3
"""Evaluate original TensorFlow component weights on MVTec Screw/Hazelnut."""
import argparse
import csv
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tf_scripts.core import restore_models, load_rgb, reconstruct, residual_map
from tf_scripts.config import apply_config, require_args, resolve_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="YAML configuration from configs/")
    p.add_argument("--dataset-root")
    p.add_argument("--category", choices=["screw", "hazelnut"])
    p.add_argument("--checkpoint-dir",
                   help="Directory with <category>_VAE_GAN_<epochs>_{encoder,decoder}.index")
    p.add_argument("--checkpoint-epochs", type=int, default=None)
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    a = apply_config(a, p, "evaluate")
    require_args(a, p, ("dataset_root", "category", "checkpoint_dir"))
    a.latent_dim = 32 if a.latent_dim is None else int(a.latent_dim)
    a.checkpoint_epochs = 30000 if a.checkpoint_epochs is None else int(a.checkpoint_epochs)
    a.output = a.output or "outputs/tensorflow/evaluation"
    a.dataset_root = str(resolve_path(a.dataset_root))
    a.checkpoint_dir = str(resolve_path(a.checkpoint_dir))
    a.output = str(resolve_path(a.output))
    import numpy as np
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    model = restore_models(a.checkpoint_dir, a.category, a.checkpoint_epochs, a.latent_dim)
    folder = Path(a.dataset_root) / a.category
    paths = sorted(p for p in (folder / "test").glob("*/*.png"))
    if not paths:
        p.error(f"No MVTec test images under {folder / 'test'}")
    image_labels, image_scores, masks, maps, rows = [], [], [], [], []
    for path in paths:
        kind = path.parent.name
        label = int(kind != "good")  # no class-index ordering assumption
        image = load_rgb(path)[None]
        recon = reconstruct(model, image)
        error = residual_map(image, recon)[0]
        score = float(error.max())
        if label:
            mask_path = folder / "ground_truth" / kind / f"{path.stem}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)
            with Image.open(mask_path) as m:
                mask = np.asarray(m.convert("L").resize((128,128),Image.Resampling.NEAREST)) > 0
        else:
            mask = np.zeros((128,128), bool)
        image_labels.append(label)
        image_scores.append(score)
        masks.append(mask.ravel())
        maps.append(error.ravel())
        rows.append({"path": str(path), "label": label, "score": score})
    result = {
        "backend": "tensorflow", "category": a.category, "samples": len(rows),
        "image_auroc": float(roc_auc_score(image_labels, image_scores)),
        "pixel_auroc": float(roc_auc_score(np.concatenate(masks), np.concatenate(maps))),
        "checkpoint_dir": str(a.checkpoint_dir),
        "image_score_definition": "max(mean_absolute_error_RGB)",
    }
    out = Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    with (out / "image_scores.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=["path","label","score"])
        writer.writeheader();writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
