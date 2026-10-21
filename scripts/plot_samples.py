#!/usr/bin/env python3
"""Paper-style sample grid: input, deterministic reconstruction and anomaly maps."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tf_scripts.config import resolve_path
from tf_scripts.paper_figures import context, image_paths, save_figure
from tf_scripts.core import restore_models, load_rgb, reconstruct, residual_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--output")
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--checkpoint-epochs", type=int)
    args = ap.parse_args()
    if args.index < 0:
        ap.error("--index must be nonnegative")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg, root, out = context(args.config, args.output)
    images = image_paths(root, one_per_class=True, index=args.index)
    prefix = resolve_path(cfg["checkpoint"])
    epochs = args.checkpoint_epochs
    if epochs is None:
        import re
        match = re.search(r"_VAE_GAN_(\d+)", str(prefix))
        if not match:
            ap.error("Cannot infer checkpoint epochs; specify --checkpoint-epochs")
        epochs = int(match.group(1))
    model = restore_models(resolve_path(args.checkpoint_dir) if args.checkpoint_dir else prefix.parent,
                           cfg["category"], epochs, int(cfg.get("latent_dim", 32)))
    fig, axes = plt.subplots(3, len(images), figsize=(3.1*len(images), 9.0),
                             squeeze=False)
    for j, path in enumerate(images):
        original = load_rgb(path)
        reconstruction = reconstruct(model, original[None])[0]
        err = residual_map(original, reconstruction)
        axes[0,j].imshow(original)
        axes[0,j].set_title(path.parent.name, fontsize=13)
        axes[1,j].imshow(np.clip(reconstruction, 0, 1))
        axes[2,j].imshow(err, cmap="Reds", vmin=0)
        for i in range(3):
            axes[i,j].axis("off")
    for i, name in enumerate(("Input", "Reconstruction", "Error map")):
        axes[i,0].set_ylabel(name, fontsize=13)
    fig.tight_layout(pad=0.7)
    print(save_figure(fig, out / f"samples_m_{cfg['category']}"))


if __name__ == "__main__":
    main()
