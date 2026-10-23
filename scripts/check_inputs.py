#!/usr/bin/env python3
"""Recreate GT_<category>.png: inputs, masks, yellow defect contours."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tf_scripts.paper_figures import context, image_paths, gt_for_image, save_figure
from tf_scripts.core import load_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", type=int, default=0, help="Zero-based sample position within each test class")
    ap.add_argument("--output")
    args = ap.parse_args()
    if args.index < 0:
        ap.error("--index must be nonnegative")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg, root, out = context(args.config, args.output)
    examples = image_paths(root, one_per_class=True, index=args.index)
    fig, axes = plt.subplots(3, len(examples), figsize=(3.1*len(examples), 9.3),
                             squeeze=False)
    for j, path in enumerate(examples):
        original = load_rgb(path)
        mask = gt_for_image(root, path)
        axes[0, j].imshow(original)
        axes[0, j].set_title(path.parent.name, fontsize=14)
        axes[1, j].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[2, j].imshow(original)
        if mask.any():
            axes[2, j].contour(mask.astype(float), levels=[0.5],
                               colors=["yellow"], linewidths=1.6)
        for i in range(3):
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
    fig.tight_layout(pad=0.8)
    print(save_figure(fig, out / f"GT_{cfg['category']}"))


if __name__ == "__main__":
    main()
