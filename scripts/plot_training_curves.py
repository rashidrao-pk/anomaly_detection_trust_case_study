#!/usr/bin/env python3
"""Plot every numeric training history metric from original checkpoint CSV."""
import argparse
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tf_scripts.config import resolve_path
from tf_scripts.paper_figures import context, save_figure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--history", help="Explicit train history CSV; otherwise checkpoint's CSV")
    ap.add_argument("--output")
    ap.add_argument("--smoothing", type=int, default=1, help="Rolling window, 1 = raw")
    args = ap.parse_args()
    if args.smoothing < 1:
        ap.error("--smoothing must be >= 1")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg, root, out = context(args.config, args.output)
    cp = resolve_path(cfg["checkpoint"])
    history = resolve_path(args.history) if args.history else cp.parent / (cp.name.rsplit("_",1)[0]+".csv")
    if not history.is_file():
        raise FileNotFoundError(f"No training history found: {history}. Provide --history.")
    with history.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Training CSV is empty: {history}")
    columns = {}
    for name in rows[0]:
        try:
            vals = np.asarray([float(r[name]) for r in rows], dtype=float)
            if np.isfinite(vals).any():
                columns[name] = vals
        except (ValueError, TypeError):
            continue
    epoch = columns.pop("epoch", np.arange(1,len(rows)+1))
    columns.pop("", None)
    columns.pop("Unnamed: 0", None)
    if not columns:
        raise ValueError(f"No numeric losses found in {history}")
    count = len(columns)
    fig, axes = plt.subplots(count, 1, figsize=(11, max(3.1*count,4)),
                             squeeze=False)
    for ax, (name, vals) in zip(axes.flat, columns.items()):
        ax.plot(epoch, vals, linewidth=0.75, alpha=0.45, label="Raw")
        if args.smoothing > 1:
            from numpy.lib.stride_tricks import sliding_window_view
            smoothed = np.convolve(vals, np.ones(args.smoothing)/args.smoothing, mode="valid")
            ax.plot(epoch[args.smoothing-1:], smoothed, linewidth=1.8, label="Smoothed")
            ax.legend()
        ax.set_ylabel(name)
        ax.grid(alpha=0.2)
    axes[-1,0].set_xlabel("Epoch")
    fig.tight_layout()
    print(save_figure(fig, out / f"training_curves_{cfg['category']}"))
    print(f"Source: {history}")


if __name__ == "__main__":
    main()
