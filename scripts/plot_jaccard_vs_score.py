#!/usr/bin/env python3
"""Plot maximum Jaccard against anomaly scores from REAL evaluation CSVs."""
import argparse
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tf_scripts.config import resolve_path
from tf_scripts.paper_figures import context, canonical_path, save_figure


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def oracle_threshold(labels, scores):
    import numpy as np
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, float)
    if labels.all() or not labels.any():
        raise ValueError("Both test labels needed for oracle threshold")
    best_quality, cutoff = -1, None
    for candidate in np.unique(scores):
        prediction = scores > candidate
        tpr = prediction[labels].mean()
        fpr = prediction[~labels].mean()
        quality = float(np.sqrt(tpr * (1 - fpr)))
        if quality > best_quality:
            best_quality, cutoff = quality, float(candidate)
    return cutoff


def plot_one(scores, explanations, method, cutoff, output, sentinel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = {name:[[], []] for name in (
        "normal/correct","normal/incorrect","anomaly/correct","anomaly/incorrect")}
    known = {canonical_path(row["image_path"]):row for row in explanations}
    if len(known) != len(explanations):
        raise ValueError("Duplicate explanation image paths")
    for row in scores:
        key = canonical_path(row["path"])
        if key not in known:
            continue
        label = int(row["label"])
        x = float(row["score"])
        correct = (x > cutoff) == bool(label)
        group = ("anomaly" if label else "normal") + ("/correct" if correct else "/incorrect")
        if label:
            val = known[key].get(method+"_best_iou","")
            if not val or val.lower() in ("nan","none"):
                continue
            y = float(val)
        else:
            y = sentinel  # Visual placeholder, NOT an IoU.
        groups[group][0].append(x)
        groups[group][1].append(y)
    colors = ["blue","orange","green","red"]
    fig, ax = plt.subplots(figsize=(8,6))
    for (name, (xs, ys)), color in zip(groups.items(),colors):
        ax.scatter(xs, ys, facecolors="none", edgecolors=color,
                   s=32, linewidths=1.3, label=f"{name} {len(xs)}")
    ax.axvline(cutoff, color="black", linestyle=":")
    ax.set_xlabel("Anomaly score", fontsize=15)
    ax.set_ylabel("Maximum Jaccard", fontsize=15)
    ax.set_title(method.upper())
    ax.set_ylim(min(sentinel-.08,-.05),1.02)
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--scores-csv", required=True)
    p.add_argument("--manifest", action="append", required=True)
    p.add_argument("--method", choices=["lime","shap","both"],default="both")
    p.add_argument("--threshold",type=float)
    p.add_argument("--threshold-mode",choices=["fixed","paper-oracle"],default="fixed")
    p.add_argument("--normal-level",type=float,default=-0.15)
    p.add_argument("--output")
    a = p.parse_args()
    if a.threshold is None and a.threshold_mode == "fixed":
        p.error("Pass --threshold X or --threshold-mode paper-oracle")
    cfg, root, out = context(a.config, a.output)
    scores = read_rows(resolve_path(a.scores_csv))
    if not scores:
        p.error("Score CSV contains no rows")
    cutoff = oracle_threshold([int(r["label"]) for r in scores],
                              [float(r["score"]) for r in scores]) if a.threshold_mode=="paper-oracle" else a.threshold
    for file in a.manifest:
        path = resolve_path(file)
        records = read_rows(path)
        if not records:
            raise ValueError(f"No explanation rows in {path}")
        for method in (["lime","shap"] if a.method=="both" else [a.method]):
            target = out / f"plt_{cfg['category']}_{method}_{path.stem}"
            print(plot_one(scores, records, method, cutoff, target, a.normal_level))
    if a.threshold_mode == "paper-oracle":
        print("NOTE: threshold was selected using test labels; this is an oracle diagnostic.")


if __name__ == "__main__":
    main()
