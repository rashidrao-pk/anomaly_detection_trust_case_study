#!/usr/bin/env python3
"""Compute everything the demo page replays, with the paper's own pipeline.

For every sample in config.toml this script runs, unchanged, the XAI-2024
notebook pipeline (AD_VAE_GAN_*.ipynb on the xai-2024-paper branch):

  1. reconstruction  xi' = d(mu(e(xi)))          (trained 30000-epoch VAE-GAN)
  2. anomaly map     m = max_c |xi' - xi|,  alpha = max(m)
  3. LIME            Quickshift (k=100) on (xi + xi')/2  [S1]
                     or on 0.33 xi + 0.33 xi' + 0.34 gamma  [S2],
                     masked superpixels replaced by xi', f = SSE(d(e(x)) - x),
                     LimeImageExplainer(random_state=1234), n = 5000
  4. SHAP            partition explainer, masker = Image(xi'), max_evals = 5000
  5. ground truth    gamma, for the optimal-IoU check

LIME's `data_labels` is wrapped to log every perturbation it draws and every
model response, and the script checks that refitting LIME's own ridge
regression on the full log rebuilds the explanation `explain_instance`
returned. The page only looks values up: what it shows after n perturbations
is the surrogate LIME fits on the first n rows of that log.

Run:  python explain.py               # all samples in config.toml
      python explain.py --only crack  # samples whose type contains "crack"
"""
import argparse
import base64
import io
import json
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


# ----------------------------------------------------------------------------- inputs
def load_mvtec(root, ds, a_type, index):
    """Load like the notebook: image_dataset_from_directory(..., 128x128, 'nearest')."""
    import tensorflow as tf
    folder = Path(root) / ds / "test" / a_type
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() == ".png")
    path = files[index]

    def read(p):
        raw = tf.io.decode_image(tf.io.read_file(str(p)), channels=3, expand_animations=False)
        return tf.image.resize(raw, (128, 128), method="nearest").numpy().astype(np.float32)

    xi = read(path) / 255.0
    if a_type == "good":
        gamma = np.zeros((128, 128), bool)
    else:
        gamma = read(Path(root) / ds / "ground_truth" / a_type / f"{path.stem}_mask.png")[..., 0] > 0
    return xi, gamma, {"file": f"{ds}/test/{a_type}/{path.name}"}


def _svg_images(repo, ds, a_type, index, part):
    spec = (f"origin/xai-2024-paper:results/{ds}/test_results/{a_type}/{index}/"
            f"{a_type}_{index}_5000_1_30000_100_blend_2_{part}.svg")
    text = subprocess.check_output(["git", "-C", str(repo), "show", spec]).decode()
    from PIL import Image
    images = []
    for b64 in re.findall(r'xlink:href="data:image/png;base64,([^"]+)"', text):
        im = Image.open(io.BytesIO(base64.b64decode(b64.replace("\n", "")))).convert("RGB")
        # Matplotlib's SVG backend stores images bottom-up and flips them on display.
        images.append(np.asarray(im.transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64) / 255)
    return images


_OPS = {}


def _hanning_operator(n_out, n_in=128):
    """Matrix of matplotlib's imshow resampling (hanning) from n_in to n_out pixels."""
    if (n_out, n_in) not in _OPS:
        import matplotlib._image as _im
        import matplotlib.image as mimg
        from matplotlib.transforms import Affine2D
        a = np.zeros((n_out, n_in))
        for i in range(n_in):
            d = np.zeros((n_in, n_in))
            d[i, :] = 1
            out = np.zeros((n_out, n_out))
            mimg.resample(d, out, Affine2D().scale(n_out / n_in), interpolation=_im.HANNING)
            a[:, i] = out[:, n_out // 2]
        _OPS[(n_out, n_in)] = a
    return _OPS[(n_out, n_in)]


def _unresample(y, lam=1e-3):
    a = _hanning_operator(y.shape[0])
    p = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T)
    return np.stack([p @ y[..., c] @ p.T for c in range(y.shape[2])], -1)


def load_archive(repo, ds, a_type, index):
    """Recover xi and gamma from the archived p1/p3 SVG figures of the paper run.

    The figures hold xi rendered at 248 px by imshow's hanning filter; that
    resampling is linear, so the 128 px input is recovered by a regularised
    inverse and re-quantised to 8 bits. gamma is read off the first IoU panel,
    where ground-truth pixels are drawn cyan (missed) or black (hit).
    """
    p1 = _svg_images(repo, ds, a_type, index, "p1")
    p3 = _svg_images(repo, ds, a_type, index, "p3")
    xi = (np.round(np.clip(_unresample(p1[0]), 0, 1) * 255) / 255).astype(np.float32)
    red = _unresample(p3[0][..., :1])[..., 0]          # red channel is 0 exactly on gamma
    gamma = red < 0.5
    return xi, gamma, {"file": f"recovered from results/{ds}/test_results/{a_type}/{index}/*_p1.svg, *_p3.svg"}


# ----------------------------------------------------------------------------- model
_MODELS = {}


def model_for(repo, ds):
    if ds not in _MODELS:
        sys.path.insert(0, str(repo))
        from tf_scripts.core import restore_models
        _MODELS[ds] = restore_models(repo / "models" / f"{ds}_VAE_GAN_30000", ds, 30000)
    return _MODELS[ds]


def reconstruct(model, batch):
    mu, _, _ = model.encoder(np.asarray(batch, np.float32))
    return model.decoder(mu).numpy()


def sse_fn(model, batch_size=50):
    """lime_predict_loss of the notebook: sum of squared reconstruction errors."""
    def f(data):
        data = np.asarray(data, np.float32)
        out = []
        for s in range(0, len(data), batch_size):
            chunk = data[s:s + batch_size]
            out.append(np.square(reconstruct(model, chunk) - chunk).sum(axis=(1, 2, 3)))
        return np.concatenate(out)[:, None]
    return f


# ----------------------------------------------------------------------------- segmentation
def quickshift(image, max_dist):
    from skimage.segmentation import quickshift as qs
    kw = dict(kernel_size=2, max_dist=max_dist, ratio=0.95, sigma=0.05)
    try:
        return qs(image, rng=1234, **kw)
    except TypeError:
        return qs(image, random_seed=1234, **kw)


def search_segment_number(image, target, init_max_dist=100):
    """The notebook's binary search on Quickshift's max_dist."""
    lmd, rmd = 0.0, float(init_max_dist)
    lsn = len(np.unique(quickshift(image, lmd)))
    rsn = len(np.unique(quickshift(image, rmd)))
    for _ in range(40):
        if rsn == target:
            break
        mmd = (lmd + rmd) / 2
        msn = len(np.unique(quickshift(image, mmd)))
        if msn <= target <= lsn:
            rsn, rmd = msn, mmd
        else:
            lsn, lmd = msn, mmd
    return quickshift(image, rmd)


# ----------------------------------------------------------------------------- LIME with a log
def lime_logged(xi, recon, segments, f, n, checkpoints, replay_rows):
    from lime import lime_base, lime_image
    from sklearn.metrics import pairwise_distances

    log = {}
    original = lime_image.LimeImageExplainer.data_labels

    def logged(self, *args, **kwargs):
        data, labels = original(self, *args, **kwargs)
        log["data"], log["labels"] = data.copy(), labels.copy()
        return data, labels

    lime_image.LimeImageExplainer.data_labels = logged
    try:
        explainer = lime_image.LimeImageExplainer(random_state=1234)
        expl = explainer.explain_instance(xi, f, hide_color=recon, num_samples=n,
                                          batch_size=50, segmentation_fn=lambda _: segments)
    finally:
        lime_image.LimeImageExplainer.data_labels = original

    data, labels = log["data"], log["labels"]
    label = expl.top_labels[0]
    k = data.shape[1]

    def beta_from(local):
        b = np.zeros(k, np.float64)
        for feat, w in local:
            b[feat] = w
        return b

    beta = beta_from(expl.local_exp[label])

    # Refit with LIME's own base on the first n rows (same kernel, same ridge).
    def kernel(d, kernel_width=0.25):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

    dist = pairwise_distances(data, data[0].reshape(1, -1), metric="cosine").ravel()
    steps = []
    for c in sorted(set(min(c, n) for c in checkpoints)):
        base = lime_base.LimeBase(kernel, random_state=explainer.random_state)
        intercept, local, score, _ = base.explain_instance_with_data(
            data[:c], labels[:c], dist[:c], label, num_features=100000, feature_selection="auto")
        steps.append({"n": int(c), "beta": beta_from(local), "intercept": float(intercept), "r2": float(score)})
    gap = float(np.abs(steps[-1]["beta"] - beta).max())
    if gap > 1e-6 * max(1.0, float(np.abs(beta).max())):
        raise RuntimeError(f"LIME log does not rebuild the explanation (max gap {gap})")
    return {"beta": beta, "intercept": float(expl.intercept[label]), "r2": float(expl.score),
            "steps": steps, "masks": data[:replay_rows].astype(np.uint8),
            "responses": labels[:, label].astype(np.float64)}


def shap_partition(xi, recon, f, max_evals):
    import shap
    masker = shap.maskers.Image(recon, recon.shape)
    explainer = shap.Explainer(f, masker, output_names=["Anomaly"])
    values = explainer(np.array([xi]), max_evals=max_evals, batch_size=50)
    return np.asarray(values.values)[0, :, :, 0, 0]            # the notebook's indexing


def optimal_iou(gamma, scores):
    """calc_IoU_curve of the paper's utils.py: IoU at every distinct cutoff."""
    s = scores.ravel()
    order = np.argsort(-s, kind="stable")
    ss, tt = s[order], gamma.ravel()[order]
    ends = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
    tp = np.cumsum(tt)[ends]
    sel = ends + 1
    iou = tp / (sel + gamma.sum() - tp)
    b = int(np.argmax(iou))
    return float(iou[b]), float(ss[ends[b]]), int(sel[b])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=HERE / "config.toml")
    ap.add_argument("--source", choices=["mvtec", "archive"])
    ap.add_argument("--mvtec-root")
    ap.add_argument("--repo", help="repository root (default: repo_root in config.toml)")
    ap.add_argument("--only", help="substring filter on dataset/type/index")
    ap.add_argument("--inline", action="store_true", help="do not start one process per sample")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text())
    source = a.source or cfg["source"]
    repo = Path(a.repo).resolve() if a.repo else (HERE / cfg["repo_root"]).resolve()
    b = cfg["budget"]
    OUT.mkdir(exist_ok=True)

    for s in cfg["sample"]:
        ds, t, i = s["dataset"], s["type"], int(s["index"])
        name = f"{ds}_{t}_{i}"
        if a.only and (a.only != name if a.inline else a.only not in name):
            continue
        if not a.inline:
            cmd = [sys.executable, __file__, "--inline", "--only", name, "--config", str(a.config),
                   "--source", source, "--repo", str(repo)]
            if a.mvtec_root:
                cmd += ["--mvtec-root", a.mvtec_root]
            subprocess.run(cmd, check=True)
            continue
        t0 = time.time()
        if source == "mvtec":
            xi, gamma, info = load_mvtec(a.mvtec_root or cfg["mvtec_root"], ds, t, i)
        else:
            xi, gamma, info = load_archive(repo, ds, t, i)
        model = model_for(repo, ds)
        f = sse_fn(model)
        recon = reconstruct(model, xi[None])[0]
        amap = np.abs(recon - xi).max(axis=2)
        alpha = float(amap.max())

        shap_map = shap_partition(xi, recon, f, b["shap_evals"])   # the heaviest step first
        import gc; gc.collect()
        settings = {}
        g01 = np.repeat(gamma[..., None], 3, -1).astype(np.float32)   # gamma in [0, 1], as in the paper run
        for tag, seg_input in (("S1", xi * 0.5 + recon * 0.5),
                               ("S2", xi * 0.33 + recon * 0.33 + g01 * 0.34)):
            if tag == "S2" and not gamma.any():
                continue
            segments = search_segment_number(seg_input, b["target_segments"])
            lime = lime_logged(xi, recon, segments, f, b["lime_samples"], b["checkpoints"], b["replay_rows"])
            heat = lime["beta"][segments]
            settings[tag] = dict(segments=segments, **lime,
                                 iou=optimal_iou(gamma, heat) if gamma.any() else None)
        arrays = dict(xi=xi, recon=recon, amap=amap, gamma=gamma, shap=shap_map)
        for tag, st in settings.items():
            arrays |= {f"{tag}_segments": st["segments"], f"{tag}_beta": st["beta"],
                       f"{tag}_masks": st["masks"], f"{tag}_responses": st["responses"],
                       f"{tag}_steps_n": [x["n"] for x in st["steps"]],
                       f"{tag}_steps_beta": np.stack([x["beta"] for x in st["steps"]]),
                       f"{tag}_steps_intercept": [x["intercept"] for x in st["steps"]],
                       f"{tag}_steps_r2": [x["r2"] for x in st["steps"]],
                       f"{tag}_intercept": st["intercept"]}
        np.savez_compressed(OUT / f"{name}.npz", **arrays)
        meta = dict(dataset=ds, type=t, index=i, note=s.get("note", ""), source=source, **info,
                    alpha=alpha, sse=float(f(xi[None])[0, 0]),
                    iou={tag: st["iou"] for tag, st in settings.items()}
                    | ({"S3": optimal_iou(gamma, shap_map)} if gamma.any() else {}),
                    seconds=round(time.time() - t0, 1))
        (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
        print(f"{name}: alpha={alpha:.4f}  IoU " +
              "  ".join(f"{k}={v[0]:.3f}" for k, v in meta["iou"].items() if v) +
              f"  ({meta['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
