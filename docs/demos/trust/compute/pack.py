#!/usr/bin/env python3
"""Pack the computed samples and the full test set into the demo's data.

  out/*.npz, out/*.json        from explain.py
  xai-2024-paper branch        test-set scores and IoUs (CSV) and the paper's Fig. 2

writes ../app/assets/data.js (window.ADX_DATA = {...}) and, with --standalone,
a single self-contained HTML page for sharing.

The x axis of the paper's Fig. 2 (anomaly scores from the notebook's batched
pass over the test set) is not stored in the CSVs, so it is read back from the
vector Fig. 2 PDFs: point positions are mapped through the axis ticks, and each
anomalous point is matched to its CSV row by its three IoU values (S1, S2, S3).
"""
import argparse
import base64
import csv
import io
import json
import subprocess
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CSV = {"hazelnut": "testresults_220_100_30000_5000_hazelnut_blend_2_['LIME_SHAP'].csv",
       "screw": "testresults_320_100_30000_5000_screw_blend_2_['LIME_SHAP'].csv"}
FIG2 = {"S1": "lime_S1", "S2": "lime_S2", "S3": "shap_S3"}
COLOURS = {(0.0, 0.5, 0.0): "anom_ok", (0.0, 0.0, 1.0): "good_ok",
           (1.0, 0.0, 0.0): "anom_mis", (1.0, 0.65, 0.0): "good_mis"}


def git_show(repo, path):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"origin/xai-2024-paper:{path}"])


# ----------------------------------------------------------------------------- encoding
def b64(raw):
    return base64.b64encode(raw).decode()


def png(img):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.clip(np.round(np.asarray(img) * 255), 0, 255).astype(np.uint8)).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + b64(buf.getvalue())


def f32(a):
    return b64(np.ascontiguousarray(np.asarray(a, "<f4")).tobytes())


def u8(a):
    return b64(np.ascontiguousarray(np.asarray(a, np.uint8)).tobytes())


def bits(a):
    return b64(np.packbits(np.asarray(a, bool).ravel()).tobytes())


# ----------------------------------------------------------------------------- Fig. 2
def fig2_points(pdf_bytes):
    import pymupdf
    page = pymupdf.open(stream=pdf_bytes, filetype="pdf")[0]
    pts, xt, yt = [], [], []
    for d in page.get_drawings():
        r = d["rect"]
        fill = tuple(round(c, 2) for c in (d.get("fill") or ()))
        if fill in COLOURS and abs(r.width - r.height) < 0.2 and r.width < 4:
            pts.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, COLOURS[fill]))
        elif abs(r.height) < 0.01 and 3 < r.width < 4:
            yt.append(r.y0)
        elif abs(r.width) < 0.01 and 3 < r.height < 4:
            xt.append(r.x0)

    def num(s):
        try:
            return float(s)
        except ValueError:
            return None
    words = page.get_text("words")
    xl = [(num(w[4]), (w[0] + w[2]) / 2) for w in words if num(w[4]) is not None and w[1] > 90]
    yl = [(num(w[4]), (w[1] + w[3]) / 2) for w in words if num(w[4]) is not None and w[0] < 35 and w[1] < 90]

    def fit(labels, ticks):
        v = np.array([a for a, _ in labels])
        t = np.array([min(ticks, key=lambda u: abs(u - c)) for _, c in labels])
        coef = np.polyfit(t, v, 1)
        assert np.abs(np.polyval(coef, t) - v).max() < 1e-6, "axis ticks do not fit a line"
        return coef
    ax, ay = fit(xl, xt), fit(yl, yt)
    return [(float(np.polyval(ax, x)), float(np.polyval(ay, y)), c) for x, y, c in pts]


def full_test_set(repo, tau):
    from scipy.optimize import linear_sum_assignment
    result = {}
    for ds, fn in CSV.items():
        rows = list(csv.DictReader(io.StringIO(git_show(repo, f"results/{ds}/{fn}").decode())))
        s1 = {(r["a_type"], int(r["img_no"])): r for r in rows if r["mask_type"] == "blend_2"}
        s2 = {(r["a_type"], int(r["img_no"])): r for r in rows if r["mask_type"] == "blend_3"}
        P = {k: fig2_points(git_show(repo, f"results/{ds}/plt_{ds}_{v}.pdf")) for k, v in FIG2.items()}
        pts = []
        for x, y, c in P["S1"]:
            y2 = min(P["S2"], key=lambda p: abs(p[0] - x) + 9 * (p[2] != c))
            y3 = min(P["S3"], key=lambda p: abs(p[0] - x) + 9 * (p[2] != c))
            pts.append((x, c, (y, y2[1], y3[1])))
        anom = [p for p in pts if p[1].startswith("anom")]
        keys = sorted(k for k in s1 if k[0] != "good")

        def ious(k):
            return (float(s1[k]["IoU_lm_max"]), float(s2[k]["IoU_lm_max"]), float(s1[k]["IoU_sp_max"]))
        cost = np.array([[sum(abs(a - b) for a, b in zip(p[2], ious(k))) for k in keys] for p in anom])
        ri, ci = linear_sum_assignment(cost)
        assert cost[ri, ci].max() < 1e-3, "Fig. 2 points do not match the CSV"
        samples = [dict(t=keys[c][0], i=keys[c][1], a=round(anom[r][0], 5), y=1,
                        iou=[round(v, 4) for v in ious(keys[c])]) for r, c in zip(ri, ci)]
        samples += [dict(t="good", i=None, a=round(p[0], 5), y=0, iou=None)
                    for p in sorted(pts) if p[1].startswith("good")]
        samples.sort(key=lambda s: s["a"])
        result[ds] = dict(tau=tau[ds], samples=samples)
    return result


# ----------------------------------------------------------------------------- samples
def pack_sample(npz, meta, full):
    z = np.load(npz)
    ds = meta["dataset"]
    row = next((s for s in full[ds]["samples"] if s["t"] == meta["type"] and s["i"] == meta["index"]), None)
    out = dict(id=npz.stem, ds=ds, type=meta["type"], index=meta["index"], note=meta["note"],
               file=meta["file"], source=meta["source"], alpha=meta["alpha"],
               alpha_paper=row["a"] if row else None, paper_iou=row["iou"] if row else None,
               xi=png(z["xi"]), recon=png(z["recon"]), amap=f32(z["amap"]),
               gamma=bits(z["gamma"]), shap=f32(z["shap"]), iou=meta["iou"])
    for tag in ("S1", "S2"):
        if f"{tag}_beta" not in z:
            continue
        masks = z[f"{tag}_masks"]
        out[tag] = dict(
            seg=u8(z[f"{tag}_segments"]), k=int(z[f"{tag}_beta"].shape[0]),
            beta=f32(z[f"{tag}_beta"]), b0=float(z[f"{tag}_intercept"]),
            rows=int(masks.shape[0]), masks=bits(masks), resp=f32(z[f"{tag}_responses"]),
            steps=[dict(n=int(n), beta=f32(b), b0=float(i), r2=float(r)) for n, b, i, r in
                   zip(z[f"{tag}_steps_n"], z[f"{tag}_steps_beta"], z[f"{tag}_steps_intercept"], z[f"{tag}_steps_r2"])])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=HERE / "config.toml")
    ap.add_argument("--repo", help="repository root (default: repo_root in config.toml)")
    ap.add_argument("--standalone", help="also write one self-contained HTML page here")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text())
    repo = Path(a.repo).resolve() if a.repo else (HERE / cfg["repo_root"]).resolve()
    full = full_test_set(repo, cfg["threshold"])
    order = [f'{s["dataset"]}_{s["type"]}_{s["index"]}' for s in cfg["sample"]]
    samples = []
    for name in order:
        if (OUT / f"{name}.npz").exists():
            samples.append(pack_sample(OUT / f"{name}.npz", json.loads((OUT / f"{name}.json").read_text()), full))
    sources = sorted({s["source"] for s in samples})
    data = dict(source=sources[0] if len(sources) == 1 else "mixed", budget=cfg["budget"],
                full=full, samples=samples)
    js = "window.ADX_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n"
    assets = HERE.parent / "app" / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "data.js").write_text(js)
    print(f"{len(samples)} samples, {len(js) / 1e6:.2f} MB -> {assets / 'data.js'}")
    if a.standalone:
        page = (HERE.parent / "app" / "index.html").read_text()
        tag = '<script src="assets/data.js"></script>'
        assert tag in page
        page = page.split("\n", 1)[1]          # the artifact host supplies doctype, html and meta
        Path(a.standalone).write_text(page.replace(tag, "<script>" + js + "</script>"))
        print("standalone ->", a.standalone)


if __name__ == "__main__":
    main()
