"""Shared configuration and dataset helpers for XAI-2024 paper figures."""
from pathlib import Path
import numpy as np
from PIL import Image
from tf_scripts.config import load_run_config, resolve_path


def context(config, output=None):
    cfg = load_run_config(config)
    root = resolve_path(cfg["dataset_root"]) / cfg["category"]
    destination = resolve_path(output or str(resolve_path(cfg["output_dir"]) / "paper_figures"))
    destination.mkdir(parents=True, exist_ok=True)
    return cfg, root, destination


def image_paths(root, include_good=True, one_per_class=False, index=0):
    base = root / "test"
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset missing: {base}")
    kinds = sorted(p for p in base.iterdir() if p.is_dir())
    if not include_good:
        kinds = [p for p in kinds if p.name != "good"]
    paths = []
    for folder in kinds:
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".png")
        if files:
            if one_per_class:
                if index >= len(files):
                    raise ValueError(f"{folder} contains only {len(files)} images; index={index}")
                paths.append(files[index])
            else:
                paths.extend(files)
    if not paths:
        raise ValueError(f"No test PNG images found in {base}")
    return paths


def gt_for_image(root, path, size=(128,128)):
    if path.parent.name == "good":
        return np.zeros((size[1], size[0]), dtype=bool)
    path_mask = root / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
    if not path_mask.exists():
        raise FileNotFoundError(f"Missing annotation: {path_mask}")
    with Image.open(path_mask) as im:
        return np.asarray(im.convert("L").resize(size, Image.Resampling.NEAREST)) > 0


def canonical_path(value):
    return str(Path(value).expanduser().resolve())


def save_figure(fig, output_stem, dpi=250):
    import matplotlib.pyplot as plt
    path = Path(str(output_stem) + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path
