#!/usr/bin/env python3
"""Train and evaluate the optional Anomalib EfficientAd baseline (issue #2).

Requires a separate PyTorch/Anomalib environment; never imports TensorFlow.
"""
import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from anomaly_detection.config import load_config, resolve_path


def validate_dataset(root, category):
    root = Path(root)
    for suffix in (f"{category}/train/good", f"{category}/test",
                   f"{category}/ground_truth"):
        if not (root / suffix).is_dir():
            raise FileNotFoundError(f"Missing MVTec AD directory: {root / suffix}")
    return root


def build_settings(cfg, category, epochs=None):
    settings = cfg.get("efficientad", {})
    n = int(epochs if epochs is not None else settings.get("max_epochs", 20))
    if n <= 0:
        raise ValueError("max_epochs must be positive")
    if category not in ("screw", "hazelnut"):
        raise ValueError("Issue #2 initially supports screw and hazelnut")
    return settings, n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--category", choices=("screw", "hazelnut"),
                        help="Override category in YAML")
    parser.add_argument("--epochs", type=int, help="Override efficientad.max_epochs")
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu", "mps"),
                        default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    category = args.category or cfg["category"]
    settings, epochs = build_settings(cfg, category, args.epochs)
    root = validate_dataset(resolve_path(cfg, cfg["dataset_root"]), category)
    output = resolve_path(cfg, cfg.get("output_dir", "outputs")) / "baselines" / "efficientad" / category
    output.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from anomalib.data import MVTecAD
        from anomalib.models import EfficientAd
        from anomalib.engine import Engine
    except ImportError as exc:
        raise SystemExit(
            "EfficientAD requires a separate PyTorch environment. "
            "See docs/EFFICIENTAD.md for installation."
        ) from exc

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    from lightning.pytorch import seed_everything
    seed_everything(seed, workers=True)
    device = args.accelerator or settings.get("accelerator", "auto")
    datamodule = MVTecAD(
        root=str(root), category=category,
        train_batch_size=int(settings.get("train_batch_size", 1)),
        eval_batch_size=int(settings.get("eval_batch_size", 1)),
        num_workers=int(settings.get("num_workers", 0)),
    )
    model = EfficientAd(
        imagenet_dir=str(settings.get("imagenet_dir", "./datasets/imagenette")),
        model_size=str(settings.get("model_size", "s")),
        lr=float(settings.get("lr", 1e-4)),
    )
    engine = Engine(
        max_epochs=epochs, accelerator=device, devices=1,
        default_root_dir=str(output),
    )
    engine.fit(model=model, datamodule=datamodule)
    test_results = engine.test(model=model, datamodule=datamodule)
    manifest = {
        "baseline": "EfficientAd", "category": category,
        "dataset_root": str(root), "epochs": epochs,
        "seed": seed, "accelerator_requested": device,
        "torch_version": torch.__version__,
        "checkpoint": str(engine.checkpoint_callback.best_model_path)
            if getattr(engine, "checkpoint_callback", None) is not None else None,
        "test_results": test_results,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "note": "This is Anomalib's native test output; separate per-image map "
                "export and VAE-GAN metric parity are not yet implemented.",
    }
    manifest_path = output / "test_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[EfficientAD] Saved: {manifest_path}")


if __name__ == "__main__":
    main()
