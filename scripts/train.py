#!/usr/bin/env python3
"""Train the original TensorFlow VAE-GAN from a CLI instead of a notebook."""
import argparse
import csv
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tf_scripts.core import model_parts, checkpoint_prefix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True, help="MVTec root containing screw/ and hazelnut/")
    p.add_argument("--category", choices=["screw", "hazelnut"], required=True)
    p.add_argument("--epochs", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--augmentation", choices=["custom", "minimal", "full", "none"], default="custom")
    p.add_argument("--output", default="outputs/tensorflow")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.latent_dim < 1:
        p.error("epochs, batch-size and latent-dim must be positive")
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from utils import Data2
    tf.random.set_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    train_dir = Path(args.dataset_root) / args.category / "train"
    if not (train_dir / "good").is_dir():
        p.error(f"No MVTec normal-training directory: {train_dir / 'good'}")
    if args.augmentation == "none":
        datagen = ImageDataGenerator(rescale=1 / 255.0)
    elif args.augmentation == "minimal":
        datagen = ImageDataGenerator(rescale=1 / 255.0,
            preprocessing_function=Data2.apply_random_transformation)
    elif args.augmentation == "custom":
        options = dict(rotation_range=40, width_shift_range=.05,
            height_shift_range=.05, brightness_range=(.8, 1.2),
            shear_range=.05, zoom_range=.05, fill_mode="nearest",
            rescale=1 / 255.0,
            preprocessing_function=Data2.apply_random_transformation)
        if args.category == "hazelnut":
            options.update(horizontal_flip=True, vertical_flip=True)
        datagen = ImageDataGenerator(**options)
    else:
        p.error("full augmentation must be verified against the archived notebook before use")
    generator = datagen.flow_from_directory(str(train_dir), classes=["good"],
        target_size=(128,128), batch_size=args.batch_size,
        class_mode=None, shuffle=True, seed=args.seed)
    model = model_parts(args.latent_dim, args.learning_rate)
    out = Path(args.output) / f"{args.category}_VAE_GAN_{args.epochs}"
    if out.exists() and any(out.iterdir()):
        p.error(f"Output directory is not empty: {out}; use a fresh directory to protect checkpoints")
    out.mkdir(parents=True, exist_ok=True)
    meta = vars(args)
    (out / "run_config.json").write_text(json.dumps(meta, indent=2))
    history = model.fit(generator, steps_per_epoch=len(generator), epochs=args.epochs)
    for component, object_ in [
        ("encoder", model.encoder), ("decoder", model.decoder),
        ("discriminator", model.discriminator), ("vae", model.vae),
        ("model", model),
    ]:
        path = checkpoint_prefix(out, args.category, args.epochs, component)
        object_.save_weights(str(path))
    keys = list(history.history)
    with (out / f"{args.category}_VAE_GAN_{args.epochs}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", *keys])
        writer.writeheader()
        for i in range(args.epochs):
            writer.writerow({"epoch": i + 1, **{k: history.history[k][i] for k in keys}})
    print(f"Saved TensorFlow component checkpoints and CSV to {out}")


if __name__ == "__main__":
    main()
