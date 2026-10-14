"""Train the original VAE-GAN architecture through a reusable function."""
from pathlib import Path
import random
import numpy as np
from .config import resolve_path
from .data import make_train_dataset
from .models import build_model

def train(cfg):
    import tensorflow as tf
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    root = resolve_path(cfg, cfg["dataset_root"])
    output = resolve_path(cfg, cfg.get("output_dir", "outputs"))
    output.mkdir(parents=True, exist_ok=True)
    data = make_train_dataset(root, cfg["category"], int(cfg.get("image_size",128)), int(cfg.get("batch_size",8)),seed)
    model = build_model(
        int(cfg.get("latent_dim",64)),
        int(cfg.get("image_size",128)),
        float(cfg.get("learning_rate",0.0001)),
    )
    history = model.fit(data, epochs=int(cfg.get("epochs",10)), verbose=1)
    prefix = str(output / "vae_gan")
    model.save_weights(prefix)
    import json
    with (output / "history.json").open("w") as fp:
        json.dump({k:[float(x) for x in vals] for k,vals in history.history.items()},fp,indent=2)
    return prefix
