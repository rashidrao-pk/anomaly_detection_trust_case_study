"""Train VAE-GAN; save all five checkpoints and CSV as in the paper notebook."""
from pathlib import Path
import csv
import random
import numpy as np
from .config import resolve_path
from .data import make_train_dataset
from .models import build_model

def checkpoint_prefix(cfg):
    output = resolve_path(cfg, cfg.get("output_dir", "outputs")) / "trained_models"
    return output / f"{cfg['category']}_VAE_GAN_{int(cfg.get('epochs', 10))}"

def save_notebook_checkpoints(model, prefix, history):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, component in (
        ("_model", model), ("_vae", model.vae),
        ("_encoder", model.vae.encoder), ("_decoder", model.vae.decoder),
        ("_discriminator", model.discriminator)
    ):
        component.save_weights(str(prefix) + suffix, save_format="tf")
    with Path(str(prefix) + ".csv").open("w", newline="") as stream:
        fields = list(history.history)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(zip(fields, row)) for row in zip(*(history.history[k] for k in fields)))
    return str(prefix) + "_model"

def train(cfg):
    import tensorflow as tf
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    root = resolve_path(cfg, cfg["dataset_root"])
    data = make_train_dataset(root, cfg["category"], int(cfg.get("image_size",128)),
                              int(cfg.get("batch_size",8)),seed)
    model = build_model(int(cfg.get("latent_dim",32)),
                        int(cfg.get("image_size",128)),
                        float(cfg.get("learning_rate",0.0001)))
    history = model.fit(data, epochs=int(cfg.get("epochs",10)), verbose=1)
    return save_notebook_checkpoints(model, checkpoint_prefix(cfg), history)
