"""Checkpoint restoration and reconstruction prediction."""
from pathlib import Path
import numpy as np
from .models import build_model
from .scoring import reconstruction_error, image_scores

def load_model(cfg, checkpoint=None):
    model = build_model(cfg.get("latent_dim",64), cfg.get("image_size",128))
    if checkpoint is not None:
        checkpoint = str(checkpoint)
    else:
        checkpoint = str(cfg["checkpoint"])
    if not Path(checkpoint + ".index").exists():
        raise FileNotFoundError(f"Checkpoint prefix does not exist: {checkpoint}")
    # Archived TF checkpoint names may require explicit partial status for
    # optimizer slots; do not hide incompatible model variable assignments.
    status = model.load_weights(checkpoint)
    if hasattr(status, "expect_partial"):
        status.expect_partial()
    return model

def reconstruct(model, images, deterministic=True):
    """Use posterior mean to make repeated evaluation deterministic."""
    images = np.asarray(images,dtype=np.float32)
    if deterministic:
        z_mean, _, _ = model.encoder(images, training=False)
        rec = model.decoder(z_mean, training=False)
    else:
        _, rec = model(images, training=False)
    return np.asarray(rec)

def predict_scores(model, images):
    rec = reconstruct(model, images)
    maps = reconstruction_error(images, rec)
    return image_scores(maps), maps, rec
