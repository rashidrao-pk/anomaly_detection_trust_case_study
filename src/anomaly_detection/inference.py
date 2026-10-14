"""Checkpoint restoration and reconstruction prediction."""
from pathlib import Path
import numpy as np
from .scoring import reconstruction_error, image_scores

def load_model(cfg, checkpoint=None):
    from .models import build_model
    model = build_model(cfg.get("latent_dim",64), cfg.get("image_size",128))
    if checkpoint is not None:
        checkpoint = str(checkpoint)
    else:
        checkpoint = str(cfg["checkpoint"])
    if not Path(checkpoint + ".index").exists():
        raise FileNotFoundError(f"Checkpoint prefix does not exist: {checkpoint}")
    # Archived TF checkpoint names may require explicit partial status for
    # optimizer slots; do not hide incompatible model variable assignments.
    try:
        status = model.load_weights(checkpoint)
    except ValueError as exc:
        if "incompatible tensor with shape" in str(exc):
            raise ValueError(
                f"Archived checkpoint is incompatible with the configured model. "
                f"Check latent_dim={cfg.get('latent_dim')} in your YAML, the "
                f"selected checkpoint={checkpoint}, and the original model "
                f"architecture. TensorFlow reported: {exc}"
            ) from exc
        raise
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
