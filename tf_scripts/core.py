"""Script-first wrappers around the original XAI-2024 TensorFlow VAE-GAN.

The original repository-root models.py remains the authority on architecture
and optimization. No changes to its paper loss or discriminator labels.
"""
from pathlib import Path
import numpy as np


def model_parts(latent_dim=32, learning_rate=1e-3):
    import tensorflow as tf
    from models import Model_VAE_GAN_functions, VAE, VAE_GAN, make_adam
    builder = Model_VAE_GAN_functions
    encoder = builder.get_encoder(latent_dim=latent_dim, image_size=[128, 128])
    decoder = builder.get_decoder(latent_dim=latent_dim)
    discriminator = builder.get_discriminator(shape=(128, 128, 3))
    vae = VAE(encoder, decoder)
    # Original train_step owns optimizers; use independent objects per instance.
    model = VAE_GAN(vae, discriminator,
        opti1=make_adam(learning_rate=learning_rate),
        opti2=make_adam(learning_rate=learning_rate),
        opti3=make_adam(learning_rate=learning_rate))
    model.compile(run_eagerly=True)
    return model


def checkpoint_prefix(folder, category, epochs, component="model"):
    return Path(folder) / f"{category}_VAE_GAN_{epochs}_{component}"


def restore_models(folder, category, epochs, latent_dim=32):
    """Restore existing paper component weights, without needing optimizer state."""
    root = Path(folder)
    vae_gan = model_parts(latent_dim=latent_dim)
    # Build all layers before loading historical component checkpoints.
    import tensorflow as tf
    _ = vae_gan(tf.zeros((1, 128, 128, 3), dtype=tf.float32))
    _ = vae_gan.discriminator(tf.zeros((1, 128, 128, 3), dtype=tf.float32))
    enc = checkpoint_prefix(root, category, epochs, "encoder")
    dec = checkpoint_prefix(root, category, epochs, "decoder")
    for component, prefix in ((vae_gan.encoder, enc), (vae_gan.decoder, dec)):
        if not Path(str(prefix) + ".index").exists():
            raise FileNotFoundError(f"Missing TensorFlow checkpoint: {prefix}.index")
        component.load_weights(str(prefix))
    return vae_gan


def load_rgb(path, size=128):
    from PIL import Image
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize(
            (size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def reconstruct(model, images, deterministic=True):
    images = np.asarray(images, dtype=np.float32)
    mean, logvar, sampled = model.encoder(images, training=False)
    z = mean if deterministic else sampled
    return model.decoder(z, training=False).numpy()


def residual_map(images, reconstruction):
    return np.abs(np.asarray(images) - np.asarray(reconstruction)).mean(axis=-1)
