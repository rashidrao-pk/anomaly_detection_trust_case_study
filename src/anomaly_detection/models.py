"""Compatibility bridge to the original TensorFlow 2.15 VAE-GAN architecture.

The compatible model architecture lives in the repo-root models.py. The
published implementation is preserved unchanged on xai-2024-paper.
"""
from models import Model_VAE_GAN_functions, VAE, VAE_GAN

def build_model(latent_dim=64, image_size=128, learning_rate=1e-4):
    if image_size != 128:
        raise ValueError("The archived decoder only supports image_size=128")
    import tensorflow as tf
    funcs = Model_VAE_GAN_functions
    enc = funcs.get_encoder(latent_dim=latent_dim, image_size=[128,128])
    dec = funcs.get_decoder(latent_dim=latent_dim)
    disc = funcs.get_discriminator(shape=(128,128,3))
    vae = VAE(enc, dec)
    model = VAE_GAN(
        vae, disc,
        opti1=tf.keras.optimizers.legacy.Adam(learning_rate),
        opti2=tf.keras.optimizers.legacy.Adam(learning_rate),
        opti3=tf.keras.optimizers.legacy.Adam(learning_rate),
    )
    model.compile(run_eagerly=False)
    # Build variables before loading TensorFlow checkpoints.
    import numpy as np
    model(np.zeros((1,128,128,3),dtype=np.float32))
    model.discriminator(np.zeros((1,128,128,3),dtype=np.float32))
    return model
