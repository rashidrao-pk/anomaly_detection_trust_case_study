"""Explicit image-level and pixel-level reconstruction-based anomaly scores."""
import numpy as np

def reconstruction_error(images, reconstructions):
    images = np.asarray(images, dtype=np.float32)
    reconstructions = np.asarray(reconstructions, dtype=np.float32)
    if images.shape != reconstructions.shape:
        raise ValueError(f"Input/reconstruction shapes differ: {images.shape} vs {reconstructions.shape}")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Expected RGB batches with shape (N,H,W,3)")
    return np.mean(np.abs(images - reconstructions), axis=-1)

def image_scores(pixel_maps):
    maps = np.asarray(pixel_maps)
    if maps.ndim != 3:
        raise ValueError("Expected anomaly maps of shape (N,H,W)")
    return np.max(maps, axis=(1,2))
