"""MVTec paper ImageDataGenerator-compatible CPU augmentation.

Mimics original data.py make_paper_train_generator: Keras random affine
(rotation, translation, shear, isotropic zoom), nearest fill, bilinear sampling,
brightness, optional hazelnut flips, then custom quarter-turn rotation.
This is not bit-identical to TF (RNG and Pillow/Scipy version differ).
"""
import random
import numpy as np
from PIL import Image, ImageEnhance


def paper_aligned_transform(image, category):
    # scipy is deliberately imported lazily: pure unit CI does not need it.
    from scipy.ndimage import affine_transform
    from torchvision.transforms import functional as TF
    rng = random
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    theta = np.deg2rad(rng.uniform(-40, 40))
    tx, ty = rng.uniform(-.05, .05) * h, rng.uniform(-.05, .05) * w
    shear = rng.uniform(-.05, .05)  # Keras shear_range is in degrees
    shear = np.deg2rad(shear)
    zoom_y = rng.uniform(.95, 1.05)
    zoom_x = rng.uniform(.95, 1.05)
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=np.float64)
    shift = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    shear_m = np.array([[1, -np.sin(shear), 0],
                        [0, np.cos(shear), 0], [0, 0, 1]], dtype=np.float64)
    zoom = np.diag([zoom_y, zoom_x, 1.0])
    transform = rotation @ shift @ shear_m @ zoom
    offset_x, offset_y = (h / 2.0 + .5), (w / 2.0 + .5)
    offset = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
    reset = np.array([[1, 0, -offset_x], [0, 1, -offset_y], [0, 0, 1]])
    transform = offset @ transform @ reset
    warped = np.stack([
        affine_transform(arr[:, :, ch], transform[:2, :2],
                         offset=transform[:2, 2], order=1, mode="nearest")
        for ch in range(3)], axis=-1)
    warped = Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8), "RGB")
    warped = ImageEnhance.Brightness(warped).enhance(rng.uniform(.8, 1.2))
    if category == "hazelnut":
        if rng.random() < .5: warped = TF.hflip(warped)
        if rng.random() < .5: warped = TF.vflip(warped)
    elif category != "screw":
        raise ValueError("Paper preprocessing is defined for screw and hazelnut only")
    # Corresponds to paper_rotation: random 0/1/2 -> k=1/3/1.
    turn = rng.choice((1, 3, 1))
    return warped.rotate(90 * turn, resample=Image.Resampling.NEAREST)
