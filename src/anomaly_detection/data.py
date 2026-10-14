"""MVTec AD image and annotation loading with deterministic sample alignment."""
from pathlib import Path
import numpy as np
from PIL import Image

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

def image_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)

def load_rgb(path, size=128):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0

def load_mask(path, size=128):
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.NEAREST)
        return np.asarray(image, dtype=np.uint8) > 0

def training_paths(root, category):
    paths = image_files(Path(root) / category / "train" / "good")
    if not paths:
        raise ValueError(f"No training images for category {category}")
    return paths

def evaluation_samples(root, category):
    """Yield image path, image label, pixel-mask path or None.

    MVTec anomaly mask naming convention: test/type/name.png ->
    ground_truth/type/name_mask.png. Do not attempt numeric sort or reorder masks.
    """
    folder = Path(root) / category / "test"
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    for kind in sorted(p for p in folder.iterdir() if p.is_dir()):
        for path in image_files(kind):
            defective = kind.name != "good"
            mask = Path(root) / category / "ground_truth" / kind.name / (path.stem + "_mask.png")
            if defective and not mask.exists():
                raise FileNotFoundError(f"Missing mask for {path}: {mask}")
            yield path, int(defective), mask if defective else None

def make_train_dataset(root, category, image_size, batch_size, seed=42, shuffle=True):
    import tensorflow as tf
    paths = training_paths(root, category)
    ds = tf.data.Dataset.from_tensor_slices([str(p) for p in paths])
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    def decode(path):
        pixels = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        pixels.set_shape([None, None, 3])
        pixels = tf.image.resize(pixels, (image_size, image_size))
        return tf.cast(pixels, tf.float32) / 255.0
    return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
