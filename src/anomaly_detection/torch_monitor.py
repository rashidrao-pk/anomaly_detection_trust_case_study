"""Periodic fixed-pair reconstructions for qualitative training diagnostics.

Uses separate seeded RNG and unaugmented images, so it does not consume the
training sampler's random stream. Produces PNG grids and pair metadata.
"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.transforms import functional as TF

from .data import evaluation_samples


def choose_pair(training_paths, dataset_root, category, seed):
    normal = [Path(p) for p in training_paths]
    anomalous = [(Path(path), Path(mask) if mask else None)
                 for path, label, mask in evaluation_samples(dataset_root, category)
                 if int(label) == 1]
    if not normal or not anomalous:
        raise ValueError("Monitor requires both normal training images and anomalous test images")
    rng = random.Random(seed + 99173)
    return [(normal[rng.randrange(len(normal))], None, "normal_train"),
            (*anomalous[rng.randrange(len(anomalous))], "anomalous_test")]


def write_pair_metadata(pair, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "examples.json").write_text(json.dumps(
        [{"split": split, "image": str(path), "mask": str(mask) if mask else None}
         for path, mask, split in pair], indent=2))


@torch.no_grad()
def save_monitor_image(model, pair, epoch, device, output_dir, image_size=128):
    previous_mode = model.training
    model.eval()
    try:
        canvas = Image.new("RGB", (image_size * 4, image_size * 2 + 54), "white")
        draw = ImageDraw.Draw(canvas)
        for idx, (path, mask_path, label) in enumerate(pair):
            with Image.open(path) as img:
                input_img = img.convert("RGB").resize(
                    (image_size, image_size), Image.Resampling.BILINEAR)
            tensor = TF.to_tensor(input_img).unsqueeze(0).to(device)
            pred = model.reconstruct(tensor).clamp(0, 1)[0].cpu()
            reconstructed = TF.to_pil_image(pred)
            err = (tensor[0].cpu() - pred).abs().mean(0).numpy()
            # Fixed scale [0,1] for cross-epoch visual comparability.
            heat = np.uint8(np.clip(err, 0, 1) * 255)
            error_img = Image.fromarray(heat, mode="L").convert("RGB")
            if mask_path is not None and mask_path.exists():
                with Image.open(mask_path) as m:
                    mask = m.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
                mask_img = Image.fromarray(
                    (np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L").convert("RGB")
            else:
                mask_img = Image.new("RGB", (image_size, image_size), "black")
            y = 25 + idx * (image_size + 2)
            for j, image in enumerate((input_img, reconstructed, error_img, mask_img)):
                canvas.paste(image, (j * image_size, y))
            draw.text((4, y - 18), label, fill="black")
        canvas.save(output_dir / f"epoch_{epoch:06d}.png")
        return output_dir / f"epoch_{epoch:06d}.png"
    finally:
        model.train(previous_mode)
