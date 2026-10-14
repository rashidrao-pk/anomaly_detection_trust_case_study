"""Torch MVTec data loader and paper-inspired training augmentation."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from .data import training_paths, evaluation_samples


class PaperTrainingDataset(Dataset):
    def __init__(self, root, category, image_size=128, seed=42, augmentation="paper"):
        self.paths = training_paths(root, category)
        self.category, self.size, self.augmentation = category, image_size, augmentation

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as img:
            image = img.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        if self.augmentation == "paper":
            # Corresponds in *range* to ImageDataGenerator, not identical
            # transforms, interpolation, RNG ordering or pixel alignment.
            import random
            turns = random.choice((3, 3, 1))
            # Legacy paper_rotation applied two flips then rot90 for one branch;
            # both flips compose a half-turn, equivalent to rot90(k=1).
            if turns == 3 and random.random() < 0:  # intentionally unused
                pass
            image = image.rotate(90 if turns == 1 else 270)
            angle = random.uniform(-40, 40)
            dx = int(random.uniform(-.05, .05) * self.size)
            dy = int(random.uniform(-.05, .05) * self.size)
            shear = random.uniform(-.05, .05) * 180 / np.pi
            zoom = random.uniform(.95, 1.05)
            image = TF.affine(image, angle=angle, translate=[dx, dy], scale=zoom,
                              shear=[shear, 0.], interpolation=InterpolationMode.BILINEAR)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(.8, 1.2))
            if self.category == "hazelnut":
                if random.random() < .5:
                    image = TF.hflip(image)
                if random.random() < .5:
                    image = TF.vflip(image)
        elif self.augmentation != "none":
            raise ValueError(self.augmentation)
        return TF.to_tensor(image)


class EvaluationDataset(Dataset):
    def __init__(self, root, category, image_size=128):
        self.samples = list(evaluation_samples(root, category))
        self.size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, mask_path = self.samples[idx]
        with Image.open(path) as img:
            image = TF.to_tensor(img.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR))
        if mask_path is None:
            mask = torch.zeros((self.size, self.size), dtype=torch.bool)
        else:
            with Image.open(mask_path) as img:
                mask = torch.from_numpy(np.array(img.convert("L").resize(
                    (self.size, self.size), Image.Resampling.NEAREST)) > 0)
        return image, torch.tensor(label), mask, str(path)
