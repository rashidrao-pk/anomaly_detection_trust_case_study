"""PyTorch port of the archived 128x128 VAE-GAN architecture.

Architecture and losses are traced to repo-root models.py; numerical parity with
TensorFlow is NOT claimed (initialization, kernels, augmentation and RNG differ).
"""
import torch
from torch import nn
from torch.nn import functional as F


class SameConv2d(nn.Module):
    """TensorFlow SAME padding, including asymmetric padding for even kernels."""
    def __init__(self, in_ch, out_ch, kernel, stride=1):
        super().__init__()
        self.kernel, self.stride = kernel, stride
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding=0)

    def forward(self, x):
        h, w = x.shape[-2:]
        oh = (h + self.stride - 1) // self.stride
        ow = (w + self.stride - 1) // self.stride
        ph = max((oh - 1) * self.stride + self.kernel - h, 0)
        pw = max((ow - 1) * self.stride + self.kernel - w, 0)
        x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.features = nn.Sequential(
            SameConv2d(3, 32, 3, 2), nn.ReLU(), nn.BatchNorm2d(32, eps=1e-3, momentum=.01),
            SameConv2d(32, 64, 3, 2), nn.ReLU(), nn.BatchNorm2d(64, eps=1e-3, momentum=.01),
            SameConv2d(64, 128, 3, 2), nn.ReLU(), nn.BatchNorm2d(128, eps=1e-3, momentum=.01),
            SameConv2d(128, 256, 3, 2), nn.ReLU(),
            nn.Flatten(), nn.Linear(8 * 8 * 256, 160), nn.Tanh(),
        )
        self.mean = nn.Linear(160, latent_dim)
        self.logvar = nn.Linear(160, latent_dim)

    def forward(self, images, sample=True):
        z = self.features(images)
        mean, logvar = self.mean(z), self.logvar(z)
        if sample:
            sampled = mean + torch.exp(.5 * logvar) * torch.randn_like(mean)
        else:
            sampled = mean
        return mean, logvar, sampled


class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 8 * 8 * 64), nn.ReLU())
        blocks = []
        channels = [64, 256, 128, 64, 32]
        for i in range(4):
            blocks.extend([
                nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2), nn.ReLU(),
                SameConv2d(channels[i+1], channels[i+1], 3), nn.ReLU(),
            ])
            if i < 3:
                blocks.append(nn.BatchNorm2d(channels[i+1], eps=1e-3, momentum=.01))
        blocks.extend([SameConv2d(32, 3, 3), nn.Sigmoid()])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, latent):
        return self.blocks(self.fc(latent).reshape(-1, 64, 8, 8))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            SameConv2d(3, 128, 8, 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.BatchNorm2d(128, eps=1e-3, momentum=.01),
            SameConv2d(128, 64, 5, 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.BatchNorm2d(64, eps=1e-3, momentum=.01),
            SameConv2d(64, 32, 4, 2), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 2 * 2, 128), nn.ReLU(), nn.Linear(128, 1),
        )

    def forward(self, images):
        return self.features(images)  # logits: BCEWithLogitsLoss


class VAEGAN(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.discriminator = Discriminator()

    def forward(self, images, sample=True):
        mean, logvar, z = self.encoder(images, sample=sample)
        return mean, logvar, self.decoder(z)

    def reconstruct(self, images):
        return self.decoder(self.encoder(images, sample=False)[0])


def paper_losses(images, recon, mean, logvar, real_logits, fake_logits):
    # tf.keras.losses.binary_crossentropy averages over RGB channels,
    # then original code sums spatial dimensions and averages batch.
    pixel_bce = F.binary_cross_entropy(recon, images, reduction="none").mean(dim=1)
    reconstruction = pixel_bce.sum(dim=(1, 2)).mean()
    kl = (-.5 * (1 + logvar - mean.square() - logvar.exp())).sum(dim=1).mean()
    real_targets = torch.zeros_like(real_logits)
    fake_targets = torch.ones_like(fake_logits)
    disc_loss = F.binary_cross_entropy_with_logits(
        torch.cat((real_logits, fake_logits)), torch.cat((real_targets, fake_targets))
    )
    # The original Keras labels are inverted: real=0, generated=1.
    generator_bce = F.binary_cross_entropy_with_logits(fake_logits, real_targets)
    gen_loss = torch.clamp_min(generator_bce - disc_loss, .0001)
    return {"reconstruction_loss": reconstruction, "kl_loss": kl,
            "disc_loss": disc_loss, "gen_loss": gen_loss,
            "vae_loss": reconstruction + kl + gen_loss}
