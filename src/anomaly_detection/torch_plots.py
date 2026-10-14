"""Matplotlib loss-history plotting without importing pyplot during training startup."""
from pathlib import Path
import numpy as np


def save_training_plots(history, destination, epoch):
    """Overwrite stable PNG/PDF filenames with full loss histories every K epochs.

    Plot data come from the in-memory history already saved in the CSV;
    figures include all epochs from the beginning, including resumed history.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    x = np.asarray([row["epoch"] for row in history], dtype=int)
    groups = [
        ("vae_reconstruction", ("vae_loss", "reconstruction_loss")),
        ("kl", ("kl_loss",)),
        ("adversarial", ("disc_loss", "gen_loss")),
    ]
    for stem, keys in groups:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key in keys:
            values = np.asarray([row[key] for row in history], dtype=float)
            ax.plot(x, values, label=key, linewidth=1.6)
        ax.set(xlabel="Epoch", ylabel="Loss", title=f"{stem.replace('_', ' ').title()} — epoch {epoch}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(destination / f"{stem}.{ext}", dpi=200)
        plt.close(fig)
