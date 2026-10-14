"""Research figures, including signed XAI 2024 attribution maps."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def save_explanation(image, reconstruction, anomaly_map, explanation, destination, signed=True):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.5))
    arrays = [image, reconstruction, anomaly_map, explanation]
    titles = ["Input", "Reconstruction", "Anomaly map", "Explanation (signed)"]
    for i, (ax, data, title) in enumerate(zip(axs, arrays, titles)):
        if i == 3 and signed:
            magnitude = max(float(np.max(np.abs(data))), 1e-12)
            im = ax.imshow(data, cmap="bwr", vmin=-magnitude, vmax=magnitude)
            fig.colorbar(im, ax=ax, fraction=0.047, pad=0.02)
        elif data.ndim == 2:
            ax.imshow(data, cmap="magma")
        else:
            ax.imshow(data)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)
