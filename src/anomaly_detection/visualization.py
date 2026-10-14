"""Deterministic, non-interactive output for research figures."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_explanation(image, reconstruction, anomaly_map, explanation, destination):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(1,4,figsize=(14,3.5))
    for ax,data,title in zip(axs,[image,reconstruction,anomaly_map,explanation],
                            ["Input","Reconstruction","Anomaly map","Explanation"]):
        ax.imshow(data,cmap="magma" if getattr(data,"ndim",0)==2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(destination,dpi=200,bbox_inches="tight")
    plt.close(fig)
