#!/usr/bin/env python3
"""Generate a reconstruction error map and optional LIME image explanation."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tf_scripts.core import restore_models, load_rgb, reconstruct, residual_map
from tf_scripts.config import apply_config, require_args, resolve_path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config", help="YAML configuration from configs/")
    p.add_argument("--dataset-root", help="Override dataset root in YAML")
    p.add_argument("--checkpoint-dir")
    p.add_argument("--category", choices=["screw","hazelnut"])
    p.add_argument("--checkpoint-epochs",type=int,default=None)
    p.add_argument("--latent-dim",type=int,default=None)
    p.add_argument("--image")
    p.add_argument("--method",choices=["residual","lime"],default=None)
    p.add_argument("--num-samples",type=int,default=None)
    p.add_argument("--output",default=None)
    a=p.parse_args()
    a = apply_config(a, p, "explain")
    require_args(a, p, ("category", "checkpoint_dir", "image"))
    a.checkpoint_epochs = 30000 if a.checkpoint_epochs is None else int(a.checkpoint_epochs)
    a.latent_dim = 32 if a.latent_dim is None else int(a.latent_dim)
    a.method = a.method or "residual"
    a.num_samples = 1000 if a.num_samples is None else int(a.num_samples)
    a.output = str(resolve_path(a.output or "outputs/tensorflow/xai"))
    a.checkpoint_dir = str(resolve_path(a.checkpoint_dir))
    a.image = str(resolve_path(a.image))
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model=restore_models(a.checkpoint_dir,a.category,a.checkpoint_epochs,a.latent_dim)
    image=load_rgb(a.image)
    rec=reconstruct(model,image[None])[0]
    err=residual_map(image,rec)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    panels=[("Input",image),("Reconstruction",rec),("Error map",err)]
    if a.method=="lime":
        from lime import lime_image
        def predict(batch):
            batch=np.asarray(batch,dtype=np.float32)
            recon=reconstruct(model,batch)
            scores=np.square(batch-recon).sum(axis=(1,2,3))
            return np.column_stack([np.zeros_like(scores),scores])
        explanation=lime_image.LimeImageExplainer(random_state=42).explain_instance(
            image,predict,top_labels=1,hide_color=0,num_samples=a.num_samples)
        mask=explanation.get_image_and_mask(
            1,positive_only=True,num_features=5,hide_rest=False)[1]
        panels.append(("LIME positive superpixels", mask))
    fig,axes=plt.subplots(1,len(panels),figsize=(4*len(panels),4))
    for ax,(title,arr) in zip(axes,panels):
        if arr.ndim==2:
            ax.imshow(arr,cmap="Reds" if title=="Error map" else "gray")
        else:
            ax.imshow(np.clip(arr,0,1))
        ax.set_title(title);ax.axis("off")
    fig.tight_layout();dst=out/(Path(a.image).stem+"_"+a.method+".png")
    fig.savefig(dst,dpi=200);plt.close(fig)
    print(dst)


if __name__=="__main__":
    main()
