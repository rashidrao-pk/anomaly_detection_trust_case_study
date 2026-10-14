#!/usr/bin/env python3
"""Run original-paper-inspired LIME/SHAP against a PyTorch VAE-GAN checkpoint."""
import argparse
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/"src"),str(root)]
from anomaly_detection.config import load_config,resolve_path
from anomaly_detection.data import image_files
from anomaly_detection.torch_explain import explain_torch_image
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--checkpoint",help="Default: PyTorch checkpoint matching config epochs")
    p.add_argument("--image",help="Explicit test image path")
    p.add_argument("--anomaly-type")
    p.add_argument("--index",type=int)
    p.add_argument("--method",choices=("lime","shap"),default="lime")
    p.add_argument("--samples",type=int,default=100)
    p.add_argument("--device",choices=("auto","cpu","mps","cuda"),default="auto")
    p.add_argument("--output")
    a=p.parse_args()
    if a.samples<2:p.error("--samples must be >=2")
    cfg=load_config(a.config)
    if a.image:
        image=Path(a.image).expanduser()
    else:
        kind=a.anomaly_type or cfg.get("explanation",{}).get("anomaly_type","good")
        index=a.index if a.index is not None else cfg.get("explanation",{}).get("image_index",0)
        choices=image_files(resolve_path(cfg,cfg["dataset_root"])/cfg["category"]/"test"/kind)
        image=choices[index]
    if not image.is_file():p.error(f"Image not found: {image}")
    output=a.output
    if not output:
        output=str(resolve_path(cfg,cfg.get("output_dir","outputs"))/"pytorch"/
                   "explanations"/f"{image.parent.name}_{image.stem}_{a.method}.png")
    explain_torch_image(cfg,a.checkpoint,image,method=a.method,num_samples=a.samples,
                        output=output,device_name=a.device)
    print(f"Saved {output}")
