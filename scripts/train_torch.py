#!/usr/bin/env python3
"""Train original-paper-inspired VAE-GAN using native PyTorch."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from anomaly_detection.config import load_config
from anomaly_detection.torch_train import fit

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--epochs",type=int)
    p.add_argument("--device",choices=["auto","cuda","cpu","mps"],default="auto")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--variant",choices=["initial","paper_faithful"],default="initial",
                   help="initial preserves previous training; paper_faithful matches original gradient ordering")
    p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--save-every",type=int,default=1)
    p.add_argument("--no-augmentation",action="store_true")
    a=p.parse_args()
    if a.epochs is not None and a.epochs<1:p.error("--epochs must be positive")
    if a.save_every<1:p.error("--save-every must be positive")
    cfg=load_config(a.config)
    if a.epochs is not None:cfg["epochs"]=a.epochs
    cfg["torch_variant"]=a.variant
    print(fit(cfg,device_name=a.device,resume=a.resume,num_workers=a.num_workers,
              save_every=a.save_every,no_augmentation=a.no_augmentation))
