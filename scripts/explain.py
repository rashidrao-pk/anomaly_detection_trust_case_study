#!/usr/bin/env python3
"""Generate LIME/SHAP image explanations from an existing TF checkpoint."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from anomaly_detection.config import load_config,resolve_path
from anomaly_detection.inference import load_model
from anomaly_detection.explain import explain_image
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--image",required=True)
    p.add_argument("--method",choices=["lime","shap"],default="lime")
    p.add_argument("--samples",type=int,default=100)
    p.add_argument("--scale",type=float,default=0.1,help="Fixed score-to-probability scale; not calibrated")
    p.add_argument("--output",default="outputs/explanation.png")
    p.add_argument("--checkpoint")
    args=p.parse_args()
    cfg=load_config(args.config)
    model=load_model(cfg,args.checkpoint or resolve_path(cfg,cfg["checkpoint"]))
    explain_image(cfg,model,args.image,args.method,args.samples,args.scale,args.output)
    print("Saved:",args.output)
