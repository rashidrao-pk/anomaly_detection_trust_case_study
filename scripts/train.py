#!/usr/bin/env python3
"""Train with notebook augmentation, with optional checkpoint resume."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
from anomaly_detection.config import load_config
from anomaly_detection.train import train
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--resume",action="store_true",help="Continue existing run using its CSV and five checkpoint components")
    p.add_argument("--resume-epochs",type=int,help="Additional epochs when resuming")
    args=p.parse_args()
    cfg=load_config(args.config)
    if args.resume_epochs is not None: cfg["resume_epochs"]=args.resume_epochs
    print("Saved checkpoint prefix:",train(cfg,resume=args.resume))
