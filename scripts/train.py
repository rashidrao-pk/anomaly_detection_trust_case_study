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
    p.add_argument("--epochs",type=int,help="Override YAML epochs for a new run (also changes checkpoint name)")
    p.add_argument("--resume",action="store_true",help="Continue existing run using its CSV and five checkpoint components")
    p.add_argument("--resume-epochs",type=int,help="Additional epochs when resuming")
    args=p.parse_args()
    if args.epochs is not None and args.epochs <= 0: p.error("--epochs must be positive")
    if args.resume_epochs is not None and args.resume_epochs <= 0: p.error("--resume-epochs must be positive")
    if args.epochs is not None and args.resume: p.error("Use --resume-epochs for continuing a run; --epochs is for new runs")
    if args.resume_epochs is not None and not args.resume: p.error("--resume-epochs requires --resume")
    cfg=load_config(args.config)
    if args.epochs is not None: cfg["epochs"]=args.epochs
    if args.resume_epochs is not None: cfg["resume_epochs"]=args.resume_epochs
    print("Saved checkpoint prefix:",train(cfg,resume=args.resume))
