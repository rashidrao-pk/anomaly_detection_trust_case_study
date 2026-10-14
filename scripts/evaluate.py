#!/usr/bin/env python3
"""Evaluate archived or freshly trained model."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from anomaly_detection.config import load_config, resolve_path
from anomaly_detection.evaluation import evaluate
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--checkpoint",help="Override the config checkpoint prefix")
    args=p.parse_args()
    cfg=load_config(args.config)
    print(evaluate(cfg, args.checkpoint))
