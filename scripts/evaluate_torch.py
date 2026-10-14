#!/usr/bin/env python3
"""Evaluate trained PyTorch checkpoint with the same modern scoring as TensorFlow."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from anomaly_detection.config import load_config
from anomaly_detection.torch_train import evaluate
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--device",choices=["auto","cuda","cpu","mps"],default="auto")
    p.add_argument("--num-workers",type=int,default=0)
    a=p.parse_args()
    print(evaluate(load_config(a.config),a.checkpoint,device_name=a.device,num_workers=a.num_workers))
