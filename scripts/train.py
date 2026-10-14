#!/usr/bin/env python3
"""Train VAE-GAN: python scripts/train.py --config configs/screw.yaml"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from anomaly_detection.config import load_config
from anomaly_detection.train import train
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    args=p.parse_args()
    print("Saved checkpoint prefix:",train(load_config(args.config)))
