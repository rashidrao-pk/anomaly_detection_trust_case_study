#!/usr/bin/env python3
"""Evaluate freshly trained checkpoint first; compare with published archive."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT))
from anomaly_detection.config import load_config
from anomaly_detection.comparison import compare_models
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--new-checkpoint",help="Path to newly trained _model checkpoint prefix")
    p.add_argument("--original-checkpoint",help="Path to archived _model checkpoint prefix")
    args=p.parse_args()
    print(json.dumps(compare_models(load_config(args.config),args.new_checkpoint,args.original_checkpoint),indent=2))
