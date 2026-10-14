#!/usr/bin/env python3
"""Run the new score-evaluation pipeline on screw and hazelnut archives.

This does not reproduce the paper's original numeric values or Jaccard protocol;
the original XAI 2024 notebooks remain preserved on xai-2024-paper.
"""
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=="__main__":
    for category in ("screw","hazelnut"):
        subprocess.run([sys.executable,str(ROOT/"scripts/evaluate.py"),"--config",str(ROOT/"configs"/f"{category}.yaml")],check=True,cwd=ROOT)
