#!/usr/bin/env bash
#SBATCH --job-name=vaegan-torch
#SBATCH --partition=epito
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vaegan-torch-%j.out
#SBATCH --error=logs/vaegan-torch-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repo root}"
: "${DATASET_ROOT:?Specify DATASET_ROOT, containing screw/train/good}"
PYTHON_BIN="${PYTHON_BIN:-/beegfs/home/mrashid/pt_312/bin/python}"
CATEGORY="${CATEGORY:-screw}"
EPOCHS="${EPOCHS:-10}"
OUT_ROOT="${OUT_ROOT:-/beegfs/home/mrashid/experiments/vaegan_torch/$CATEGORY}"
if [[ "$CATEGORY" != screw && "$CATEGORY" != hazelnut ]]; then echo "Invalid CATEGORY" >&2; exit 2; fi
if [[ ! "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then echo "EPOCHS must be positive" >&2; exit 2; fi
test -d "$DATASET_ROOT/$CATEGORY/train/good"
test -x "$PYTHON_BIN"
mkdir -p "$OUT_ROOT"
export PYTHONPATH=""
export CATEGORY DATASET_ROOT OUT_ROOT
RUN_CONFIG="$OUT_ROOT/${CATEGORY}_torch.yaml"
export RUN_CONFIG
"$PYTHON_BIN" - <<'PY'
import os, yaml
from pathlib import Path
cfg=yaml.safe_load(Path("configs",os.environ["CATEGORY"]+".yaml").read_text())
cfg["dataset_root"]=str(Path(os.environ["DATASET_ROOT"]).resolve())
cfg["output_dir"]=str(Path(os.environ["OUT_ROOT"]).resolve())
Path(os.environ["RUN_CONFIG"]).write_text(yaml.safe_dump(cfg,sort_keys=False))
PY
"$PYTHON_BIN" - <<'PY'
import torch
print("PyTorch:",torch.__version__,"CUDA:",torch.version.cuda,"GPUs:",torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("Refusing training: GPU unavailable")
PY
"$PYTHON_BIN" scripts/train_torch.py --config "$RUN_CONFIG" --epochs "$EPOCHS" --device cuda --num-workers 4
"$PYTHON_BIN" scripts/evaluate_torch.py --config "$RUN_CONFIG" --device cuda \
    --checkpoint "$OUT_ROOT/pytorch/trained_models/${CATEGORY}_VAE_GAN_${EPOCHS}.pt"
echo "Finished: $OUT_ROOT/pytorch"
