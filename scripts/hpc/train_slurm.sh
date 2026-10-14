#!/usr/bin/env bash
#SBATCH --job-name=xai-screw
#SBATCH --partition=epito
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/xai-%x-%j.out
#SBATCH --error=logs/xai-%x-%j.err

set -euo pipefail
# Submit from repo root after running: mkdir -p logs
# E.g. DATASET_ROOT=/beegfs/home/.../MVtec EPOCHS=100 sbatch scripts/hpc/train_slurm.sh
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
: "${DATASET_ROOT:?Set DATASET_ROOT to your MVTec AD root (containing screw/train/good)}"
: "${CONDA_ENV_PATH:?Set CONDA_ENV_PATH to the xai-ad environment directory}"
CATEGORY="${CATEGORY:-screw}"
EPOCHS="${EPOCHS:-100}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/outputs/hpc/$CATEGORY}"
if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then echo "EPOCHS must be a positive integer" >&2; exit 2; fi
if [[ "$CATEGORY" != screw && "$CATEGORY" != hazelnut ]]; then echo "CATEGORY must be screw or hazelnut" >&2; exit 2; fi
if [[ ! -d "$DATASET_ROOT/$CATEGORY/train/good" ]]; then echo "Missing dataset: $DATASET_ROOT/$CATEGORY/train/good" >&2; exit 2; fi
if [[ ! -x "$CONDA_ENV_PATH/bin/python" ]]; then echo "Missing environment Python: $CONDA_ENV_PATH/bin/python" >&2; exit 2; fi
mkdir -p "$OUT_ROOT"
RUN_PREFIX="$OUT_ROOT/trained_models/${CATEGORY}_VAE_GAN_${EPOCHS}"
if [[ -e "${RUN_PREFIX}_model.index" || -e "${RUN_PREFIX}.csv" ]]; then
    echo "Refusing to overwrite existing run: $RUN_PREFIX (choose a new OUT_ROOT)" >&2
    exit 2
fi
export DATASET_ROOT CATEGORY OUT_ROOT
CFG="$OUT_ROOT/config_epito_${CATEGORY}_${EPOCHS}.yaml"
export CFG
"$CONDA_ENV_PATH/bin/python" - <<'PY'
import os
from pathlib import Path
import yaml
root=Path.cwd()
src=root/"configs"/(os.environ["CATEGORY"]+".yaml")
cfg=yaml.safe_load(src.read_text())
cfg["dataset_root"]=str(Path(os.environ["DATASET_ROOT"]).expanduser().resolve())
cfg["output_dir"]=str(Path(os.environ["OUT_ROOT"]).expanduser().resolve())
target=Path(os.environ["CFG"])
target.write_text(yaml.safe_dump(cfg,sort_keys=False))
print("[config]",target)
PY
echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-none} category=$CATEGORY epochs=$EPOCHS"
"$CONDA_ENV_PATH/bin/python" - <<'PY'
import tensorflow as tf
print("[tensorflow]",tf.__version__,"GPUs:",tf.config.list_physical_devices("GPU"))
if not tf.config.list_physical_devices("GPU"):
    raise SystemExit("No GPU visible; refusing to start a long CPU training job")
PY
"$CONDA_ENV_PATH/bin/python" scripts/train.py --config "$CFG" --epochs "$EPOCHS"
"$CONDA_ENV_PATH/bin/python" scripts/test_checkpoint.py --config "$CFG" --epochs "$EPOCHS"
"$CONDA_ENV_PATH/bin/python" scripts/evaluate.py --config "$CFG" --checkpoint "${RUN_PREFIX}_model"
echo "[done] checkpoints: ${RUN_PREFIX}_model ; output dir: $OUT_ROOT"
