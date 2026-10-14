# Training VAE-GAN on Epito (Slurm / BeeGFS)

This is an HPC adaptation of the **TensorFlow VAE-GAN** pipeline in main.
It is not the separate PyTorch EfficientAD baseline.

## 1. Log in and inspect resources

```bash
ssh mrashid@mercurio-login
sinfo -o '%P %G %D %l'
squeue -u "$USER"
cd /beegfs/home/mrashid/repos/anomaly_detection_trust_case_study
git switch main
git pull --ff-only origin main
mkdir -p logs
```

If the repository does not exist on BeeGFS, clone it once using the correct
GitHub SSH credentials. Verify the chosen Slurm partition and `--gres` using
`sinfo`/`scontrol show partition epito`; cluster partition/GPUs and limits
vary. The example `scripts/hpc/train_slurm.sh` assumes `epito` and one GPU,
but does not guarantee your site's exact resource syntax.

## 2. Create the Linux environment (on a compute allocation if needed)

```bash
conda create -p /beegfs/home/mrashid/envs/xai-ad -y python=3.10
conda activate /beegfs/home/mrashid/envs/xai-ad
python -m pip install -U pip wheel setuptools
python -m pip install 'tensorflow[and-cuda]==2.15.0'
python -m pip install -e '.[test]'
python -m pip check
python -c 'import tensorflow as tf; print(tf.__version__,tf.config.list_physical_devices("GPU"))'
```

Some HPC sites require activating CUDA modules or a specific driver. Test
TensorFlow on an allocated **GPU compute node**, not on the login node.
Anomalib/EfficientAD uses a separate PyTorch environment. The legacy
TensorFlow Addons dependency is currently declared by the project, though
the newer training path does not need to import it directly.

## 3. Point to the actual dataset

Set `DATASET_ROOT` to the MVTec root containing
`screw/train/good`, `screw/test`, `screw/ground_truth`
and equivalent Hazelnut directories. Example:

```bash
export DATASET_ROOT=/beegfs/home/mrashid/datasets/AD/MVTec
test -d "$DATASET_ROOT/screw/train/good" || echo 'Edit DATASET_ROOT first'
```

The path above is **illustrative**; inspect your existing BeeGFS folders
and correct it. No dataset is downloaded or copied.

## 4. Submit 100-epoch Screw training

```bash
export CONDA_ENV_PATH=/beegfs/home/mrashid/envs/xai-ad
export CATEGORY=screw
export EPOCHS=100
export OUT_ROOT=/beegfs/home/mrashid/experiments/anomaly_detection/screw_100
mkdir -p logs "$OUT_ROOT"
sbatch --export=ALL,DATASET_ROOT="$DATASET_ROOT",CONDA_ENV_PATH="$CONDA_ENV_PATH",CATEGORY="$CATEGORY",EPOCHS="$EPOCHS",OUT_ROOT="$OUT_ROOT" scripts/hpc/train_slurm.sh
```

Change `CATEGORY=hazelnut` and `OUT_ROOT` for a second job.
The script creates a run-local YAML file and never changes the macOS YAML.
It fails rather than overwriting an existing `_model` checkpoint or CSV.

Watch:

```bash
squeue -u "$USER"
tail -f logs/xai-screw-<SLURM_JOB_ID>.out
```

The job runs training, one-per-class reconstruction QA, and AUROC evaluation
using **the new checkpoint** (not the archive).

## 5. Checkpoint and output files

```text
screw_100/
  config_epito_screw_100.yaml
  trained_models/screw_VAE_GAN_100_model.index
  trained_models/screw_VAE_GAN_100_model.data-00000-of-00001
  trained_models/screw_VAE_GAN_100_{vae,encoder,decoder,discriminator}.index
  trained_models/screw_VAE_GAN_100.csv
  checkpoint_tests/screw_VAE_GAN_100_model/
  metrics.json
  image_scores.csv
```

## Important operational limits

- The trainer saves checkpoint components and CSV **after the entire requested
  fit completes**. If Slurm kills the job for walltime/OOM, unsaved progress is
  lost. Start with 10-100 epochs or use shorter staged runs; periodic checkpoint
  callbacks are a future improvement.
- Do not run training on the login node.
- The current training generator uses the paper augmentation profile. Exact
  scientific parity with the historical notebook is not yet fully established.
- `evaluate.py` writes results to the configured output directory; an
  archived/new model comparison should be done separately after both
  checkpoints are accessible on Epito.


## Python 3.10 and Conda unavailable (Epito02)

If `python3.10` and `conda` are not available but `python3.11` is installed,
create the environment using the standard library without admin privileges:

```bash
python3.11 -m venv /beegfs/home/mrashid/envs/xai-ad
source /beegfs/home/mrashid/envs/xai-ad/bin/activate
python --version
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'tensorflow[and-cuda]==2.15.0'
cd ~/repos/AD/anomaly_detection_trust_case_study
python -m pip install -e '.[test]'
python -m pip check
```

Python 3.11 is permitted by `pyproject.toml` (`>=3.9,<3.12`).
TensorFlow 2.15 CUDA availability must be tested on an allocated GPU compute
node. Check `nvidia-smi` then run:

```bash
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

The existing Slurm script names the variable `CONDA_ENV_PATH`, but actually
uses its `bin/python` executable, so the same variable may point to this venv.
If `venv` fails because `ensurepip` is unavailable, use a user-installed
Python distribution or ask cluster support instead of attempting privileged
system installations. TensorFlow Addons is an old optional dependency in the
current packaging: confirm `pip check` and imports before submitting jobs.


## Python interpreter absent on Epito02 (diagnostics)

Do not assume the OS's suggested Python packages are installed. Run:

```bash
command -v python3; python3 --version
ls -l /usr/bin/python3* 2>/dev/null
command -v module || true
command -v uv || true
command -v pixi || true
command -v micromamba || true
module avail python 2>&1 || true
module avail anaconda 2>&1 || true
ls -ld /beegfs/home/mrashid/envs/* 2>/dev/null || true
```

If a cluster module or existing environment provides Python 3.10 or 3.11,
use that interpreter to create a venv. Otherwise install a user-space Python
using an HPC-approved tool such as micromamba/uv/pixi, subject to site policy,
or ask cluster support. Never use sudo or install system packages on compute
nodes. Verify GPU access in the Slurm allocation and keep Python 3.12 out of
the TensorFlow 2.15 environment.
