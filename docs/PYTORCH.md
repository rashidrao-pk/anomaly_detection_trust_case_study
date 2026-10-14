# PyTorch VAE-GAN: architecture and paper comparability

This **native PyTorch** implementation is additive. The original TensorFlow
files and the published notebooks on `xai-2024-paper` are preserved.
Do not load TensorFlow `.index`/ `.data` files with PyTorch: new
`.pt` checkpoints must be trained. No cross-framework parity results
have yet been measured.

## Epito A100 setup

The existing PyTorch `pt_312` environment may work if it has
`torch`, `torchvision`, `numpy`, `Pillow`, `PyYAML` and
`scikit-learn`; do not export a Python 3.12 PyTorch path into Python 3.10.

```bash
source /beegfs/home/mrashid/pt_312/bin/activate
cd ~/repos/AD/anomaly_detection_trust_case_study
git pull --ff-only origin main
python -c 'import torch, torchvision; print(torch.__version__,torch.cuda.is_available())'
python -m pip install numpy pillow pyyaml scikit-learn
```

If those packages already exist, do not reinstall them. The optional
dependency group `.[pytorch]` also declares needed Python packages, but
**avoid `pip install -e .`** in a PyTorch-only environment because the
package's required dependencies currently include TensorFlow.

Use a **copied YAML** with Linux-specific dataset/output paths, for example
`/beegfs/home/mrashid/datasets/AD/MVTec`. Do not edit the original archived
paper notebooks. Verify `$DATASET_ROOT/screw/train/good` exists.

```bash
cp configs/screw.yaml configs/screw_epito.local.yaml
# edit dataset_root and output_dir in the copy
python scripts/train_torch.py --config configs/screw_epito.local.yaml --epochs 10 --device cuda
python scripts/evaluate_torch.py --config configs/screw_epito.local.yaml --device cuda --checkpoint outputs/screw/pytorch/trained_models/screw_VAE_GAN_10.pt
```

Replace the checkpoint path if `output_dir` was customized. After the smoke
test, use a fresh output directory and `--epochs 100` to train a full run.
Training checkpoints are written **every epoch** by default (including model,
both optimizer states, epoch, history, parameters), and `--resume` continues
to the total `--epochs` selected. Keep `--epochs` consistent when resuming,
since it is part of the file name. Avoid two jobs writing the same checkpoint.

## Paper logic traced to `models.py`

| Component | Original | New PyTorch |
| --- | --- | --- |
| Encoder | conv 32/64/128/256, 128px RGB, dense 160, mean/logvar | same logical layers, NCHW layout |
| Decoder | 8×8×64, four upsampling blocks, sigmoid RGB | same logical shapes |
| Discriminator | conv 128/64/32, pool, dense 128/1 | same logical shapes; BCE logits |
| VAE | sampling, spatial-sum RGB-averaged BCE plus summed KL | same loss scaling |
| Adversarial | real label 0, generated label 1, clamped generator BCE minus discriminator BCE | same label convention/formula |
| Optimizers | Adam 1e-3 for VAE and discriminator | same LR (default PyTorch Adam epsilon differs) |
| Input | 128 RGB, /255, MVTec screw/hazelnut | 128 RGB, /255 |
| Image score | max of mean-absolute-RGB pixel errors | same current refactor scoring |
| Pixel score | mean-absolute-RGB error | same current refactor scoring |

## Explicit differences / limits

- Keras `padding='same'` is approximated with exact asymmetric shape-padding
  logic. Layer initialization and framework kernel behavior remain different.
- The PyTorch augmentation uses PIL/torchvision affine/brightness operations
  with parameter ranges inspired by the Keras `ImageDataGenerator`. It
  **does not exactly duplicate** preprocessing transform selection,
  interpolation, fill-mode and random streams.
- The PyTorch adversarial update is **sequential** (discriminator then VAE)
  while the Keras original computes losses in shared GradientTapes before
  both optimizer updates; BatchNorm statistics also differ. Results must be
  treated as a research reimplementation rather than numerical reproduction.
- The PyTorch LIME/SHAP CLI reproduces the *refactored* paper explanation\n  targets (posterior-mean reconstruction MSE, Quickshift masks, SHAP image\n  masker), but it has not passed qualitative/numerical parity checks with\n  archived explanations. The original paper-figure regeneration and\n  TensorFlow checkpoint weights are **not** converted to PyTorch.
- Original notebook's training time and epochs cannot be inferred from
  `epochs: 10`, which is a smoke-test setting.
- Prior result for the **archived TensorFlow** Screw checkpoint: image AUROC
  ~0.9248 and pixel AUROC ~0.9403 using the modern refactor evaluator.
  These are **not PyTorch results**, nor proof of matching the original
  paper's reported numbers.

## Reproducibility record

For each run save: Git commit, copied YAML, GPU, CUDA/PyTorch versions, seed,
training duration, checkpoint path and the evaluator's `metrics.json`.
Report independently the archived TF checkpoint and the new PyTorch checkpoint;
avoid claiming framework parity without matched runs and augmentation audits.


## PyTorch explanations

Install optional explanation dependencies in the PyTorch environment:
```bash
python -m pip install lime shap scikit-image matplotlib
python scripts/explain_torch.py --config configs/screw_epito.local.yaml \\
  --method lime --samples 1000 --device cuda
python scripts/explain_torch.py --config configs/screw_epito.local.yaml \\
  --method shap --samples 100000 --device cuda
```
SHAP image masking is computationally costly. Outputs go to
`<output_dir>/pytorch/explanations/` (PNG and NPY). This is an
implementation of the *existing refactored explanation algorithm*,
not a verified exact replay of the 2024 notebook.

## Scheduled Epito A100 training (no TensorFlow)

Use a compute-node-accessible PyTorch environment (defaults to
`/beegfs/home/mrashid/pt_312/bin/python`). The scheduler script generates
a Linux-specific YAML for the selected category. Confirm the dataset path
before submitting:

```bash
cd ~/repos/AD/anomaly_detection_trust_case_study
git pull --ff-only origin main
mkdir -p logs
export DATASET_ROOT=/beegfs/home/mrashid/datasets/AD/MVTec
export PYTHON_BIN=/beegfs/home/mrashid/pt_312/bin/python
export OUT_ROOT=/beegfs/home/mrashid/experiments/vaegan_torch/screw_test
test -d "$DATASET_ROOT/screw/train/good"
"$PYTHON_BIN" -c "import torch,torchvision,sklearn,yaml,PIL; print(torch.__version__,torch.cuda.is_available())"
CATEGORY=screw EPOCHS=10 sbatch --export=ALL scripts/hpc/train_torch_slurm.sh
squeue -u "$USER"
```

For 100 epochs choose a **different** `OUT_ROOT` and set `EPOCHS=100`.
Check `sinfo` / `scontrol show partition epito` if the cluster changes
partition or GRES syntax. The script does not mutate the archived YAML.
