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

## Training loss diagnostics

The native trainer logs all five per-epoch components, including the same
three headline terms shown by the TensorFlow notebook:
`vae_loss`, `disc_loss`, `gen_loss`. It additionally prints
`reconstruction_loss` and `kl_loss`. Discriminator loss is displayed
with eight decimal places rather than rounding to four decimals, since
near-zero losses can indicate discriminator saturation. These metrics are
also written to the checkpoint history CSV. The reported adversarial and
discriminator values use a **sequential** discriminator/VAE step, unlike
the original shared-gradient-tape TensorFlow update; differences do not by
themselves establish faithful paper reproduction.

Pulling this logging change does **not** change the output of an already
running Python process. Stop/restart or resume from a checkpoint to use the
new logging. Keep the current checkpoints intact.

## Three-model paper comparability experiment (October 2026)

1. **Original TensorFlow**: evaluate the archived `.index` checkpoint using
   the original/refactored TF evaluator with exactly the same dataset root,
   masks, resolution and scoring. Its previously reported Screw refactor
   AUROCs (~0.9248 image / ~0.9403 pixel) are historical, not remeasured.
2. **Initial PyTorch**: existing `..._30000.pt` training with sequential
   discriminator-first update. This checkpoint is preserved and remains
   the default `--variant initial`. Its early discriminator loss was
   approximately 4e-7 at epoch 23; evaluate rather than presuming failure.
3. **Paper-faithful PyTorch**: new `--variant paper_faithful`. It computes
   discriminator and VAE gradients from one parameter state and applies
   them only after both have been computed, matching the ordering in
   repo-root `models.py`. It also uses Adam epsilon 1e-7 (Keras legacy
   default). The original Keras discriminator is called on a combined
   real/reconstruction batch and again on reconstructed images; both calls
   are mirrored here. Checkpoint suffix: `_paper_faithful.pt`.

Important: the corrected implementation is *more faithful to the original
optimization ordering*, **not proven numerically equivalent**. Framework
initialization, BatchNorm execution, generator data augmentation, RNG,
optimizer details and preprocessing still differ. It uses the same
inverted real/fake label convention as the published code, intentionally.

From the existing Epito PyTorch venv:

```bash
git pull --ff-only origin main
# Ensure configs/screw_epito.yaml points to the correct MVTec path
# Train a small corrected run (separate checkpoint filename):
python scripts/train_torch.py --config configs/screw_epito.yaml \
  --variant paper_faithful --epochs 100 --device cuda --num-workers 4
# Or evaluate an existing initial checkpoint:
python scripts/evaluate_torch.py --config configs/screw_epito.yaml \
  --checkpoint outputs/screw_epito/pytorch/trained_models/screw_VAE_GAN_30000.pt --device cuda
# Compare from the separately exported metrics.json files:
python scripts/compare_three_models.py \
  --tensorflow outputs/reference_tf/metrics.json \
  --initial outputs/initial_torch/metrics.json \
  --corrected outputs/corrected_torch/metrics.json \
  --output outputs/three_way
```

The comparison script requires **actual evaluation JSONs**, does not
fabricate missing metrics, and records paths. To avoid output collisions,
copy/rename each evaluation JSON immediately after each run. Do not
change the active training process to load new code without restarting.


## Startup environment summary

Before epoch 1 (or the first resumed epoch), `train_torch.py` prints exactly
three labeled startup lines: `[DATASET]` (resolved category directory and
training image count), `[GPU]` (CUDA GPU name, physical memory and PyTorch
CUDA build; or MPS/CPU), and `[TRAIN]` (variant, epochs, starting epoch,
batch size, input resolution, latent dimension, learning rate, augmentation,
workers, checkpoint frequency and output checkpoint). GPU memory is total
physical device memory, **not** current available VRAM. Startup lines do
not change model math or checkpoint format.


## Periodic reconstruction monitoring

Set `--monitor-every-k K` (default 0, disabled) to save a PNG grid every
K epochs. At startup the trainer randomly chooses **one normal image from
`train/good` and one anomalous image from `test/<defect>`**, using a
separate seeded random generator. The selected pair stays **fixed for the
entire run** so images are comparable across epochs; choices are recorded
in `examples.json`. No training augmentation is applied to the monitor
inputs. Each PNG has two rows (normal training and anomalous test) and
four columns (original, deterministic posterior-mean reconstruction,
absolute-error grayscale map, ground-truth mask; blank mask for normal).
The error-map grayscale uses a fixed [0,1] scale, not per-image
normalization. The anomalous test image is **qualitative monitoring only**:
do not use it to optimize training or select a checkpoint for final test
performance; use a held-out validation set for model selection.

```bash
python scripts/train_torch.py --config configs/screw_epito.yaml \\
  --variant paper_faithful --epochs 30000 --device cuda \\
  --num-workers 4 --save-every 1 --monitor-every-k 100
```

Outputs: `<output_dir>/pytorch/monitor/paper_faithful/epoch_000100.png`,
`epoch_000200.png`, ... and `examples.json`. This option works on
`--resume` but cannot be injected into a process that is already running;
restart with `--resume` and the same `--epochs` if necessary. The image
rows are labeled; columns are input, reconstruction, error, mask.

## GitHub Actions unit checks

The `Python unit checks` workflow intentionally installs a small set of
lightweight dependencies rather than GPU libraries. Pure Python helpers
must import without TensorFlow or Matplotlib. Heavy model dependencies are
imported lazily at the point of training, checkpoint restoration,
explanation rendering, or visualization. PyTorch-specific tests use
`pytest.importorskip("torch")`, and TensorFlow integration tests should
be executed in a separate TensorFlow-equipped CI environment when added.
This ensures the lightweight runner genuinely executes all dependency-free
tests without masking failed assertions.


### October 2026 CI regression

The outdated `tests/test_explanation_scores.py` imported the removed
`anomaly_probability` helper. It now tests the actual XAI 2024
`reconstruction_mse` explanation target (sum of squared RGB errors,
deterministic reconstruction), using a NumPy-only mock so CI can collect
and execute tests without TensorFlow. This is a unit test of the scoring
contract, not a replacement for TensorFlow model integration tests.
