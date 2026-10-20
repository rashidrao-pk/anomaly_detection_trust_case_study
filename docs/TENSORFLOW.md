# TensorFlow script-first workflow (XAI-World 2024)

This branch retains the original `models.py` architecture and `utils.Data2`
augmentation. The original notebooks are kept in `notebooks/` for archival
reproducibility; Python scripts are the recommended interface for new runs.

**Status:** These CLI wrappers were developed from the archived Python
sources. Their ability to restore all historical checkpoint formats and
reproduce reported paper metrics must be verified in a TensorFlow 2.15
environment before asserting numerical reproduction.

## Install

Use Python 3.10, TensorFlow 2.15 and the pinned `requirements.txt`.
Do not use the PyTorch-only Python 3.12 environment for this branch.

## Train

```bash
python scripts/train.py --dataset-root /path/to/mvtec-ad \
  --category screw --epochs 30000 --batch-size 32 \
  --latent-dim 32 --augmentation custom --output outputs/tensorflow

python scripts/train.py --dataset-root /path/to/mvtec-ad \
  --category hazelnut --epochs 30000 --batch-size 32 \
  --latent-dim 32 --augmentation custom --output outputs/tensorflow
```

For `custom`, Screw uses no flips while Hazelnut uses horizontal/vertical
flips. Both use the paper's random affine, brightness and custom rotation
logic. `none` disables random transformations. Legacy `full` is deliberately
not enabled until its notebook choice is verified.

Training uses the original custom `VAE_GAN.train_step`, not the PyTorch
optimizer routine. The script saves encoder, decoder, discriminator, VAE and
model weight prefixes as `*.index` plus `*.data-*`, and a CSV history.
Training currently has no resumable optimizer-state workflow; protect
long runs against preemption.

## Evaluate existing archival weights

```bash
python scripts/evaluate.py --dataset-root /path/to/mvtec-ad \
  --category screw --checkpoint-dir models/screw_VAE_GAN_30000 \
  --checkpoint-epochs 30000 --output outputs/tensorflow/evaluation/screw
```

This evaluates image AUROC from the maximum RGB-mean absolute reconstruction
error and pixel AUROC from the corresponding residual maps. Anomalies are
identified by directory name rather than class index. Paper figure metrics
might differ if the notebook used other scoring, latent sampling, or masks.

## Explain one image

```bash
python scripts/explain.py --category screw \
  --checkpoint-dir models/screw_VAE_GAN_30000 \
  --checkpoint-epochs 30000 \
  --image /path/to/mvtec-ad/screw/test/scratch_head/000.png \
  --method residual --output outputs/tensorflow/xai
```

Optional `--method lime` evaluates a reconstruction-MSE target on LIME
superpixel perturbations. The notebook's complete XAI benchmark and SHAP
variants are not yet converted; do not claim methodological parity for
those explanations. LIME results need independent evaluation of faithfulness.

## Branch policy

- `tensorflow`: script-first TensorFlow development
- `pytorch`: PyTorch training/XAI
- `xai-2024-paper`: archival reference, unchanged

Generated `outputs/`, `results/`, and Python caches are excluded from Git.


## YAML configuration (recommended)

All three scripts now accept `--config`. Four existing YAMLs are supported:
`configs/screw.yaml`, `configs/hazelnut.yaml` (local Mac),
`configs/screw_epito.yaml`, `configs/hazelnut_epito.yaml` (Epito).
`dataset_root`, `category`, `output_dir`, `batch_size`,
`latent_dim`, `learning_rate`, `seed` and `epochs` are loaded
for training; CLI options override YAML values. A relative output path
is interpreted relative to repository root.

```bash
python scripts/train.py --config configs/screw_epito.yaml --epochs 30000
python scripts/evaluate.py --config configs/screw_epito.yaml
python scripts/explain.py --config configs/screw_epito.yaml
python scripts/explain.py --config configs/hazelnut.yaml --method residual
```

**Important:** YAML files currently set `epochs: 10` as a test default;
pass `--epochs 30000` for the full paper run. Evaluation/explanation
instead infer `checkpoint_epochs=30000` from the configured `checkpoint:`
path, **not** the training `epochs` setting. `explanation.anomaly_type`
and `explanation.image_index` select the test image by its sorted filename;
`explanation.method` and `num_samples` configure XAI. Existing
`checkpoint` files are used for evaluation via their parent folder.
The `checkpoint` YAML field ends in `_model` while evaluation restores
encoder and decoder component weights from the same folder.

## Explain every test image (batch mode)

The following processes *all* MVTec test images, across good and every anomaly
category, using the dataset path from your Mac config:

```bash
python scripts/explain.py --config configs/screw.yaml --all --method residual
python scripts/explain.py --config configs/screw.yaml --all --method lime \
  --num-samples 100 --skip-existing
```

On Epito use `configs/screw_epito.yaml`; for Hazelnut choose the
corresponding `hazelnut*.yaml`. To test on just five images:

```bash
python scripts/explain.py --config configs/screw.yaml --all --limit 5 --method lime
```

To explain only one defect category (including `good`):

```bash
python scripts/explain.py --config configs/screw.yaml --all \
  --anomaly-type scratch_head --method lime
```

Batch figures are grouped at `<output_dir>/batch/<category>/<defect>/<stem>_<method>.png`
and a `manifest_<method>.csv` includes input path, label, defect, PNG, and
residual score. Each defect folder is separate, so repeated filenames across
categories never overwrite one another. `--skip-existing` avoids recomputing
already generated figures. For an interrupted run, the manifest is regenerated
from the current run and marks skipped files.

LIME can be slow across the whole test split: first use `--limit`.
The plotted residual map is a reconstruction discrepancy, not a calibrated
probability. LIME targets the legacy reconstruction-MSE surrogate; it should
not be represented as an experimentally validated explanation without a
faithfulness/GT localization study. No test labels are used for training.


### LIME visualization and output interpretation

LIME now explains the anomaly-score column explicitly using `labels=(1,)`.
It converts squared RGB reconstruction error to a bounded, monotonic,
sample-relative score `error / (error + original_image_error)` so the LIME
classifier receives two columns summing to one. **This is a visualization
surrogate, not a calibrated anomaly probability.** The LIME panel overlays
signed influential regions on the image: **red** indicates features supporting
higher error and **blue** indicates features opposing it. This avoids the
all-black mask caused by `positive_only=True` when there are no selected
positive features. Run without `--skip-existing` to regenerate old figures.
