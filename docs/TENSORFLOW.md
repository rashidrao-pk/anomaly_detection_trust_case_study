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


### Coefficient-weighted LIME heatmaps

Unlike the earlier binary `get_image_and_mask` display, the LIME panel
maps signed values from `explanation.local_exp[1]` onto each segment in
`explanation.segments`. The map is normalized to the strongest absolute
attribution **within that image** for visualization only; contributions
below 15% of that strength are hidden, leaving the underlying RGB image
visible. Red indicates a positive contribution to the sample-relative error
surrogate, blue a negative contribution. A large blue area does not imply a
normal classification. These display transformations do not change the
underlying LIME fit or quantitative anomaly-detection evaluation.

Re-run LIME without `--skip-existing` to overwrite existing binary overlays.


## Extended XAI output: ground truth, LIME, SHAP

The `explain.py` CLI now renders the class folder and zero-padded image ID in
its figure heading. Every figure includes **Input**, **Input + GT** (cyan
contour where anomaly labels exist), **Reconstruction**, and **Reds error map**.
For a normal image, GT is correctly absent.

Choose `--method lime`, `--method shap`, or `--method all` to append signed
attribution overlays (red positive, blue negative). Magnitude controls opacity
so broad low-magnitude negative superpixels no longer obscure the entire
image. LIME uses SLIC segments; SHAP uses an image blur masker with the
Partition explainer. Both explain the *same bounded, sample-relative
reconstruction-error surrogate*, not a trained classification probability.
SHAP is not guaranteed to outperform LIME; use GT-based validation before
making faithfulness claims. `--shap-samples` is a model-evaluation budget;
SHAP is comparatively expensive. Results are visualization-only until tested.

```bash
# Quick run on Mac
python scripts/explain.py --config configs/screw.yaml --method all \\
  --num-samples 200 --shap-samples 100
# Five test images
python scripts/explain.py --config configs/screw.yaml --all --limit 5 \\
  --method all --num-samples 200 --shap-samples 100
# Full test set; avoid --skip-existing when replacing old figures
python scripts/explain.py --config configs/screw.yaml --all --method all \\
  --num-samples 1000 --shap-samples 200
```

You can select `--method shap` for SHAP-only results or `--method lime`
for LIME-only results. Identical image basenames in different anomaly
categories are kept in separate folders and recorded in the CSV manifest.


### XAI-2024 original heatmap styling

The LIME and SHAP panels use **opaque** `cmap='bwr'` images with symmetric
`vmin=-max(abs(weights))`, `vmax=+max(abs(weights))`, and an individual
vertical colorbar, matching `AD_VAE_GAN_SCREW.ipynb` and
`AD_VAE_GAN_HAZELNUT.ipynb` from the `xai-2024-paper` branch.
The prior transparent, magnitude-weighted red/blue overlays did **not**
match these publication plots; that alternative helper remains available
for optional overlays but is not used by the standard panels.
The current standalone reconstruction-error panel intentionally retains
`Reds`, whereas the archive used `bwr` for some anomaly-map figures.
**Styling parity does not imply algorithmic parity:** the current script
uses a bounded reconstruction-error surrogate and RGB-averaged SHAP;
archived notebook model outputs, channel selection, segmentation, and
sampling may differ. Compare them before claiming numerical reproduction.


## Paper-faithful explanation pipeline (2024 reproduction)

The standard `scripts/explain.py` XAI path now matches the archived paper
notebooks' definitions: **single-output raw reconstruction MSE**, deterministic
encoder-mean reconstruction, **Quickshift** with kernel_size=2, ratio=0.95,
sigma=0.05 and seed 1234, and the archived binary search for approximately
100 segments of the input/reconstruction 50:50 blend. LIME uses
`hide_color=reconstruction`, `random_state=1234` and **5,000 samples**.
SHAP uses `shap.maskers.Image(reconstruction, reconstruction.shape)`, a
single anomaly output, **5,000 evaluations**, and the notebook's channel-0
map (`values[0, :, :, 0, 0]`). Both plots use opaque `bwr`, symmetric
vmin/vmax and individual colorbars. Batch size defaults to **200**.

The earlier bounded two-class error surrogate, SLIC segmentation, black
perturbation background, blur SHAP masker and RGB-channel averaging are no
longer used by the standard XAI path. A shorter smoke test can override:
`--num-samples 100 --shap-samples 100 --target-segments 100`.
The previously added `Reds` residual display and the class/GT panels remain.

```bash
python scripts/explain.py --config configs/screw.yaml --method all
python scripts/explain.py --config configs/screw.yaml --method all \
  --num-samples 100 --shap-samples 100
```

These definitions mirror the archived notebook code; runtime/numerical
reproduction must still be validated with the user's TensorFlow and SHAP
versions and the identical dataset image. The archival SHAP plot selected
RGB channel 0; this is intentional, not spatial averaging.


## Explain only, or explain + evaluate

Without additional flags, `scripts/explain.py` creates explanation panels
only. Add `--evaluate-explanations` to append the original XAI-2024-style
optimal-Jaccard (IoU) ground-truth agreement panels for each requested
LIME/SHAP map and include the scores and thresholds in the batch CSV.

```bash
# Explain only
python scripts/explain.py --config configs/screw.yaml --method all

# Explain + evaluate with original paper-style overlap panels
python scripts/explain.py --config configs/screw.yaml --method all --evaluate-explanations

# Batch explain + evaluate
python scripts/explain.py --config configs/screw.yaml --all --method all --evaluate-explanations
```

For speed, append `--num-samples 100 --shap-samples 100` to test first.
Figures produced with evaluation have `_eval.png` suffix and CSV manifests
have `_eval.csv`, preserving previous explanation-only outputs.

The Jaccard overlap image follows the paper's `combine_groundtruth_explanation`
colors: white = neither, cyan = GT-only, red = explanation-only, black =
overlap. IoU is maximized over unique attribution thresholds, with tied
values grouped together. For normal images with no positive GT mask, the
explanation localization IoU is N/A rather than zero.

**Scientific limitation:** This best-possible, GT-selected threshold is an
oracle diagnostic of localization potential, **not** a usable test-time
decision rule or an unbiased assessment of deployment performance.
