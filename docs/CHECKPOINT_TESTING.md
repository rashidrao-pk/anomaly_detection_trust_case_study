# Checkpoint reconstruction-quality inspection

Use `scripts/test_checkpoint.py` after training to inspect one representative
test image from the normal `good` category and **every** available anomaly
type, selected by the same sorted zero-based index within each class.

## Newly trained checkpoint (recommended)

```bash
conda activate xai-ad
python scripts/train.py --config configs/screw.yaml --epochs 100
python scripts/test_checkpoint.py --config configs/screw.yaml --epochs 100
```

Here `--epochs 100` means **select the already-trained 100-epoch
checkpoint**, not train again. The model is read from
`outputs/screw/trained_models/screw_VAE_GAN_100_model`.

For the existing 10-epoch checkpoint:

```bash
python scripts/test_checkpoint.py --config configs/screw.yaml --epochs 10
```

For your archived XAI 2024 model:

```bash
python scripts/test_checkpoint.py --config configs/screw.yaml
```

Or provide any checkpoint prefix:

```bash
python scripts/test_checkpoint.py --config configs/screw.yaml \
  --checkpoint outputs/screw/trained_models/screw_VAE_GAN_100_model
```

Choose the second image per category without hard-coding filenames:

```bash
python scripts/test_checkpoint.py --config configs/screw.yaml --epochs 100 --index 1
```

## Outputs

By default, files are stored in
`outputs/screw/checkpoint_tests/<checkpoint-name>/`:

- `reconstruction_grid.png` and `.pdf`: one row per test class, with
  input, reconstruction, per-pixel mean absolute error, ground truth and
  error inside the ground-truth mask
- `reconstruction_metrics.csv` and `.json`: per-example MAE, MSE,
  PSNR, SSIM, maximum pixel mean absolute error, and errors inside/outside GT

You can use `--output-dir <directory>` to change the location.

Images use the current evaluation pipeline's 128x128 RGB bilinear resizing.
The ground-truth masks use nearest-neighbor resizing, and reconstructions
use the encoder posterior mean. Metrics use images scaled to [0,1].
The selection is deterministic; it is **not** a randomized or representative
dataset-level statistical sample.

These are qualitative and reconstruction-fidelity diagnostics: use
`scripts/compare_models.py` for full image/pixel AUROC and do not
interpret low reconstruction error as proof of anomaly localization.
