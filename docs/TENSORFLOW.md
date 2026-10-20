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
