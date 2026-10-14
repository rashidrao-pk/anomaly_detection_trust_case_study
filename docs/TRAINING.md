# Training variants and checkpoint resume

## Requirements

Activate the `xai-ad` Conda environment. Set `dataset_root` in `configs/screw.yaml` or `configs/hazelnut.yaml` to the MVTec AD root. All commands run from the repository root.

## Paper-compatible training

```bash
python scripts/train.py --config configs/screw.yaml
python scripts/train.py --config configs/hazelnut.yaml
```

The `training_profile: paper` setting applies the notebook's custom ImageDataGenerator augmentation, rescale 1/255, image size 128, batch size 32, latent dimension 32, and Adam LR 0.001. `training_profile: modern` chooses a simpler tf.data pipeline **not** intended to reproduce paper augmentation.

The YAML `epochs: 10` is a smoke-test training budget, not the archived original run's 30,000 configured epochs.

## Save format

The trainer saves five TensorFlow checkpoint components with suffixes `_model`, `_vae`, `_encoder`, `_decoder`, `_discriminator`, plus `.csv` history under `outputs/<category>/trained_models/<category>_VAE_GAN_<epochs>`.

## Resume training

```bash
python scripts/train.py --config configs/screw.yaml --resume --resume-epochs 50
```

The CLI reads the existing CSV to determine the initial epoch and restores the saved model components. `--resume-epochs` means *additional* epochs. It overwrites checkpoints in that run's prefix; copy the run beforehand if intermediate results must be retained. Full optimizer state and random state restoration have not been established: this is a weight-based resume, not guaranteed bitwise-continuous training.

Starting training without `--resume` with the same config can overwrite the prior 10-epoch run. Use a separate YAML/output directory for independent experiments.
