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

## Override epochs from the command line

To start a **fresh 100-epoch run** without editing YAML:

```bash
python scripts/train.py --config configs/screw.yaml --epochs 100
```

This sets the effective `epochs` configuration to 100 and saves checkpoints
under `outputs/screw/trained_models/screw_VAE_GAN_100_*`, with training
history `screw_VAE_GAN_100.csv`. It does not overwrite the existing
`screw_VAE_GAN_10_*` files.

To compare the new 100-epoch model to the archived model:

```bash
python scripts/compare_models.py --config configs/screw.yaml \
  --new-checkpoint outputs/screw/trained_models/screw_VAE_GAN_100_model
```

To **continue the 10-epoch run** for 90 additional epochs rather than
starting a new run, use:

```bash
python scripts/train.py --config configs/screw.yaml --resume --resume-epochs 90
```

That resumes the prefix named in the YAML (`screw_VAE_GAN_10`) and appends
history to that run. Do **not** combine `--epochs` with `--resume`, because
`--epochs` would change the checkpoint prefix used to find the saved weights.
Use a distinct experiment configuration if you need multiple resumed runs.
