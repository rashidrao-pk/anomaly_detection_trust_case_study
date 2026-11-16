# Download the pretrained checkpoints

The repository keeps the model directory structure as a placeholder, but the
TensorFlow checkpoint files are not stored in Git. Download them from the
[Hugging Face model repository](https://huggingface.co/rashidrao/AD_MVTec_XAI_TensorFlow)
before running evaluation, explanation, or the paper-figure scripts.

From the repository root, install the Hugging Face Hub command-line client if
needed and download the model files into the expected paths:

```bash
python -m pip install -U huggingface_hub
hf download rashidrao/AD_MVTec_XAI_TensorFlow \
  --include "models/**" \
  --local-dir .
```

The command recreates these directories:

```text
models/
├── hazelnut_VAE_GAN_30000/
└── screw_VAE_GAN_30000/
```

The downloaded files include the TensorFlow `.index` and `.data-*` files and
the training-history CSV expected by the existing configs. You do not need to
run `hf auth login` when the Hugging Face repository is public. If it becomes
private or gated, authenticate first:

```bash
hf auth login
```

Verify that the expected component weights are present before running a
script:

```bash
test -f models/hazelnut_VAE_GAN_30000/hazelnut_VAE_GAN_30000_encoder.index
test -f models/screw_VAE_GAN_30000/screw_VAE_GAN_30000_encoder.index
```

For example, the existing configuration can then be used directly:

```bash
python scripts/evaluate.py --config configs/hazelnut.yaml
```

All downloaded checkpoint files are ignored by Git. The `.gitkeep` files only
document and preserve the required directory layout.
