#!/usr/bin/env python3
"""Export a trained PyTorch paper-aligned VAE-GAN as a portable HF model folder.

This script does not publish anything and never modifies the training checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from anomaly_detection.torch_train import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--license", default="mit")
    parser.add_argument("--model-name", default="VAE-GAN MVTec Screw (PyTorch)")
    args = parser.parse_args()
    src = Path(args.checkpoint).expanduser().resolve()
    if not src.is_file():
        parser.error(f"Checkpoint not found: {src}")
    dst = Path(args.output).expanduser().resolve()
    if dst == src or src in dst.parents:
        parser.error("Output must be separate from training checkpoint")
    dst.mkdir(parents=True, exist_ok=True)
    model, state = load_checkpoint({}, src, torch.device("cpu"))
    model.eval()
    config = {
        "architecture": "VAEGAN",
        "latent_dim": int(state["latent_dim"]),
        "image_size": int(state.get("image_size", 128)),
        "category": str(state.get("category", "unknown")),
        "torch_variant": str(state.get("torch_variant", "initial")),
        "training_profile": str(state.get("training_profile", "unknown")),
        "checkpoint_epoch": int(state["epoch"]),
        "input_channels": 3,
        "pixel_range": [0.0, 1.0],
        "reconstruction": "encoder posterior mean -> decoder; RGB image 128x128",
        "anomaly_map": "mean absolute RGB reconstruction error",
        "image_score": "max over anomaly map",
    }
    if config["image_size"] != 128:
        raise ValueError("Only 128x128 model architecture is supported")
    torch.save({"model": model.state_dict(), **config}, dst / "model.pt")
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    name = args.model_name
    readme = f"""---
license: {args.license}
tags:
- anomaly-detection
- pytorch
- vae-gan
- industrial-inspection
- explainable-ai
datasets:
- mvtec-ad
---

# {name}

This is a PyTorch implementation of the research VAE-GAN in
[anomaly_detection_trust_case_study](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study).

**Status: evaluation required.** Upload only after reporting measured results.
The original TensorFlow model and PyTorch reimplementation are not numerically identical.

- Category: `{config['category']}`
- Variant: `{config['torch_variant']}`
- Augmentation profile: `{config['training_profile']}`
- Checkpoint epoch: {config['checkpoint_epoch']}
- Input: RGB image resized to 128 x 128, pixel range [0, 1]
- Anomaly score: maximum of mean absolute RGB reconstruction error
- Explanations: reconstruction-error heatmaps; LIME and SHAP implementations
  are maintained in the source repository.

## Inference

Clone the code repository, install PyTorch/torchvision and load this `model.pt`:

```python
import torch
from anomaly_detection.torch_models import VAEGAN

saved = torch.load("model.pt", map_location="cpu", weights_only=True)
model = VAEGAN(saved["latent_dim"], paper_aligned=saved["torch_variant"] == "paper_aligned")
model.load_state_dict(saved["model"])
model.eval()

# x: preprocessed NCHW RGB tensor, values in [0, 1], size 128x128
with torch.inference_mode():
    recon = model.reconstruct(x)
    anomaly_map = (x - recon).abs().mean(dim=1)
    image_score = anomaly_map.flatten(1).amax(dim=1)
```

## Evaluation and limitations

**Fill in the exact evaluation metrics and source checkpoint before publishing.**
Reconstruction-error heatmaps can highlight normal edges rather than genuine
anomalies. Test-set metrics must not be used for checkpoint selection.
Check the repository's documentation for methodological differences from TensorFlow.

## Citation

Please cite the original accompanying paper and link to the source repository.
"""
    (dst / "README.md").write_text(readme)
    print(f"[export] {dst}")
    print("[export] model.pt, config.json, README.md")
    print("[export] No upload performed; verify metrics and model card before publication.")


if __name__ == "__main__":
    main()
