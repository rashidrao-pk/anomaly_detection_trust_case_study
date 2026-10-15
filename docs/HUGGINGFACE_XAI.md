# Publishing a validated PyTorch VAE-GAN to Hugging Face

## Goal

Keep `paper_aligned` as the closest currently implemented port of the
archived TensorFlow architecture, and do **not** present it as proven exact
numerical equivalence. Prior Screw diagnostics (checkpoint at epoch 7770)
measured image AUROC 0.4901 and pixel AUROC 0.7665. Do not publish this
checkpoint as a high-performing anomaly detector.

## Decision gate

1. Use normal images from **train/good only** to create reproducible
   training/validation splits. Evaluate normal validation reconstruction
   error and monitoring panels. Avoid model selection on MVTec test labels.
2. Run `paper_aligned` with a fresh output directory for each controlled
   augmentation/loss experiment. Preserve all checkpoint histories.
3. Compare the selected checkpoint against archived TensorFlow reference on
   the same preprocessing, images and anomaly-score definition.
4. Document the predeclared image and pixel metrics and limitations in the
   model card. Reserve original MVTec test results as exploratory if used in
   debugging.
5. Only then upload a verified release.

## Export on Epito (checkpoint stays untouched)

```bash
python scripts/export_hf_torch.py \
  --checkpoint outputs/screw/pytorch/trained_models/screw_VAE_GAN_30000_paper_aligned.pt \
  --output exports/vae-gan-screw-pytorch
```

This creates `model.pt` (weights plus metadata), `config.json`, and a
draft `README.md` model card. Verify model card metrics and limitations.
Do not distribute dataset images or data samples without checking rights.

## Upload (only after validation)

```bash
python -m pip install -U huggingface_hub
hf auth login
hf repo create YOUR_USERNAME/vae-gan-screw-pytorch --repo-type model
hf upload YOUR_USERNAME/vae-gan-screw-pytorch exports/vae-gan-screw-pytorch . --repo-type model
```

Authentication and publishing run on the user's Epito environment, not
from CI. If `hf` is unavailable, use
`huggingface_hub.HfApi().upload_folder(...)`.

## XAI research direction

Use the **same validated checkpoint** for all explanation comparisons:
posterior-mean reconstruction, continuous reconstruction MSE as explanation
target (archived paper), and measured anomaly maps for localization. Compare
LIME, SHAP, Integrated Gradients and gradient-based alternatives on the same
examples, using the same seeds, segmentation budgets, runtime accounting, and
ground-truth masks where available. Report faithfulness and stability as
well as localization performance, and avoid assuming prettier heatmaps mean
more faithful explanations.

**Important:** `torch.load(weights_only=True)` still requires trusting the
artifact source and environment. Release verifiable checksums and versioned
training configuration alongside weights.
