# Issue #2 — EfficientAD baseline (work in progress)

[Issue #2](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/issues/2)
adds an **optional PyTorch/Anomalib EfficientAd** baseline to MVTec AD Screw
and Hazelnut. This is independent of the archived TensorFlow VAE-GAN.
The original XAI 2024 notebook and its results must not be modified.

## Install (separate Conda environment)

Do **not** install Anomalib into the existing TensorFlow `xai-ad` environment.

```bash
conda create -n xai-efficientad python=3.11 -y
conda activate xai-efficientad
python -m pip install -U pip
# Install PyTorch for your device using https://pytorch.org/get-started/locally/
python -m pip install "anomalib[full]"
python -m pip install -e .
python -c "from anomalib.models import EfficientAd; print('EfficientAD available')"
```

This Anomalib integration follows its [documented MVTecAD/Engine API](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/efficient_ad.html).
Check the Anomalib and PyTorch dependency versions and license before recording
a benchmark. EfficientAD may download external pretrained weights and may
require ImageNet/Imagenette assets. Read Anomalib's EfficientAD documentation
before running it offline.

## Run baseline training and Anomalib-native evaluation

The new command reads the MVTec root from the existing YAML. It validates
`<dataset_root>/<category>/train/good`, `test` and `ground_truth`.

```bash
python scripts/train_efficientad.py --config configs/screw.yaml --epochs 20
python scripts/train_efficientad.py --config configs/hazelnut.yaml --epochs 20
```

On a CPU-only development machine:

```bash
python scripts/train_efficientad.py --config configs/screw.yaml --epochs 1 --accelerator cpu
```

On Apple Silicon, try `--accelerator mps` only after verifying the installed
Anomalib/Lightning operations support it. NVIDIA CUDA typically uses `gpu`.

The script writes Anomalib's checkpoint/logs and
`outputs/<category>/baselines/efficientad/<category>/test_summary.json`.
It **does not currently export the required per-image scores/maps** for
comparison with the VAE-GAN output schema, or calculate inference times.
Those remain acceptance tasks for issue #2.

## Acceptance checklist

- [ ] Test the baseline end-to-end on Screw and Hazelnut.
- [ ] Export per-image scores, pixel anomaly maps and ground-truth-aligned outputs.
- [ ] Calculate image AUROC, pixel AUROC and inference latency in a common protocol.
- [ ] Produce representative anomaly-map figures and CSV/JSON results.
- [ ] Verify pretrained asset licensing and exact dependency versions.
- [ ] Preserve VAE-GAN functionality and validate separate environments.

**Status:** integration scaffold only; not tested against local MVTec AD data.
Do not close issue #2 until the checklist is met.
