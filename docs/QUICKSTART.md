# Lightweight reproducibility tutorial

This tutorial runs the TensorFlow pipeline on one MVTec AD Hazelnut image. It
is intended as a quick smoke test of model loading, anomaly scoring, and
visualization; it is not a full reproduction of the paper experiments.

## 1. Create an environment

Use Python 3.9–3.11 and install the repository dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

The pipeline works on CPU. A GPU is optional and can reduce model restore and
explanation time when TensorFlow has been installed with a compatible GPU
runtime. A residual-map run on one image is lightweight and is normally
completed in seconds to a few minutes, depending mainly on TensorFlow startup
and hardware. LIME and SHAP are more expensive because they evaluate many
perturbed versions of the image; the low-budget example below is for a quick
check, not paper-quality results.

## 2. Obtain the dataset and checkpoint

Download the MVTec AD dataset from the
[official MVTec AD page](https://www.mvtec.com/company/research/datasets/mvtec-ad)
and extract it without committing it to this repository. Set the extracted
dataset directory as `DATASET_ROOT`:

```bash
export DATASET_ROOT=/path/to/mvtec-ad
```

The Hazelnut files should include paths like:

```text
$DATASET_ROOT/
└── hazelnut/
    ├── test/
    │   ├── crack/
    │   └── good/
    └── ground_truth/
        └── crack/
```

Download the pretrained TensorFlow files using the instructions in
[MODELS.md](MODELS.md):

```bash
python -m pip install -U huggingface_hub
hf download rashidrao/AD_MVTec_XAI_TensorFlow \
  --include "models/**" \
  --local-dir .
```

## 3. Select one image

Select the first Hazelnut crack image. The command checks that a sample was
found before continuing:

```bash
IMAGE="$(find "$DATASET_ROOT/hazelnut/test/crack" \
  -type f \( -name '*.png' -o -name '*.jpg' \) | sort | head -n 1)"
test -n "$IMAGE" && echo "Using $IMAGE"
```

## 4. Run anomaly scoring and create a visualization

The residual method loads the pretrained encoder and decoder, reconstructs the
image, computes the RGB mean absolute error map, and writes a panel containing
the input, ground truth, reconstruction, and error map:

```bash
python scripts/explain.py \
  --dataset-root "$DATASET_ROOT" \
  --category hazelnut \
  --checkpoint-dir models/hazelnut_VAE_GAN_30000 \
  --checkpoint-epochs 30000 \
  --image "$IMAGE" \
  --method residual \
  --output outputs/tutorial
```

Expected output:

```text
outputs/tutorial/crack_<image-name>_residual.png
```

The red error map is the model's pixel-level anomaly evidence. The maximum
error is the image-level anomaly score used by the evaluation script.

## 5. Optionally run low-budget LIME and SHAP

After the residual run succeeds, generate the same panel with quick,
low-budget LIME and SHAP explanations:

```bash
python scripts/explain.py \
  --dataset-root "$DATASET_ROOT" \
  --category hazelnut \
  --checkpoint-dir models/hazelnut_VAE_GAN_30000 \
  --checkpoint-epochs 30000 \
  --image "$IMAGE" \
  --method all \
  --num-samples 100 \
  --shap-samples 100 \
  --lime-batch-size 16 \
  --output outputs/tutorial
```

Expected output:

```text
outputs/tutorial/crack_<image-name>_all.png
```

Increase `--num-samples` and `--shap-samples` for more stable explanations.
The paper-style runs use substantially larger evaluation budgets and should be
run through the full [TensorFlow workflow](TENSORFLOW.md).

## Troubleshooting

- `Missing TensorFlow checkpoint`: rerun the Hugging Face download and verify
  that `models/hazelnut_VAE_GAN_30000/` contains both encoder and decoder
  `.index` files.
- `No images in .../test`: check that `DATASET_ROOT` points to the extracted
  MVTec AD directory, not directly to its `hazelnut` subdirectory.
- Missing ground truth: use an anomalous image such as one under `test/crack`
  for this tutorial. Normal images under `test/good` do not have masks.
