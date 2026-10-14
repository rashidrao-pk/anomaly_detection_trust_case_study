# LIME and SHAP explanation guide

Image selection comes from `dataset_root`, `category`, and `explanation.anomaly_type` plus `explanation.image_index` in YAML.

## Archived model (default YAML checkpoint)

```bash
python scripts/explain.py --config configs/screw.yaml --method lime
python scripts/explain.py --config configs/screw.yaml --method shap
```

## Specify test sample without filesystem paths

```bash
python scripts/explain.py --config configs/screw.yaml --anomaly-type scratch_head --index 5 --method lime
```

`--index` is zero-based across sorted images.

## Explain the newly trained model

The same `--checkpoint` flag permits the new checkpoint, for example with default 10 epochs:

```bash
python scripts/explain.py --config configs/screw.yaml --method lime --checkpoint outputs/screw/trained_models/screw_VAE_GAN_10_model
python scripts/explain.py --config configs/screw.yaml --method shap --checkpoint outputs/screw/trained_models/screw_VAE_GAN_10_model
```

**Output caution:** By default both archived and new checkpoint runs save to the same filename under `outputs/screw/explanations`. Use a distinct `--output` name to avoid overwriting comparisons.

```bash
python scripts/explain.py --config configs/screw.yaml --method lime --output outputs/screw/explanations/new_10_lime.png --checkpoint outputs/screw/trained_models/screw_VAE_GAN_10_model
```

An image and `.npy` map are saved. LIME now follows the notebook's raw reconstruction sum-squared-error target, reconstruction masking baseline, Quickshift segmentation of the 50/50 input/reconstruction blend and signed beta heatmap. Complete explanation/Jaccard parity and SHAP equivalence still require empirical checks.
