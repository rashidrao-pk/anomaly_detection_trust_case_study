# Per-image XAI 2024 eight-panel figures

The original Screw notebook, cell 42, saves files such as
`results/screw/imgs/good_0_5000_1_30000_100_blend_2.png`.
These contain eight panels: original input, reconstruction, anomaly map,
signed LIME, signed SHAP, LIME/GT Jaccard overlap, SHAP/GT Jaccard overlap,
and GT-contour visual analysis. The archived notebook also overlays ROC
curves on its last panel; the new script initially displays the GT contour
and an AUROC label, so the output is **not pixel-identical** to the archive.

## Regenerate the same sample using archived weights

From the repository root:

```bash
conda activate xai-ad
python scripts/paper_image.py --config configs/screw.yaml \
  --anomaly-type good --index 0 --samples 5000 --shap-evals 5000 \
  --segments 100 --epochs-label 30000 \
  --output outputs/screw/paper_figures/imgs/good_0_5000_1_30000_100_blend_2.png
```

The `--checkpoint` defaults to the archived checkpoint in the YAML.
If `--output` is omitted, the script constructs the historical filename
under `outputs/screw/paper_figures/imgs/`. This keeps the frozen
`results/screw/imgs/` archive untouched.

## Another anomaly type

```bash
python scripts/paper_image.py --config configs/screw.yaml \
  --anomaly-type scratch_head --index 0
```

## Newly trained weights

```bash
python scripts/paper_image.py --config configs/screw.yaml \
  --checkpoint outputs/screw/trained_models/screw_VAE_GAN_10_model \
  --epochs-label 10 --anomaly-type good --index 0
```

The epoch label is used **only for the output filename**. Check the
checkpoint prefix to ensure that the intended weights are loaded.

LIME follows cell 42's 5,000 perturbations, Quickshift of the half-input,
half-reconstruction blend, reconstruction masking baseline, raw MSE
predictor and signed beta values. SHAP uses a reconstruction-image masker,
5,000 evaluations by default, and channel-zero attribution as in the notebook.
Jaccard thresholds are automatically optimized against MVTec ground truth.

**Limitations:** Original notebook interpolation, stochastic controls,
SHAP/LIME versions and the exact visual styling can change results. The
notebook uses test ground truth when optimizing Jaccard; these are
descriptive oracle IoU values, not deployable thresholds. The script
currently uses a modernized last panel and should not be described as
an exact reproduction. Runtime can be substantial for SHAP on CPU or GPU.
