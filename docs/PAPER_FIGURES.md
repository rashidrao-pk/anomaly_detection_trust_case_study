# TensorFlow paper-figure scripts

All four scripts take a YAML `--config` (such as `configs/screw.yaml`)
and save **PNG and PDF** in `<output_dir>/paper_figures/` by default.

## 1. Input / GT inspection
```bash
python scripts/check_inputs.py --config configs/screw.yaml --index 0
```
Produces `GT_screw.png`: input, binary GT, and yellow GT contour for
one image from each test class (including good).

## 2. Representative input/reconstruction/anomaly maps
```bash
python scripts/plot_samples.py --config configs/screw.yaml --index 0
```
Produces `samples_m_screw.png` with original input, pretrained TensorFlow
posterior-mean reconstruction and a red reconstruction-error map, for one
image per class.

## 3. Training curves
```bash
python scripts/plot_training_curves.py --config configs/screw.yaml
python scripts/plot_training_curves.py --config configs/screw.yaml --smoothing 100
```
Uses the historical `models/screw_VAE_GAN_30000/screw_VAE_GAN_30000.csv`
unless `--history` is supplied. Automatically draws every numeric metric;
no invented loss values.

## 4. Maximum Jaccard versus anomaly score
Produce source CSV files from the actual test data:
```bash
python scripts/evaluate.py --config configs/screw.yaml --output outputs/screw/evaluation
python scripts/explain.py --config configs/screw.yaml --all --include-good --method all --evaluate-explanations
```

Plot LIME and SHAP scatter diagrams:
```bash
python scripts/plot_jaccard_vs_score.py --config configs/screw.yaml \
 --scores-csv outputs/screw/evaluation/image_scores.csv \
 --manifest outputs/screw/batch/screw/manifest_all_eval.csv \
 --method both --threshold-mode paper-oracle
```

Pass multiple `--manifest` options to make plots for different settings.
Or use `--threshold VALUE` (ideally calibrated on a separate validation set)
instead of `--threshold-mode paper-oracle`.

Correctness groups: normal/correct blue; normal/incorrect orange;
anomaly/correct green; anomaly/incorrect red. Normal samples use a
**display-only** y=-0.15 because localization IoU is undefined for
normal images. The oracle threshold maximizes sqrt(TPR*(1-FPR)) on the
score labels; do not treat it as unbiased deployment performance.
Unmatched image scores and missing IoUs are never invented.
