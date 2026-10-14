# Evaluation and checkpoint comparison

## Evaluate archived model

```bash
python scripts/evaluate.py --config configs/screw.yaml
```

This loads the archive prefix from `checkpoint` in YAML and writes `image_scores.csv` and `metrics.json` under `output_dir`.

## Compare newly trained model and archive

Train the new model first:

```bash
python scripts/train.py --config configs/screw.yaml
python scripts/compare_models.py --config configs/screw.yaml
```

The comparison loads the newly trained checkpoint first, the archived checkpoint second, and scores the *same* 160 Screw test images. Results:
`outputs/screw/comparison/metrics_comparison.json` and `per_image_scores.csv`.

Both use the refactor's deterministic posterior-mean reconstruction and maximum across the spatial per-channel mean absolute-error map. This does not establish parity with original published paper metrics.

## Observed run: 2026-10-14 (Apple M4 Pro, Screw)

User-run console output (10 epochs, paper-compatible training profile, 160 test images):

| Metric | New 10-epoch checkpoint | Archived checkpoint | New − archived |
|---|---:|---:|---:|
| Image-level AUROC | 0.9772494364 | 0.9247796680 | +0.0524697684 |
| Pixel-level AUROC | 0.8536758774 | 0.9403469932 | −0.0866711158 |

Mean absolute image-score difference: 0.2305485186.

The new checkpoint detects defective *images* more effectively under this metric, but localizes defects less effectively. These are descriptive sample metrics, not evidence of statistical significance or a generally superior algorithm. The training budgets differ radically and need matched-run and multi-seed evaluation.

### Recommended diagnostics

1. Generate paired overlays of input, ground-truth mask, original/new anomaly maps and explanations.
2. Report image-wise metric distributions, defect subcategories, bootstrap confidence intervals, and sensitivity to scoring method.
3. Preserve seeds, epoch counts, dataset splits and exact configs for every run.
4. Evaluate 50/100/500+ epochs before making claims about convergence or parity.
