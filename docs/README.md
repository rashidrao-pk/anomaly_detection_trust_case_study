# Training, evaluation, and comparison guides

Run commands from the repository root with `conda activate xai-ad`. The paper's original notebooks and model artifacts remain on the `xai-2024-paper` branch.

## Guides

- [Training and resuming](TRAINING.md) — paper-compatible augmentation, checkpoints, resuming
- [Evaluation and model comparison](EVALUATION.md) — evaluate both checkpoints and interpret image/pixel AUROC
- [LIME and SHAP](EXPLANATIONS.md) — explain images selected by YAML rather than hard-coded paths
- [Migration and compatibility](PYTHON_SCRIPTS.md) — installation and limits of the refactor

**Documentation rule for this repository:** whenever a CLI option, YAML setting, output format, or experiment variant changes, update the relevant `docs/` guide in the same pull request. Record observed results separately from hypotheses and note the commit/config/environment used. Do not claim numerical paper reproduction until parity checks pass.

## Quick workflow

```bash
python scripts/train.py --config configs/screw.yaml
python scripts/compare_models.py --config configs/screw.yaml
python scripts/explain.py --config configs/screw.yaml --method lime
python scripts/explain.py --config configs/screw.yaml --method shap
```

- [Paper figure regeneration](PAPER_FIGURES.md) — archive inventory and runnable commands
