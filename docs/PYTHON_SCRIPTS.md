# Notebook-free workflow (in development)

The original notebooks, model weights and generated experimental artifacts are
preserved unchanged on the [xai-2024-paper](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/tree/xai-2024-paper) branch.

## Environment

Use Python 3.10 on a platform supported by TensorFlow 2.15.
Install the original dependency set for the root-level model compatibility module:

### using `venv`

```bash
python -m venv .venv
source .venv/bin/activate
```

### using `conda`

```bash
conda create -n xai-ad python=3.10 -y
conda activate xai-ad

python -m pip install --upgrade pip setuptools wheel
```

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[test]"
python -m pytest -q
```

Download MVTec AD separately and extract categories under
`data/mvtec_ad/{screw,hazelnut}/{train,test,ground_truth}`.
No images are downloaded automatically or committed.

## Commands

```bash
python scripts/train.py --config configs/screw.yaml
python scripts/evaluate.py --config configs/screw.yaml
python scripts/evaluate.py --config configs/hazelnut.yaml
python scripts/explain.py --config configs/screw.yaml --method lime
python scripts/explain.py --config configs/hazelnut.yaml --anomaly-type crack --index 2 --method lime
python scripts/reproduce_paper.py
```

The explanation command selects a test image using `dataset_root`, `category`, and
`explanation.anomaly_type` / `explanation.image_index` in the YAML config. The index
is zero-based and files are sorted. Use `--anomaly-type` and `--index` to override
the config, or `--image` for an explicitly supplied file. Results are saved under
the configured `output_dir/explanations/` unless `--output` is passed.
Newly trained checkpoints are written
to `outputs/<category>/vae_gan`; supply `--checkpoint outputs/<category>/vae_gan`
to evaluate a new model.

**Important:** The archived checkpoint compatibility, output equivalence,
SHAP runtime, and end-to-end parity still require empirical validation.
This initial script pipeline uses deterministic posterior-mean reconstruction,
pixel-wise absolute reconstruction error, image max error and AUROC. It does **not**
claim numeric replication of the original paper's trained predictor, optimal
Jaccard analysis or explanation settings. Those remain available in the paper branch.
Do not present this refactoring as validated paper reproduction until these
comparisons have been run and documented.

## PR verification checklist

- [ ] Test fresh Python 3.10 environment
- [ ] Restore both archived TensorFlow checkpoints
- [ ] Run one complete category through training
- [ ] Compare original and CLI reconstructions/scoring and note tolerance
- [ ] Confirm masks align and compute original Jaccard protocol
- [ ] Run both LIME and SHAP with example images
- [ ] Update all entry points and remove notebooks from main only after parity

## XAI 2024 LIME parity notes

The LIME CLI now follows the original screw notebook's reconstruction-loss
predictor (sum of squared pixel errors), posterior-mean reconstruction,
Quickshift segmentation of the 50/50 input-reconstruction blend with target 100
segments, reconstruction-image masking baseline, 5000 perturbations, seed 1234
and signed beta heatmap with a symmetric blue-white-red scale.

The original notebook's per-image optimal Jaccard thresholding, study-level
aggregation, and full SHAP parity remain to be validated. Treat AUROC and
explanations as provisional until numeric parity is established.

## Newly trained vs archived checkpoint comparison

The notebook saved five TensorFlow checkpoints and one CSV per run. The CLI now
uses the same suffixes: `_model`, `_vae`, `_encoder`, `_decoder`,
`_discriminator`, and `.csv`, under
`outputs/<category>/trained_models/<category>_VAE_GAN_<epochs>`.

```bash
python scripts/train.py --config configs/screw.yaml
python scripts/compare_models.py --config configs/screw.yaml
```

The comparison evaluates the **new** checkpoint first and the archived checkpoint
second, on the exact same sorted MVTec test images and masks. It writes
`outputs/screw/comparison/metrics_comparison.json` and `per_image_scores.csv`.
It uses the **current CLI max-MAE protocol**, not the original notebook's
reported paper metrics, and should not be interpreted as paper-result parity.

The default training config runs 10 epochs. This is not comparable to the
archived 30,000-epoch model as a controlled architecture comparison. Matching
training duration, preprocessing, augmentation and initialization are needed
for a fair replication experiment. Training has NOT been run in this PR.

## Notebook-faithful training profile and resume

The `training_profile: paper` mode uses the original notebook's
`Data2.load_data(..., augmentation_target='custom')`: Keras
`ImageDataGenerator` on `train/`, 128px RGB, rotation ±40 degrees,
width/height shifts 0.05, brightness 0.8–1.2, shear/zoom 0.05,
nearest fill and rescale 1/255, plus the original random 90°/270°
preprocessing and additional flips for hazelnut. Batch size defaults to 32.
The original notebook uses three Adam optimizers at their default learning
rate 0.001 (legacy.Adam here for Apple Metal support).

```bash
python scripts/train.py --config configs/screw.yaml
python scripts/train.py --config configs/screw.yaml --resume --resume-epochs 50
python scripts/compare_models.py --config configs/screw.yaml
```

Resume appends metrics to the same CSV and overwrites the same five named
checkpoint files; copy the run before resuming if you need each intermediate
checkpoint. `--resume-epochs` is **additional** epochs. Naming retains the
original initial `epochs` field so subsequent resume runs resolve the
same directory. The notebook likewise used CSV-row counts as its resume
epoch index, while it did not explicitly guarantee restoring Adam's momentum
or learning-rate-scheduler state. This is **weight-based**, not bitwise-exact
stateful resumption. Results cannot be expected to match the archived weights
because random history and the full original training run are unavailable.

This default remains a 10-epoch smoke test to avoid launching a 30,000-epoch
job unexpectedly. To reproduce the notebook's specified training duration,
change `epochs` to 30000 in a new run directory (or use staged training
in a separate experiment config).
