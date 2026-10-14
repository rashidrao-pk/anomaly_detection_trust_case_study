# Notebook-free workflow (in development)

The original notebooks, model weights and generated experimental artifacts are
preserved unchanged on the [xai-2024-paper](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/tree/xai-2024-paper) branch.

## Environment

Use Python 3.10 on a platform supported by TensorFlow 2.15.
Install the original dependency set for the root-level model compatibility module:

```bash
python -m venv .venv
source .venv/bin/activate
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
python scripts/explain.py --config configs/screw.yaml --image data/mvtec_ad/screw/test/scratch_head/000.png --method lime
python scripts/reproduce_paper.py
```

Adjust `--image` to an actual image path. Newly trained checkpoints are written
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
