1. Generate a SHAP explanation

```bash
python scripts/explain.py \
  --config configs/screw.yaml \
  --method shap
```

2. Generate a LIME explanation

```bash
python scripts/explain.py \
  --config configs/screw.yaml \
  --method lime
```

3. Generate both LIME and SHAP

```bash
python scripts/explain.py --config configs/screw.yaml --method lime
python scripts/explain.py --config configs/screw.yaml --method shap
```

4. Run both for Hazelnut

```bash
python scripts/explain.py --config configs/hazelnut.yaml --method lime
python scripts/explain.py --config configs/hazelnut.yaml --method shap
```
