"""Evaluate detection and localization; export per-image scores and summary."""
import csv
import json
from pathlib import Path
import numpy as np
from .config import resolve_path
from .data import evaluation_samples, load_rgb, load_mask
from .inference import load_model, predict_scores

def safe_auroc(labels, scores):
    from sklearn.metrics import roc_auc_score
    if len(set(int(y) for y in labels)) < 2:
        return None
    return float(roc_auc_score(labels,scores))

def evaluate(cfg, checkpoint=None):
    root=resolve_path(cfg,cfg["dataset_root"])
    output=resolve_path(cfg,cfg.get("output_dir","outputs"))
    output.mkdir(parents=True,exist_ok=True)
    if checkpoint is None:
        checkpoint=resolve_path(cfg,cfg["checkpoint"])
    print(f'[+] Loading model from {checkpoint} ...')
    print(f'[+] Evaluating {cfg["category"]} using checkpoint {checkpoint} and saving results to {output}')

    model=load_model(cfg,checkpoint)
    rows=[]
    all_masks=[]
    all_maps=[]
    for path,label,mask_path in evaluation_samples(root,cfg["category"]):
        image=load_rgb(path,int(cfg.get("image_size",128)))
        scores,maps,_=predict_scores(model,image[None])
        mask=np.zeros(maps.shape[1:], dtype=bool) if mask_path is None else load_mask(mask_path,int(cfg.get("image_size",128)))
        rows.append({"path":str(path),"label":label,"score":float(scores[0])})
        all_masks.append(mask.ravel())
        all_maps.append(maps[0].ravel())
    if not rows:
        raise ValueError("No evaluation images were found")
    with (output/"image_scores.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["path","label","score"])
        writer.writeheader()
        writer.writerows(rows)
    summary={
        "category":cfg["category"],
        "samples":len(rows),
        "image_auroc":safe_auroc([r["label"] for r in rows],[r["score"] for r in rows]),
        "pixel_auroc":safe_auroc(np.concatenate(all_masks),np.concatenate(all_maps))
    }
    with (output/"metrics.json").open("w") as f:
        json.dump(summary,f,indent=2)
    return summary
