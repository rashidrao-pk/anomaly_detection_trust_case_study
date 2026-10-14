"""Compare newly trained and archived VAE-GAN on exactly the same images."""
import csv
import json
from pathlib import Path
import numpy as np
from .config import resolve_path
from .data import evaluation_samples,load_rgb,load_mask
from .evaluation import safe_auroc
from .inference import load_model, predict_scores
from .train import checkpoint_prefix

def _evaluate_loaded(cfg, model, samples):
    scores, labels, pixel_scores, masks = [], [], [], []
    for path, label, mask_path in samples:
        img = load_rgb(path,int(cfg.get("image_size",128)))
        image_score, amap, _ = predict_scores(model,img[None])
        scores.append(float(image_score[0]))
        labels.append(label)
        pixel_scores.append(amap[0].reshape(-1))
        mask = np.zeros(amap.shape[1:],dtype=bool) if mask_path is None else load_mask(mask_path,int(cfg.get("image_size",128)))
        masks.append(mask.reshape(-1))
    return {
        "image_auroc":safe_auroc(labels,scores),
        "pixel_auroc":safe_auroc(np.concatenate(masks),np.concatenate(pixel_scores)),
        "scores":scores,
    }

def compare_models(cfg, new_checkpoint=None, original_checkpoint=None):
    root=resolve_path(cfg,cfg["dataset_root"])
    samples=list(evaluation_samples(root,cfg["category"]))
    if not samples: raise ValueError("No test images found")
    new_checkpoint=Path(new_checkpoint) if new_checkpoint else Path(str(checkpoint_prefix(cfg))+"_model")
    original_checkpoint=Path(original_checkpoint) if original_checkpoint else resolve_path(cfg,cfg["checkpoint"])
    if new_checkpoint.resolve()==original_checkpoint.resolve():
        raise ValueError("New and archived checkpoints must be different paths")
    result={}
    # Evaluate newly trained weights FIRST. Same sample list and score protocol.
    for name,checkpoint in (("new",new_checkpoint),("original",original_checkpoint)):
        print(f"[compare] Loading {name} weights: {checkpoint}", flush=True)
        result[name]=_evaluate_loaded(cfg,load_model(cfg,checkpoint),samples)
        result[name]["checkpoint"]=str(checkpoint)
    output=resolve_path(cfg,cfg.get("output_dir","outputs"))/"comparison"
    output.mkdir(parents=True,exist_ok=True)
    with (output/"per_image_scores.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["path","label","new_score","original_score","new_minus_original"])
        for sample,a,b in zip(samples,result["new"]["scores"],result["original"]["scores"]):
            writer.writerow([str(sample[0]),sample[1],a,b,a-b])
    summary={
        "category":cfg["category"],"samples":len(samples),
        "scoring":"max over spatial mean RGB absolute reconstruction error; deterministic posterior mean",
        "new_checkpoint":str(new_checkpoint),"original_checkpoint":str(original_checkpoint),
        "new":{k:v for k,v in result["new"].items() if k!="scores"},
        "original":{k:v for k,v in result["original"].items() if k!="scores"},
        "differences":{
            metric: (result["new"][metric]-result["original"][metric]
                     if result["new"][metric] is not None and result["original"][metric] is not None else None)
            for metric in ["image_auroc","pixel_auroc"]
        },
        "mean_absolute_image_score_difference":float(np.mean(np.abs(np.asarray(result["new"]["scores"])-np.asarray(result["original"]["scores"]))))
    }
    (output/"metrics_comparison.json").write_text(json.dumps(summary,indent=2))
    return summary
