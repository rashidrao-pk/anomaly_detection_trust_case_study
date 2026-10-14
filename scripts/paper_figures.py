#!/usr/bin/env python3
"""Regenerate paper-like plots without inventing unavailable experiments."""
import argparse,csv,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
from anomaly_detection.config import load_config,resolve_path
def rows(path):
    with open(path,newline="") as f:return list(csv.DictReader(f))
def save(fig,out,name):
    out.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(out/(name+"."+ext),dpi=200,bbox_inches="tight")
    plt.close(fig)
    print("[figure]",out/name)
def curves(path,out,category):
    data=rows(path)
    if not data:raise ValueError("Empty history CSV: "+str(path))
    for col in ("vae_loss","disc_loss","gen_los"):
        if col not in data[0]:continue
        fig,ax=plt.subplots(figsize=(10,4))
        ax.plot(range(1,len(data)+1),[float(r[col]) for r in data])
        ax.set(xlabel="Epoch",ylabel=col,title=category+" "+col)
        save(fig,out,"traincurve_vae_VAE_GAN_"+col)
def ground_truth(cfg,out,index):
    from anomaly_detection.data import evaluation_samples,load_rgb,load_mask
    samples=list(evaluation_samples(resolve_path(cfg,cfg["dataset_root"]),cfg["category"]))
    groups={}
    for row in samples:groups.setdefault(row[0].parent.name,[]).append(row)
    names=sorted(groups,key=lambda k:(k!="good",k))
    fig,axes=plt.subplots(3,len(names),figsize=(2.5*len(names),6),squeeze=False)
    for j,name in enumerate(names):
        path,_,maskpath=groups[name][min(index,len(groups[name])-1)]
        img=load_rgb(path,int(cfg.get("image_size",128)))
        mask=np.zeros(img.shape[:2],bool) if maskpath is None else load_mask(maskpath,int(cfg.get("image_size",128)))
        overlay=img.copy()
        from skimage.segmentation import find_boundaries
        overlay[find_boundaries(mask,mode="outer")]=[1.,1.,0.]
        for i,data in enumerate((img,mask,overlay)):
            axes[i,j].imshow(data,cmap="gray" if i==1 else None)
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
        axes[0,j].set_title(name)
    save(fig,out,"GT_"+cfg["category"])
def samples(cfg,out,checkpoint):
    from anomaly_detection.data import training_paths,load_rgb
    from anomaly_detection.inference import load_model,reconstruct
    paths=training_paths(resolve_path(cfg,cfg["dataset_root"]),cfg["category"])[:6]
    images=np.array([load_rgb(p,int(cfg.get("image_size",128))) for p in paths])
    rec=reconstruct(load_model(cfg,checkpoint),images)
    diff=np.sum((images-rec)**2,axis=-1)
    mx=max(float(diff.max()),1e-8)
    fig,axes=plt.subplots(3,len(paths),figsize=(2*len(paths),6),squeeze=False)
    for j in range(len(paths)):
        for i,im in enumerate((images[j],rec[j],diff[j])):
            axes[i,j].imshow(im,cmap="bwr" if i==2 else None,vmin=-mx if i==2 else None,vmax=mx if i==2 else None)
            axes[i,j].axis("off")
    save(fig,out,"samples_m_"+cfg["category"])
def jaccard_figure(path,out,category,method,blend,threshold):
    if threshold is None:raise ValueError("Supply --threshold from the notebook's delta_opt")
    col="IoU_lm_max" if method=="lime" else "IoU_sp_max"
    data=[r for r in rows(path) if r.get("mask_type")==blend and r.get(col) not in ("",None)]
    if not data:raise ValueError("No matching "+blend+" records in "+str(path))
    fig,ax=plt.subplots(figsize=(4,3))
    for truth,pred,color,label in ((False,False,"blue","normal/correct"),(False,True,"orange","normal/incorrect"),(True,True,"green","anomaly/correct"),(True,False,"red","anomaly/incorrect")):
        sub=[r for r in data if (r["a_type"]!="good")==truth and (float(r["anomaly_score"])>threshold)==pred]
        if sub:ax.scatter([float(r["anomaly_score"]) for r in sub],[float(r[col])-(0.15 if not truth else 0.) for r in sub],c=color,s=6,alpha=.6,label=label+" "+str(len(sub)))
    ax.axvline(threshold,ls=":",color="black",linewidth=.8)
    ax.set(xlabel="Anomaly score",ylabel="Maximum Jaccard",ylim=(-.25,1.05))
    ax.legend(fontsize=6)
    variant={"lime_blend_2":"lime_S1","lime_blend_3":"lime_S2","shap_blend_2":"shap_S3"}[method+"_"+blend]
    save(fig,out,"plt_"+category+"_"+variant)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--kind",choices=["curves","gt","samples","s1","s2","s3","all"],default="all")
    p.add_argument("--history")
    p.add_argument("--results-csv")
    p.add_argument("--checkpoint")
    p.add_argument("--threshold",type=float)
    p.add_argument("--index",type=int,default=1)
    p.add_argument("--output-dir")
    a=p.parse_args()
    cfg=load_config(a.config)
    from anomaly_detection.train import checkpoint_prefix
    prefix=checkpoint_prefix(cfg)
    out=Path(a.output_dir) if a.output_dir else resolve_path(cfg,cfg.get("output_dir","outputs"))/"paper_figures"
    tasks=["curves","gt","samples","s1","s2","s3"] if a.kind=="all" else [a.kind]
    missing=[]
    for task in tasks:
        try:
            if task=="curves":curves(Path(a.history or str(prefix)+".csv"),out,cfg["category"])
            elif task=="gt":ground_truth(cfg,out,a.index)
            elif task=="samples":samples(cfg,out,a.checkpoint or str(prefix)+"_model")
            else:
                if not a.results_csv:raise ValueError("--results-csv required")
                method,blend={"s1":("lime","blend_2"),"s2":("lime","blend_3"),"s3":("shap","blend_2")}[task]
                jaccard_figure(a.results_csv,out,cfg["category"],method,blend,a.threshold)
        except (OSError,ValueError,KeyError,IndexError) as exc:
            print("[unavailable] "+task+": "+str(exc),file=sys.stderr)
            missing.append(task)
    if missing:sys.exit("Missing ingredients for: "+", ".join(missing))
if __name__=="__main__":main()
