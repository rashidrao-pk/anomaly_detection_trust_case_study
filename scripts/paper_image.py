#!/usr/bin/env python3
"""Re-create notebook cell 42 eight-panel LIME+SHAP per-image figure."""
import argparse,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
from anomaly_detection.config import load_config,resolve_path
from anomaly_detection.data import evaluation_samples,load_rgb,load_mask
from anomaly_detection.inference import load_model,reconstruct
from anomaly_detection.explain import reconstruction_mse,quickshift_segments

def get_sample(cfg,kind,index):
    rows=[r for r in evaluation_samples(resolve_path(cfg,cfg["dataset_root"]),cfg["category"]) if r[0].parent.name==kind]
    if not rows:raise ValueError(f"No samples of type {kind}")
    if index<0 or index>=len(rows):raise IndexError(f"Index {index} invalid for {kind}; {len(rows)} images")
    return rows[index]

def optimal_iou(ground_truth,scores):
    """Notebook calc_IoU_curve: threshold at each distinct descending score."""
    ground=np.asarray(ground_truth,dtype=bool).ravel()
    vals=np.asarray(scores,dtype=np.float64).ravel()
    ordering=np.argsort(-vals,kind="stable")
    labels=ground[ordering]; sorted_scores=vals[ordering]
    last=np.flatnonzero(np.r_[sorted_scores[1:]!=sorted_scores[:-1],True])
    intersect=np.cumsum(labels)[last].astype(float)
    k=last+1
    union=k+ground.sum()-intersect
    ious=np.divide(intersect,union,out=np.zeros_like(intersect),where=union!=0)
    top=int(np.argmax(ious))
    return float(ious[top]),float(sorted_scores[last[top]])

def rgb_overlap(ground,heat,threshold):
    # Original utils.visualize.combine_groundtruth_explanation
    gt=np.asarray(ground,dtype=bool)
    active=np.asarray(heat)>=threshold
    out=np.zeros((*gt.shape,3),dtype=np.uint8)
    out[...,0]=255*(~gt)
    out[...,1]=255*(~active)
    out[...,2]=255*(~active)
    return out

def generate(cfg,checkpoint,kind,index,samples,max_evals,segments_target,output,seed):
    from lime.lime_image import LimeImageExplainer
    from sklearn.metrics import roc_curve,roc_auc_score
    import shap
    from skimage.segmentation import find_boundaries
    path,label,maskpath=get_sample(cfg,kind,index)
    sz=int(cfg.get("image_size",128))
    image=load_rgb(path,sz)
    gt=np.zeros((sz,sz),dtype=bool) if maskpath is None else load_mask(maskpath,sz)
    model=load_model(cfg,checkpoint)
    rec=reconstruct(model,image[None])[0]
    anomaly_map=np.max(np.abs(rec-image),axis=-1)
    residual=np.linalg.norm(rec-image,axis=-1)
    blend=image*.5+rec*.5
    seg=quickshift_segments(blend,segments_target,seed)
    count=len(np.unique(seg))
    print(f"[gallery] sample={path} segments={count}")
    def predict(batch):
        return reconstruction_mse(model,batch)[:,None]
    lime=LimeImageExplainer(random_state=seed).explain_instance(
        image,predict,labels=(0,),segmentation_fn=lambda _:seg,
        hide_color=rec,batch_size=200,num_samples=samples)
    beta=dict(lime.local_exp[0])
    lmap=np.zeros(seg.shape,np.float32)
    for k,v in beta.items():lmap[seg==k]=v
    liou,lth=optimal_iou(gt,lmap)
    print(f"[gallery] LIME optimum Jaccard={liou:.5f}, threshold={lth:.6g}")
    masker=shap.maskers.Image(rec,rec.shape)
    shap_explainer=shap.Explainer(predict,masker,output_names=["Anomaly"])
    # Follow notebook max_evals=5000 rather than enforcing exhaustive
    # 2*image.size+1 calls (which was a bug in the existing SHAP CLI).
    sh_values=shap_explainer(image[None],max_evals=max_evals,batch_size=200)
    arr=np.asarray(sh_values.values)[0]
    if arr.ndim==4:arr=arr[...,0]
    if arr.ndim!=3:raise ValueError(f"Unexpected SHAP shape: {arr.shape}")
    # Notebook cell 42 used channel 0, not channel-mean SHAP.
    smap=arr[:,:,0]
    siou,sth=optimal_iou(gt,smap)
    print(f"[gallery] SHAP optimum Jaccard={siou:.5f}, threshold={sth:.6g}")
    overlay=image.copy()
    overlay[find_boundaries(gt,mode="outer")]=[1.,1.,0.]
    fig,axs=plt.subplots(1,8,figsize=(14,2))
    panels=[image,rec,anomaly_map,lmap,smap,rgb_overlap(gt,lmap,lth),rgb_overlap(gt,smap,sth),overlay]
    titles=["Input","Reconstruction","Anomaly map","LIME","SHAP","Jaccard LIME","Jaccard SHAP","GT / ROC"]
    for i,(ax,img,title) in enumerate(zip(axs,panels,titles)):
        if i in (2,3,4):
            limit=max(float(np.max(np.abs(img))),1e-10)
            plot=ax.imshow(img,cmap="bwr",vmin=-limit,vmax=limit)
            if i in (3,4):fig.colorbar(plot,ax=ax,fraction=.048,pad=.02,shrink=.8)
        else:ax.imshow(img)
        ax.set_title(title,fontsize=8)
        ax.set_xticks([]);ax.set_yticks([])
    axs[5].text(5,20,f"{liou:.3f}",fontsize=9)
    axs[6].text(5,20,f"{siou:.3f}",fontsize=9)
    if label and gt.any() and len(np.unique(residual))>1:
        try:
            auroc=roc_auc_score(gt.ravel(),residual.ravel())
            axs[7].set_xlabel(f"pixel AUROC: {auroc:.3f}",fontsize=8)
        except ValueError:pass
    fig.tight_layout(pad=.2)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=250,bbox_inches="tight")
    plt.close(fig)
    print("[gallery] saved:",output)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--checkpoint",help="Checkpoint prefix; defaults to archived checkpoint in YAML")
    p.add_argument("--anomaly-type",default="good")
    p.add_argument("--index",type=int,default=0)
    p.add_argument("--samples",type=int,default=5000)
    p.add_argument("--shap-evals",type=int,default=5000)
    p.add_argument("--segments",type=int,default=100)
    p.add_argument("--seed",type=int,default=1234)
    p.add_argument("--epochs-label",type=int,default=30000,help="Filename label, not actual training duration")
    p.add_argument("--output",help="Override destination")
    args=p.parse_args()
    cfg=load_config(args.config)
    if cfg["category"] not in ("screw","hazelnut"):raise ValueError("Unsupported paper category")
    checkpoint=args.checkpoint or resolve_path(cfg,cfg["checkpoint"])
    filename=f"{args.anomaly_type}_{args.index}_{args.samples}_1_{args.epochs_label}_{args.segments}_blend_2.png"
    output=args.output or str(resolve_path(cfg,cfg.get("output_dir","outputs"))/"paper_figures"/"imgs"/filename)
    generate(cfg,checkpoint,args.anomaly_type,args.index,args.samples,args.shap_evals,args.segments,output,args.seed)
if __name__=="__main__":main()
