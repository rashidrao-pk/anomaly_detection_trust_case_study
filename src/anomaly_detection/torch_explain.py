"""Native PyTorch inference adapter for the original LIME/SHAP protocol."""
from pathlib import Path
import numpy as np
import torch
from .config import resolve_path
from .data import load_rgb
from .torch_train import load_checkpoint, device_for, checkpoint_path, output_root
from .explain import quickshift_segments
from .visualization import save_explanation


class TorchReconstruction:
    def __init__(self,cfg,path=None,device_name="auto"):
        self.device=device_for(device_name)
        self.model,_=load_checkpoint(cfg,path or checkpoint_path(cfg),self.device)
        self.model.eval()

    @torch.inference_mode()
    def reconstruct(self,batch):
        arr=np.asarray(batch,dtype=np.float32)
        inputs=torch.from_numpy(arr).permute(0,3,1,2).to(self.device)
        rec=self.model.reconstruct(inputs)
        return rec.permute(0,2,3,1).cpu().numpy()

    def mse(self,batch):
        batch=np.asarray(batch,dtype=np.float32)
        return np.sum((self.reconstruct(batch)-batch)**2,axis=(1,2,3))


def explain_torch_image(cfg, checkpoint=None, image_path=None, method="lime", num_samples=100,
                        output=None, device_name="auto"):
    adapter=TorchReconstruction(cfg,checkpoint,device_name)
    image=load_rgb(image_path,int(cfg.get("image_size",128)))
    reconstruction=adapter.reconstruct(image[None])[0]
    settings=cfg.get("explanation",{})
    if method=="lime":
        from lime.lime_image import LimeImageExplainer
        blend=(image+reconstruction)*.5
        segments=quickshift_segments(blend,int(settings.get("target_segments",100)),
                                     int(settings.get("random_state",1234)))
        exp=LimeImageExplainer(random_state=int(settings.get("random_state",1234)))
        result=exp.explain_instance(
            image,lambda xs:adapter.mse(xs)[:,None],labels=(0,),
            segmentation_fn=lambda _:segments,hide_color=reconstruction,
            batch_size=int(settings.get("batch_size",200)),num_samples=num_samples)
        heat=np.zeros(segments.shape,dtype=np.float32)
        for segment,weight in result.local_exp[0]:
            heat[segments==segment]=weight
    elif method=="shap":
        import shap
        masker=shap.maskers.Image(reconstruction,reconstruction.shape)
        exp=shap.Explainer(lambda xs:adapter.mse(xs)[:,None],masker,
                           output_names=["Anomaly"])
        result=exp(image[None],max_evals=max(num_samples,2*image.size+1))
        values=np.asarray(result.values)[0]
        if values.ndim==4:values=values[...,0]
        heat=np.mean(values,axis=-1) if values.ndim==3 else values
    else:
        raise ValueError(f"Unsupported explanation method {method}")
    if output is not None:
        output=Path(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        maps=np.mean(np.abs(image-reconstruction),axis=-1)
        save_explanation(image,reconstruction,maps,heat,output,signed=True)
        np.save(output.with_suffix(".npy"),heat)
    return heat
