"""Black-box LIME and SHAP explanations of reconstruction anomaly scores.

Explanations target image-level anomaly probability calibrated by a provided
scale factor. This is NOT a learned classification probability.
"""
from pathlib import Path
import numpy as np
from .data import load_rgb
from .inference import predict_scores
from .visualization import save_explanation

def anomaly_probability(scores, scale):
    if scale <= 0:
        raise ValueError("scale must be > 0")
    values = np.maximum(np.asarray(scores, dtype=np.float64), 0.0)
    # Smooth monotonic mapping avoids hard clipping all high anomaly scores to 1.
    return values / (values + scale)

def explain_image(cfg, model, image_path, method="lime", num_samples=100, scale=0.1, output=None):
    image=load_rgb(image_path,int(cfg.get("image_size",128)))
    observed_scores = []
    def probabilities(batch):
        scores,_,_=predict_scores(model,np.asarray(batch,dtype=np.float32))
        observed_scores.extend(np.asarray(scores, dtype=np.float64).tolist())
        anomaly=anomaly_probability(scores,scale)
        return np.column_stack([1-anomaly, anomaly])
    if method=="lime":
        from lime import lime_image
        exp=lime_image.LimeImageExplainer(random_state=int(cfg.get("seed",42)))
        explanation=exp.explain_instance(image, probabilities,
            top_labels=2,num_samples=num_samples,hide_color=0)
        label=1
        weights=dict(explanation.local_exp[label])
        heat=np.zeros(explanation.segments.shape,dtype=np.float32)
        for key,value in weights.items():
            heat[explanation.segments==key]=value
    elif method=="shap":
        import shap
        masker=shap.maskers.Image("blur(8,8)",image.shape)
        explainer=shap.Explainer(probabilities,masker,output_names=["normal","anomaly"])
        result=explainer(image[None],max_evals=max(num_samples,2*image.size+1),outputs=[1])
        # (N,H,W,C,1): reduce absolute contributions to pixel map.
        heat=np.mean(np.abs(np.asarray(result.values)[0,...,0]),axis=-1)
    else:
        raise ValueError(f"Unsupported method: {method}")
    if observed_scores:
        low, high = min(observed_scores), max(observed_scores)
        print(f"[explain] Perturbed-image anomaly scores: min={low:.8g}, max={high:.8g}")
        if np.isclose(low, high, rtol=1e-6, atol=1e-10):
            print("[explain] WARNING: nearly constant model response to perturbations; explanation may be uninformative.")
    print(f"[explain] Explanation map: min={float(np.min(heat)):.8g}, max={float(np.max(heat)):.8g}, nonzero={int(np.count_nonzero(heat))}/{heat.size}")
    if not np.any(heat):
        print("[explain] WARNING: all-zero explanation; verify perturbation response, score scale and segmentation.")
    _,maps,recon=predict_scores(model,image[None])
    if output:
        save_explanation(image,recon[0],maps[0],heat,output)
        np.save(str(Path(output).with_suffix(".npy")),heat)
    return heat
