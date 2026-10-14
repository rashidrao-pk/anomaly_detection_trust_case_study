"""Native PyTorch training, resumable checkpoints and evaluation."""
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import resolve_path
from .torch_models import VAEGAN, paper_losses
from .torch_data import PaperTrainingDataset, EvaluationDataset
from .scoring import reconstruction_error, image_scores
from sklearn.metrics import roc_auc_score


def safe_auroc(labels, scores):
    labels = np.asarray(labels)
    if np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def device_for(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but PyTorch has no CUDA GPU")
    return torch.device(name)


def output_root(cfg):
    return resolve_path(cfg, cfg.get("output_dir", "outputs")) / "pytorch"


def checkpoint_path(cfg):
    variant = cfg.get("torch_variant", "initial")
    if variant not in ("initial", "paper_faithful"):
        raise ValueError(f"Unknown torch_variant: {variant}")
    suffix = "" if variant == "initial" else "_paper_faithful"
    return output_root(cfg) / "trained_models" / (
        f"{cfg['category']}_VAE_GAN_{int(cfg.get('epochs', 10))}{suffix}.pt")


def load_checkpoint(cfg, path, device):
    # Explicit weights_only avoids loading arbitrary Python objects.
    state = torch.load(path, map_location=device, weights_only=True)
    model = VAEGAN(int(state["latent_dim"])).to(device)
    model.load_state_dict(state["model"])
    return model, state


def fit(cfg, *, device_name="auto", resume=False, num_workers=0, save_every=1,
        no_augmentation=False):
    device = device_for(device_name)
    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    root = resolve_path(cfg, cfg["dataset_root"])
    if int(cfg.get("image_size", 128)) != 128:
        raise ValueError("Archived architecture requires 128x128 images")
    mode = "none" if no_augmentation or cfg.get("training_profile", "paper") == "modern" else "paper"
    ds = PaperTrainingDataset(root, cfg["category"], 128, seed, mode)
    loader = DataLoader(ds, batch_size=int(cfg.get("batch_size", 32)),
                        shuffle=True, num_workers=num_workers, drop_last=False)
    model = VAEGAN(int(cfg.get("latent_dim", 32))).to(device)
    lr = float(cfg.get("learning_rate", .001))
    variant = cfg.get("torch_variant", "initial")
    if variant not in ("initial", "paper_faithful"):
        raise ValueError(f"Unknown torch_variant: {variant}")
    epsilon = 1e-7 if variant == "paper_faithful" else 1e-8
    vae_opt = torch.optim.Adam(list(model.encoder.parameters())+list(model.decoder.parameters()),lr=lr,eps=epsilon)
    disc_opt = torch.optim.Adam(model.discriminator.parameters(),lr=lr,eps=epsilon)
    path = checkpoint_path(cfg); path.parent.mkdir(parents=True, exist_ok=True)
    start=0
    history=[]
    if resume:
        model, state = load_checkpoint(cfg, path, device)
        vae_opt.load_state_dict(state["vae_optimizer"])
        disc_opt.load_state_dict(state["disc_optimizer"])
        start = int(state["epoch"])
        history = state.get("history", [])
        if state.get("torch_variant", "initial") != variant:
            raise ValueError("Checkpoint training variant differs from requested variant")
    elif path.exists():
        raise FileExistsError(f"Will not overwrite {path}; use --resume or change output_dir/epochs")
    total = int(cfg.get("epochs",10))
    if resume and start >= total:
        raise ValueError(f"Checkpoint already at epoch {start}, target is {total}")
    dataset_dir = root / cfg["category"]
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu_line = (
            f"[GPU] name={props.name} | memory={props.total_memory / (1024 ** 3):.1f} GiB "
            f"| CUDA={torch.version.cuda} | device={device}"
        )
    elif device.type == "mps":
        gpu_line = "[GPU] name=Apple Metal/MPS | memory=shared | device=mps"
    else:
        gpu_line = "[GPU] name=CPU | memory=N/A | device=cpu"
    print(f"[DATASET] path={dataset_dir} | category={cfg['category']} | training_images={len(ds)}", flush=True)
    print(gpu_line, flush=True)
    print(
        f"[TRAIN] variant={variant} | epochs={total} | starting_epoch={start} "
        f"| batch_size={cfg.get('batch_size', 32)} | image_size=128 "
        f"| latent_dim={cfg.get('latent_dim', 32)} | lr={lr:g} | "
        f"augmentation={mode} | workers={num_workers} | save_every={save_every} "
        f"| checkpoint={path}", flush=True
    )
    for epoch in range(start,total):
        model.train()
        sums = {k:0. for k in ("vae_loss","reconstruction_loss","kl_loss","disc_loss","gen_loss")}
        count = 0
        t0 = time.perf_counter()
        for images in loader:
            images = images.to(device)
            batch = images.shape[0]
            mean, logvar, recon = model(images)
            if variant == "paper_faithful":
                # Original models.py computes D and VAE losses from the same
                # forward parameters, then applies both gradient sets.
                # Discriminator is called once on a concatenated real/fake batch,
                # then a second time on generated data as in the TF source.
                combined_logits = model.discriminator(torch.cat((images, recon), dim=0))
                real_logits, fake_logits = combined_logits.split(batch, dim=0)
                extra_fake_logits = model.discriminator(recon)
                from torch.nn import functional as F
                target_real = torch.zeros_like(real_logits)
                target_fake = torch.ones_like(fake_logits)
                dloss = F.binary_cross_entropy_with_logits(
                    combined_logits, torch.cat((target_real, target_fake), dim=0))
                gen_loss = torch.clamp_min(
                    F.binary_cross_entropy_with_logits(extra_fake_logits, target_real) - dloss,
                    0.0001)
                reconstruction = F.binary_cross_entropy(recon, images, reduction="none").mean(dim=1).sum(dim=(1, 2)).mean()
                kl_loss = (-.5 * (1 + logvar - mean.square() - logvar.exp())).sum(dim=1).mean()
                vae_loss = reconstruction + kl_loss + gen_loss
                losses = {"vae_loss": vae_loss, "reconstruction_loss": reconstruction,
                          "kl_loss": kl_loss, "disc_loss": dloss, "gen_loss": gen_loss}
                vae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
                disc_params = list(model.discriminator.parameters())
                # Compute BOTH gradient sets before either optimizer step.
                # autograd.grad returns no graph-owned .grad side effects.
                dgrads = torch.autograd.grad(dloss, disc_params, retain_graph=True)
                vgrads = torch.autograd.grad(vae_loss, vae_params)
                disc_opt.zero_grad(set_to_none=True)
                vae_opt.zero_grad(set_to_none=True)
                for p, g in zip(disc_params, dgrads):
                    p.grad = g
                for p, g in zip(vae_params, vgrads):
                    p.grad = g
                disc_opt.step()
                vae_opt.step()
            else:
                # D step: update only discriminator using *detached* generated image
                disc_opt.zero_grad(set_to_none=True)
                rlog = model.discriminator(images)
                flog = model.discriminator(recon.detach())
                dloss = nn.functional.binary_cross_entropy_with_logits(
                    torch.cat((rlog,flog)), torch.cat((torch.zeros_like(rlog),torch.ones_like(flog))))
                dloss.backward()
                disc_opt.step()
                # VAE step: freeze discriminator weights; retain gradient to reconstruction.
                for p in model.discriminator.parameters(): p.requires_grad_(False)
                vae_opt.zero_grad(set_to_none=True)
                # Keras computes losses using the *same* forward state. The two
                # sequential optimizer steps are an approximation, not exact parity.
                rlog2 = model.discriminator(images)
                flog2 = model.discriminator(recon)
                losses = paper_losses(images,recon,mean,logvar,rlog2,flog2)
                losses["vae_loss"].backward()
                vae_opt.step()
                for p in model.discriminator.parameters(): p.requires_grad_(True)
            for key in sums: sums[key] += float(losses[key].detach()) * batch
            count += batch
        row={"epoch":epoch+1,"seconds":round(time.perf_counter()-t0,2)}
        row.update({k:sums[k]/count for k in sums})
        history.append(row)
        print(
            f"[epoch {epoch+1:03d}/{total}] "
            f"vae_loss={row['vae_loss']:.4f} "
            f"reconstruction_loss={row['reconstruction_loss']:.4f} "
            f"kl_loss={row['kl_loss']:.4f} "
            f"disc_loss={row['disc_loss']:.8f} "
            f"gen_loss={row['gen_loss']:.6f} "
            f"time={row['seconds']:.1f}s", flush=True
        )
        if (epoch+1)%save_every==0 or epoch+1==total:
            state={"epoch":epoch+1,"latent_dim":int(cfg.get("latent_dim",32)),
                   "model":model.state_dict(),"vae_optimizer":vae_opt.state_dict(),
                   "disc_optimizer":disc_opt.state_dict(),"history":history,
                   "category":cfg["category"],"image_size":128,
                   "seed":seed,"training_profile":mode,"torch_variant":variant}
            tmp=path.with_suffix(".pt.tmp")
            torch.save(state,tmp)
            tmp.replace(path)
            with path.with_suffix(".csv").open("w",newline="") as out:
                writer=csv.DictWriter(out,fieldnames=list(history[0]))
                writer.writeheader();writer.writerows(history)
    return path


@torch.inference_mode()
def evaluate(cfg, path=None, *, device_name="auto", num_workers=0):
    device = device_for(device_name)
    path = Path(path) if path else checkpoint_path(cfg)
    model, state = load_checkpoint(cfg,path,device)
    model.eval()
    ds=EvaluationDataset(resolve_path(cfg,cfg["dataset_root"]),cfg["category"],128)
    loader=DataLoader(ds,batch_size=int(cfg.get("batch_size",32)),shuffle=False,num_workers=num_workers)
    rows=[]; masks=[]; maps=[]
    for images, labels, mask, paths in loader:
        images=images.to(device)
        recon=model.reconstruct(images)
        err=(images-recon).abs().mean(dim=1)
        scores=err.flatten(1).amax(dim=1)
        for i,p in enumerate(paths):
            rows.append({"path":p,"label":int(labels[i]),"score":float(scores[i])})
        masks.append(mask.numpy().ravel());maps.append(err.cpu().numpy().ravel())
    if not rows: raise ValueError("No evaluation samples")
    result={"backend":"pytorch","category":cfg["category"],"checkpoint":str(path),
            "samples":len(rows),
            "image_auroc":safe_auroc([x["label"] for x in rows],[x["score"] for x in rows]),
            "pixel_auroc":safe_auroc(np.concatenate(masks),np.concatenate(maps))}
    out=output_root(cfg)/"evaluation";out.mkdir(parents=True,exist_ok=True)
    with (out/"image_scores.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["path","label","score"]);w.writeheader();w.writerows(rows)
    (out/"metrics.json").write_text(json.dumps(result,indent=2))
    return result
