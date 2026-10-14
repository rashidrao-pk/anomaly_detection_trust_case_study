"""Notebook-compatible checkpoint naming, history and training resume."""
from pathlib import Path
from types import SimpleNamespace
import csv
import random
import numpy as np
from .config import resolve_path
from .data import make_train_dataset,make_paper_train_generator

def checkpoint_prefix(cfg):
    output=resolve_path(cfg,cfg.get("output_dir","outputs"))/"trained_models"
    return output/f"{cfg['category']}_VAE_GAN_{int(cfg.get('epochs',10))}"

def read_history(prefix):
    path=Path(str(prefix)+".csv")
    if not path.exists(): return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))

def save_notebook_checkpoints(model,prefix,history):
    prefix=Path(prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    for suffix,component in (("_model",model),("_vae",model.vae),
                             ("_encoder",model.vae.encoder),("_decoder",model.vae.decoder),
                             ("_discriminator",model.discriminator)):
        component.save_weights(str(prefix)+suffix,save_format="tf")
    with Path(str(prefix)+".csv").open("w",newline="") as stream:
        fields=list(history.history)
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(zip(fields,row)) for row in zip(*(history.history[k] for k in fields)))
    return str(prefix)+"_model"

def _reload_components(model,prefix):
    for suffix,component in (("_model",model),("_vae",model.vae),
                             ("_encoder",model.vae.encoder),("_decoder",model.vae.decoder),
                             ("_discriminator",model.discriminator)):
        source=str(prefix)+suffix
        if not Path(source+".index").exists():
            raise FileNotFoundError(f"Resume checkpoint missing: {source}.index")
        status=component.load_weights(source)
        if hasattr(status,"expect_partial"): status.expect_partial()

def train(cfg,resume=False):
    import tensorflow as tf
    from .models import build_model
    seed=int(cfg.get("seed",42))
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    root=resolve_path(cfg,cfg["dataset_root"])
    profile=cfg.get("training_profile","paper")
    image_size=int(cfg.get("image_size",128))
    batch_size=int(cfg.get("batch_size",32))
    if profile=="paper":
        data=make_paper_train_generator(root,cfg["category"],image_size,batch_size,seed)
    elif profile=="modern":
        data=make_train_dataset(root,cfg["category"],image_size,batch_size,seed)
    else: raise ValueError(f"Unknown training_profile: {profile}")
    model=build_model(int(cfg.get("latent_dim",32)),image_size,float(cfg.get("learning_rate",.001)))
    prefix=checkpoint_prefix(cfg)
    history_rows=read_history(prefix) if resume else []
    if resume:
        if not history_rows: raise FileNotFoundError(f"Resume requested but history absent: {prefix}.csv")
        _reload_components(model,prefix)
        print(f"[resume] Restored checkpoint and {len(history_rows)} history rows")
        print("[resume] This matches notebook weight-based resume; optimizer state restoration is not guaranteed.")
    start_epoch=len(history_rows)
    additional=int(cfg.get("resume_epochs",cfg.get("epochs",10))) if resume else int(cfg.get("epochs",10))
    if additional<=0: raise ValueError("Number of epochs must be positive")
    history=model.fit(data,initial_epoch=start_epoch,epochs=start_epoch+additional,verbose=1)
    old_keys=list(history_rows[0]) if history_rows else []
    new_keys=list(history.history)
    if history_rows and set(old_keys)!=set(new_keys):
        raise ValueError(f"History columns changed: {old_keys} vs {new_keys}")
    merged={key:[float(row[key]) for row in history_rows]+[float(v) for v in history.history[key]]
            for key in new_keys}
    return save_notebook_checkpoints(model,prefix,SimpleNamespace(history=merged))
