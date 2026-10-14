import pytest
torch=pytest.importorskip("torch")
from anomaly_detection.torch_models import VAEGAN
from anomaly_detection.torch_train import checkpoint_path

def test_variants_do_not_overwrite(tmp_path):
    config={"category":"screw","epochs":30000,"output_dir":str(tmp_path)}
    original=checkpoint_path(config)
    config["torch_variant"]="paper_faithful"
    corrected=checkpoint_path(config)
    assert original!=corrected
    assert original.name=="screw_VAE_GAN_30000.pt"
    assert corrected.name=="screw_VAE_GAN_30000_paper_faithful.pt"

def test_shared_forward_gradients():
    model=VAEGAN(4)
    model.train()
    x=torch.rand(2,3,128,128)
    mean,logvar,recon=model(x)
    batch=x.shape[0]
    logits=model.discriminator(torch.cat([x,recon]))
    real,fake=logits.split(batch)
    dloss=torch.nn.functional.binary_cross_entropy_with_logits(
        logits,torch.cat([torch.zeros_like(real),torch.ones_like(fake)]))
    extra=model.discriminator(recon)
    gen=torch.clamp_min(torch.nn.functional.binary_cross_entropy_with_logits(
        extra,torch.zeros_like(extra))-dloss,.0001)
    vae=gen+torch.nn.functional.binary_cross_entropy(recon,x)
    disc_params=list(model.discriminator.parameters())
    vae_params=list(model.encoder.parameters())+list(model.decoder.parameters())
    dgrads=torch.autograd.grad(dloss,disc_params,retain_graph=True)
    vgrads=torch.autograd.grad(vae,vae_params)
    assert all(torch.isfinite(g).all() for g in dgrads+vgrads)
