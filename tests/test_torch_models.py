import pytest
torch=pytest.importorskip("torch")
from anomaly_detection.torch_models import VAEGAN,paper_losses

def test_architecture_shapes():
    model=VAEGAN(32).eval()
    with torch.no_grad():
        x=torch.rand(2,3,128,128)
        mean, logvar, reconstruction=model(x,sample=False)
        assert mean.shape==logvar.shape==(2,32)
        assert reconstruction.shape==x.shape
        assert model.discriminator(x).shape==(2,1)
        assert torch.all((reconstruction>=0)&(reconstruction<=1))

def test_paper_loss_scaling():
    x=torch.ones(2,3,128,128)
    r=torch.full_like(x,.5)
    mean=torch.zeros(2,32)
    logvar=torch.zeros_like(mean)
    a=paper_losses(x,r,mean,logvar,torch.zeros(2,1),torch.zeros(2,1))
    assert torch.isclose(a["reconstruction_loss"],torch.tensor(128*128*.69314718),atol=.01)
    assert a["kl_loss"]==0
    assert torch.isfinite(a["vae_loss"])
