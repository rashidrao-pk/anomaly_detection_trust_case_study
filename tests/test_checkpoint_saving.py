from pathlib import Path
from types import SimpleNamespace
from anomaly_detection.train import checkpoint_prefix,save_notebook_checkpoints
def test_checkpoint_prefix(tmp_path):
    cfg={"_config_file":str(tmp_path/"config.yaml"),"category":"screw",
         "output_dir":str(tmp_path/"outputs"),"epochs":10}
    assert str(checkpoint_prefix(cfg)).endswith("trained_models/screw_VAE_GAN_10")
def test_notebook_saving_names(tmp_path):
    class Component:
        def __init__(self):self.calls=[]
        def save_weights(self, path, save_format):self.calls.append((path,save_format))
    model=Component()
    model.vae=Component()
    model.vae.encoder=Component()
    model.vae.decoder=Component()
    model.discriminator=Component()
    prefix=tmp_path/"screw_VAE_GAN_10"
    hist=SimpleNamespace(history={"vae_loss":[1,2],"disc_loss":[3,4],"gen_los":[5,6]})
    result=save_notebook_checkpoints(model,prefix,hist)
    assert result==str(prefix)+"_model"
    for suffix,comp in [("_model",model),("_vae",model.vae),("_encoder",model.vae.encoder),("_decoder",model.vae.decoder),("_discriminator",model.discriminator)]:
        assert comp.calls==[(str(prefix)+suffix,"tf")]
    assert (tmp_path/"screw_VAE_GAN_10.csv").exists()
