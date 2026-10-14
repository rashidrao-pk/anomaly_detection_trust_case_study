from pathlib import Path
import pytest
from PIL import Image
import numpy as np

from scripts.explain import choose_image

def test_select_from_config(tmp_path):
    root = tmp_path / "mvtec"
    folder = root / "screw" / "test" / "scratch_head"
    folder.mkdir(parents=True)
    for filename in ("002.png", "001.png"):
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(folder / filename)
    cfg = {"dataset_root": str(root), "category": "screw",
           "explanation": {"anomaly_type": "scratch_head", "image_index": 1}}
    assert choose_image(cfg).name == "002.png"
    assert choose_image(cfg, index=0).name == "001.png"
    with pytest.raises(IndexError):
        choose_image(cfg, index=2)
