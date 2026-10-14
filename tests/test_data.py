import numpy as np
from PIL import Image
from anomaly_detection.data import evaluation_samples,load_mask,training_paths

def test_mvtec_order_and_matching(tmp_path):
    root=tmp_path/"data"
    (root/"screw"/"test"/"good").mkdir(parents=True)
    (root/"screw"/"test"/"scratch").mkdir()
    (root/"screw"/"ground_truth"/"scratch").mkdir(parents=True)
    (root/"screw"/"train"/"good").mkdir(parents=True)
    im=Image.fromarray(np.zeros((4,4,3),dtype=np.uint8))
    im.save(root/"screw"/"test"/"good"/"1.png")
    im.save(root/"screw"/"test"/"scratch"/"2.png")
    im.save(root/"screw"/"train"/"good"/"0.png")
    Image.fromarray(np.full((4,4),255,dtype=np.uint8)).save(root/"screw"/"ground_truth"/"scratch"/"2_mask.png")
    items=list(evaluation_samples(root,"screw"))
    assert [x[1] for x in items]==[0,1]
    assert items[0][2] is None
    assert load_mask(items[1][2],4).all()
    assert len(training_paths(root,"screw"))==1
