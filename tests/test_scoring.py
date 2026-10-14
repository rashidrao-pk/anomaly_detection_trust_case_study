import numpy as np
import pytest
from anomaly_detection.scoring import reconstruction_error,image_scores

def test_zero_error():
    x=np.zeros((2,8,8,3),dtype=np.float32)
    maps=reconstruction_error(x,x)
    assert maps.shape==(2,8,8)
    np.testing.assert_array_equal(image_scores(maps),[0,0])

def test_localized_error():
    x=np.zeros((1,4,4,3),dtype=np.float32)
    y=x.copy()
    y[0,1,2,:]=1
    assert image_scores(reconstruction_error(x,y))[0]==pytest.approx(1.0)

def test_reject_shape_mismatch():
    with pytest.raises(ValueError):
        reconstruction_error(np.zeros((1,8,8,3)),np.zeros((1,4,4,3)))
