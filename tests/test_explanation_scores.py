import numpy as np
import pytest
from anomaly_detection.explain import anomaly_probability

def test_no_saturation_for_scores_above_scale():
    y=anomaly_probability([0.1,0.2,1.0],0.1)
    assert 0 < y[0] < y[1] < y[2] < 1.0

def test_zero_score_and_invalid_scale():
    assert anomaly_probability([0],0.1)[0] == 0
    with pytest.raises(ValueError):
        anomaly_probability([0.1],0)
