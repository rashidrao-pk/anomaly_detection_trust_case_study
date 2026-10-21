"""Visualization and MVTec annotation helper regression checks."""
import unittest
import numpy as np
from tf_scripts.xai_utils import attribution_map, shap_spatial_map


class AttributionTests(unittest.TestCase):
    def test_lime_signs(self):
        actual = attribution_map(np.array([[0, 1], [2, 1]]),
                                 [(0, -0.3), (1, 0.8)])
        np.testing.assert_allclose(actual, [[-0.3, 0.8], [0.0, 0.8]])

    def test_shap_class_one(self):
        values = np.zeros((1, 2, 2, 3, 2))
        values[..., 1] = 0.5
        np.testing.assert_allclose(shap_spatial_map(values, (2, 2, 3)),
                                   np.full((2, 2), 0.5))


if __name__ == "__main__":
    unittest.main()
