"""Small dependency-light regression tests for coefficient rendering."""
import unittest
import numpy as np
from tf_scripts.lime_viz import signed_attribution


class LimeAttributionTests(unittest.TestCase):
    def test_weights_preserve_sign_and_magnitude(self):
        segments = np.array([[0, 0, 1], [2, 2, 1]])
        heatmap = signed_attribution(segments, [(0, -0.1), (1, 0.7)])
        np.testing.assert_allclose(heatmap, [
            [-0.1, -0.1, 0.7], [0., 0., 0.7]
        ])

    def test_invalid_segments(self):
        with self.assertRaises(ValueError):
            signed_attribution(np.zeros((2, 2, 3)), [(0, 1)])


if __name__ == "__main__":
    unittest.main()
