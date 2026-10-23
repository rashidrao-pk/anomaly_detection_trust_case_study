import unittest
import numpy as np
from tf_scripts.xai_utils import paper_shap_spatial_map, paper_segments


class PaperXaiTests(unittest.TestCase):
    def test_first_channel_one_output(self):
        a = np.zeros((1, 2, 3, 3, 1))
        a[0, :, :, 0, 0] = 2
        a[0, :, :, 1, 0] = -4
        np.testing.assert_array_equal(paper_shap_spatial_map(a, (2,3,3)),
                                      np.full((2,3), 2))

    def test_shap_rejects_two_output_values(self):
        with self.assertRaises(ValueError):
            paper_shap_spatial_map(np.zeros((1,2,3,3,2)),(2,3,3))


if __name__ == "__main__":
    unittest.main()
