import unittest
import numpy as np
from tf_scripts.xai_utils import optimal_jaccard, paper_jaccard_visualization


class OracleIoUTests(unittest.TestCase):
    def test_perfect_segmentation(self):
        gt = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
        score, cutoff, pred = optimal_jaccard(
            gt, np.array([[0.9, 0.8, 0.3], [0.2, 0.1, 0.0]]))
        self.assertAlmostEqual(score, 1.0)
        np.testing.assert_array_equal(pred, gt)

    def test_score_ties_do_not_split(self):
        gt = np.array([[1, 0], [0, 0]], dtype=bool)
        score, cutoff, pred = optimal_jaccard(
            gt, np.array([[0.8, 0.8], [0.2, 0.2]]))
        self.assertAlmostEqual(score, 0.5)
        self.assertEqual(int(pred.sum()), 2)

    def test_normal_is_not_scored(self):
        score, cutoff, pred = optimal_jaccard(
            np.zeros((2,2), dtype=bool), np.ones((2,2)))
        self.assertTrue(np.isnan(score))

    def test_reference_colors(self):
        gt = np.array([[0, 1], [1, 0]], dtype=bool)
        pred = np.array([[0, 0], [1, 1]], dtype=bool)
        actual = paper_jaccard_visualization(gt, pred)
        expected = np.array([[[255,255,255], [0,255,255]],
                             [[0,0,0], [255,0,0]]])
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
