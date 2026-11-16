"""CPU-only regression tests for anomaly scoring utilities."""
import unittest

import numpy as np

from tf_scripts.core import residual_map
from tf_scripts.xai_utils import anomaly_score_predictor


class _TensorLike:
    """Minimal tensor stand-in for testing without TensorFlow model weights."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


class _ZeroReconstructionModel:
    """Model-shaped test double that reconstructs every image as black."""

    def encoder(self, images, training=False):
        images = np.asarray(images)
        return images, np.zeros_like(images), images

    def decoder(self, latent, training=False):
        return _TensorLike(np.zeros_like(latent))


class ResidualMapTests(unittest.TestCase):
    def test_identical_images_have_zero_residual(self):
        image = np.full((2, 3, 3), 0.5, dtype=np.float32)

        actual = residual_map(image, image.copy())

        self.assertEqual(actual.shape, (2, 3))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, np.zeros((2, 3), dtype=np.float32))

    def test_residual_is_channel_mean_absolute_error(self):
        image = np.zeros((1, 2, 3), dtype=np.float32)
        reconstruction = np.array(
            [[[1.0, 2.0, 3.0], [255.0, 0.0, 3.0]]], dtype=np.float32
        )

        actual = residual_map(image, reconstruction)

        np.testing.assert_allclose(actual, [[2.0, 86.0]])
        self.assertTrue(np.isfinite(actual).all())

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            residual_map(np.zeros((2, 2, 3)), np.zeros((2, 2, 1)))

    def test_rejects_arrays_without_channels(self):
        with self.assertRaises(ValueError):
            residual_map(np.zeros((2, 2)), np.zeros((2, 2)))


class AnomalyScorePredictorTests(unittest.TestCase):
    def test_outputs_probabilities_with_expected_shape_and_dtype(self):
        image = np.ones((2, 2, 3), dtype=np.float32)
        predictor = anomaly_score_predictor(
            _ZeroReconstructionModel(), image, np.zeros_like(image), batch_size=2
        )

        actual = predictor(
            np.stack(
                [np.zeros_like(image), image, np.full_like(image, 2.0)], axis=0
            )
        )

        self.assertEqual(actual.shape, (3, 2))
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual.sum(axis=1), np.ones(3))
        np.testing.assert_allclose(actual[:, 1], [0.0, 0.5, 0.8])

    def test_batches_do_not_change_scores(self):
        image = np.ones((2, 2, 3), dtype=np.float32)
        samples = np.stack(
            [np.zeros_like(image), image, np.full_like(image, 2.0)], axis=0
        )
        predictor = anomaly_score_predictor(
            _ZeroReconstructionModel(), image, np.zeros_like(image), batch_size=1
        )

        actual = predictor(samples)

        self.assertEqual(actual.shape, (3, 2))
        self.assertTrue(np.all(np.diff(actual[:, 1]) > 0))


if __name__ == "__main__":
    unittest.main()
