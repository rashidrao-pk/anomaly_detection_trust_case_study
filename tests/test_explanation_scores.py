"""Dependency-light checks for the current LIME/SHAP reconstruction target."""
import numpy as np
import pytest
from anomaly_detection import explain


def test_reconstruction_mse_uses_sum_of_squared_rgb_error(monkeypatch):
    images = np.zeros((2, 3, 4, 3), dtype=np.float32)
    reconstructed = images.copy()
    reconstructed[0, 1, 2, 0] = 0.5
    reconstructed[1, :, :, :] = 0.25

    def fake_reconstruct(model, batch, deterministic=True):
        assert deterministic is True
        np.testing.assert_array_equal(batch, images)
        return reconstructed

    monkeypatch.setattr(explain, "reconstruct", fake_reconstruct)
    scores = explain.reconstruction_mse(object(), images)
    np.testing.assert_allclose(scores, [0.25, 3 * 4 * 3 * 0.25**2], rtol=1e-6)


def test_reconstruction_mse_zero_for_identical_images(monkeypatch):
    images = np.random.default_rng(1).random((2, 3, 4, 3), dtype=np.float32)
    monkeypatch.setattr(explain, "reconstruct",
                        lambda model, batch, deterministic=True: batch.copy())
    np.testing.assert_array_equal(explain.reconstruction_mse(None, images),
                                  np.zeros(2, dtype=np.float32))
