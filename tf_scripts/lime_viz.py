"""Utilities for displaying actual LIME superpixel coefficients."""
import numpy as np


def signed_attribution(segments, coefficients):
    """Map (superpixel_id, signed_weight) to a full-resolution attribution map.

    Missing or zero coefficients remain zero; do not interpret zero as negative.
    """
    seg = np.asarray(segments)
    if seg.ndim != 2:
        raise ValueError("LIME segmentation must be 2D")
    result = np.zeros(seg.shape, dtype=np.float32)
    for segment_id, weight in coefficients:
        result[seg == int(segment_id)] = float(weight)
    return result
