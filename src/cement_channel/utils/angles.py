from __future__ import annotations

from typing import Any

import numpy as np


def wrap_deg(theta: float | np.ndarray) -> float | np.ndarray:
    """Wrap degrees to [0, 360)."""
    wrapped = np.mod(theta, 360.0)
    if np.isscalar(theta):
        return float(wrapped)
    return np.asarray(wrapped, dtype=np.float32)


def circular_distance_deg(
    theta_a: float | np.ndarray,
    theta_b: float | np.ndarray,
) -> float | np.ndarray:
    """Return the unsigned minimum circular distance in degrees."""
    delta = np.abs(np.mod(np.asarray(theta_a) - np.asarray(theta_b) + 180.0, 360.0) - 180.0)
    if np.isscalar(theta_a) and np.isscalar(theta_b):
        return float(delta)
    return delta.astype(np.float32)


def signed_circular_delta_deg(
    theta_a: float | np.ndarray,
    theta_b: float | np.ndarray,
) -> float | np.ndarray:
    """Return signed circular delta theta_a - theta_b in [-180, 180)."""
    delta = np.mod(np.asarray(theta_a) - np.asarray(theta_b) + 180.0, 360.0) - 180.0
    if np.isscalar(theta_a) and np.isscalar(theta_b):
        return float(delta)
    return delta.astype(np.float32)


def circular_mean_deg(
    angles_deg: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    axis: int | None = None,
) -> float | np.ndarray:
    """Compute circular mean in degrees."""
    angles = np.asarray(angles_deg, dtype=np.float64)
    radians = np.deg2rad(angles)
    if weights is None:
        sin_mean = np.nanmean(np.sin(radians), axis=axis)
        cos_mean = np.nanmean(np.cos(radians), axis=axis)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        sin_mean = np.nansum(np.sin(radians) * weight_array, axis=axis) / np.nansum(
            weight_array,
            axis=axis,
        )
        cos_mean = np.nansum(np.cos(radians) * weight_array, axis=axis) / np.nansum(
            weight_array,
            axis=axis,
        )
    mean = np.rad2deg(np.arctan2(sin_mean, cos_mean))
    wrapped = wrap_deg(mean)
    if np.isscalar(wrapped) or np.asarray(wrapped).shape == ():
        return float(wrapped)
    return np.asarray(wrapped, dtype=np.float32)


def as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)
