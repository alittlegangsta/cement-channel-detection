from __future__ import annotations

from typing import Literal

import numpy as np

from cement_channel.utils.angles import wrap_deg

RelBearingConvention = Literal["plus", "minus", "no_rotation"]


def align_azimuth_to_high_side(
    theta_raw_deg: float | np.ndarray,
    relbearing_deg: float | np.ndarray,
    *,
    convention: RelBearingConvention,
) -> float | np.ndarray:
    if convention == "plus":
        return wrap_deg(np.asarray(theta_raw_deg) + np.asarray(relbearing_deg))
    if convention == "minus":
        return wrap_deg(np.asarray(theta_raw_deg) - np.asarray(relbearing_deg))
    if convention == "no_rotation":
        return wrap_deg(theta_raw_deg)
    raise ValueError(f"Unsupported RelBearing convention: {convention}")


def aligned_azimuth_candidates(
    theta_raw_deg: float | np.ndarray,
    relbearing_deg: float | np.ndarray,
) -> dict[str, float | np.ndarray]:
    return {
        "plus": align_azimuth_to_high_side(
            theta_raw_deg,
            relbearing_deg,
            convention="plus",
        ),
        "minus": align_azimuth_to_high_side(
            theta_raw_deg,
            relbearing_deg,
            convention="minus",
        ),
        "no_rotation": align_azimuth_to_high_side(
            theta_raw_deg,
            relbearing_deg,
            convention="no_rotation",
        ),
    }


def orientation_confidence_from_inclination(
    inc_deg: float | np.ndarray,
    *,
    i_min_deg: float = 1.0,
    i_stable_deg: float = 5.0,
) -> float | np.ndarray:
    if i_stable_deg <= i_min_deg:
        raise ValueError("i_stable_deg must be greater than i_min_deg.")
    inc = np.asarray(inc_deg, dtype=np.float32)
    confidence = (inc - float(i_min_deg)) / (float(i_stable_deg) - float(i_min_deg))
    confidence = np.clip(confidence, 0.0, 1.0)
    confidence = np.where(np.isfinite(inc), confidence, 0.0).astype(np.float32)
    if np.isscalar(inc_deg):
        return float(confidence)
    return confidence


def orientation_uncertain_mask(
    inc_deg: float | np.ndarray,
    *,
    i_min_deg: float = 1.0,
) -> bool | np.ndarray:
    inc = np.asarray(inc_deg, dtype=np.float32)
    mask = (~np.isfinite(inc)) | (inc <= float(i_min_deg))
    if np.isscalar(inc_deg):
        return bool(mask)
    return mask.astype(bool)


def default_xsi_side_azimuth_deg(side_count: int = 8, *, theta0_deg: float = 0.0) -> np.ndarray:
    if side_count < 1:
        raise ValueError("side_count must be positive.")
    return wrap_deg(theta0_deg + np.arange(side_count, dtype=np.float32) * (360.0 / side_count))


def default_cast_azimuth_deg(azimuth_count: int = 180, *, theta0_deg: float = 0.0) -> np.ndarray:
    if azimuth_count < 1:
        raise ValueError("azimuth_count must be positive.")
    return wrap_deg(
        theta0_deg + np.arange(azimuth_count, dtype=np.float32) * (360.0 / azimuth_count)
    )
