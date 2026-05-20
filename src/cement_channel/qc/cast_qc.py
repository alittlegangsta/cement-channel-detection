from __future__ import annotations

import numpy as np

from cement_channel.qc.xsi_qc import ArrayQCResult, summarize_numeric_array


def run_cast_zc_qc(cast_zc: np.ndarray) -> ArrayQCResult:
    result = summarize_numeric_array("cast_zc", cast_zc)
    warnings = list(result.warnings)
    errors = list(result.errors)
    if cast_zc.ndim != 2:
        errors.append(f"cast_zc must be rank 2 [depth, cast_azimuth], observed {cast_zc.shape}")
    else:
        if cast_zc.shape[1] != 180:
            warnings.append(f"cast_zc azimuth count is {cast_zc.shape[1]}, expected 180.")
    if result.min is not None and result.min < 0:
        warnings.append("cast_zc contains negative values; confirm units and preprocessing.")
    return _replace_messages(result, warnings, errors)


def run_pose_range_qc(
    inc_deg: np.ndarray,
    rel_bearing_deg: np.ndarray,
) -> dict[str, ArrayQCResult]:
    inc = summarize_numeric_array("inc_deg", inc_deg)
    rel = summarize_numeric_array("rel_bearing_deg", rel_bearing_deg)
    inc_warnings = list(inc.warnings)
    rel_warnings = list(rel.warnings)
    if inc.min is not None and inc.max is not None and (inc.min < 0 or inc.max > 180):
        inc_warnings.append("inc_deg values fall outside [0, 180].")
    if rel.min is not None and rel.max is not None and (rel.min < 0 or rel.max >= 360):
        rel_warnings.append("rel_bearing_deg values fall outside [0, 360).")
    return {
        "inc_deg": _replace_messages(inc, inc_warnings, list(inc.errors)),
        "rel_bearing_deg": _replace_messages(rel, rel_warnings, list(rel.errors)),
    }


def _replace_messages(
    result: ArrayQCResult,
    warnings: list[str],
    errors: list[str],
) -> ArrayQCResult:
    return ArrayQCResult(
        name=result.name,
        shape=result.shape,
        dtype=result.dtype,
        finite_ratio=result.finite_ratio,
        nan_ratio=result.nan_ratio,
        inf_ratio=result.inf_ratio,
        zero_ratio=result.zero_ratio,
        clipping_like_ratio=result.clipping_like_ratio,
        min=result.min,
        max=result.max,
        mean=result.mean,
        std=result.std,
        warnings=warnings,
        errors=errors,
    )
