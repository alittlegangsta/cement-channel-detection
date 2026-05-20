from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArrayQCResult:
    name: str
    shape: list[int]
    dtype: str
    finite_ratio: float | None
    nan_ratio: float | None
    inf_ratio: float | None
    zero_ratio: float | None
    clipping_like_ratio: float | None
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_xsi_waveform_qc(waveform: np.ndarray) -> ArrayQCResult:
    result = summarize_numeric_array("xsi_waveform", waveform)
    warnings = list(result.warnings)
    errors = list(result.errors)
    if waveform.ndim != 4:
        errors.append(
            "xsi_waveform must be rank 4 [depth, receiver, side, time], "
            f"observed {waveform.shape}"
        )
    else:
        if waveform.shape[2] != 8:
            warnings.append(f"xsi_waveform side count is {waveform.shape[2]}, expected 8.")
        if waveform.shape[3] > 1024:
            warnings.append("xsi_waveform time dimension exceeds expected raw sample count 1024.")
    return _replace_messages(result, warnings, errors)


def summarize_numeric_array(name: str, array: np.ndarray) -> ArrayQCResult:
    values = np.asarray(array)
    warnings: list[str] = []
    errors: list[str] = []
    if values.size == 0:
        errors.append(f"{name} is empty.")
        return ArrayQCResult(
            name=name,
            shape=[int(item) for item in values.shape],
            dtype=str(values.dtype),
            finite_ratio=None,
            nan_ratio=None,
            inf_ratio=None,
            zero_ratio=None,
            clipping_like_ratio=None,
            min=None,
            max=None,
            mean=None,
            std=None,
            warnings=warnings,
            errors=errors,
        )

    finite = np.isfinite(values)
    if np.issubdtype(values.dtype, np.floating):
        nan = np.isnan(values)
        inf = np.isinf(values)
    else:
        nan = np.zeros(values.shape, dtype=bool)
        inf = np.zeros(values.shape, dtype=bool)
    finite_ratio = float(np.mean(finite))
    nan_ratio = float(np.mean(nan))
    inf_ratio = float(np.mean(inf))
    zero_ratio = float(np.mean(values == 0))
    finite_values = values[finite]
    if finite_values.size == 0:
        errors.append(f"{name} has no finite values.")
        return ArrayQCResult(
            name=name,
            shape=[int(item) for item in values.shape],
            dtype=str(values.dtype),
            finite_ratio=finite_ratio,
            nan_ratio=nan_ratio,
            inf_ratio=inf_ratio,
            zero_ratio=zero_ratio,
            clipping_like_ratio=None,
            min=None,
            max=None,
            mean=None,
            std=None,
            warnings=warnings,
            errors=errors,
        )

    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    max_abs = float(np.max(np.abs(finite_values)))
    clipping_like_ratio = 0.0
    if max_abs > 0:
        clipping_like_ratio = float(np.mean(np.abs(finite_values) >= max_abs))
    if finite_ratio < 1.0:
        warnings.append(f"{name} contains non-finite values.")
    if clipping_like_ratio > 0.1:
        warnings.append(f"{name} has high clipping-like ratio: {clipping_like_ratio:.3f}.")

    return ArrayQCResult(
        name=name,
        shape=[int(item) for item in values.shape],
        dtype=str(values.dtype),
        finite_ratio=finite_ratio,
        nan_ratio=nan_ratio,
        inf_ratio=inf_ratio,
        zero_ratio=zero_ratio,
        clipping_like_ratio=clipping_like_ratio,
        min=min_value,
        max=max_value,
        mean=float(np.mean(finite_values)),
        std=float(np.std(finite_values)),
        warnings=warnings,
        errors=errors,
    )


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
