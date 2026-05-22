from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class PlottingDependencyError(RuntimeError):
    """Raised when optional plotting dependencies are not installed."""


def require_pyplot() -> Any:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise PlottingDependencyError(
            "matplotlib is required for review figure generation. "
            "Install the optional plotting extra with: pip install -e .[plotting]"
        ) from exc
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig: Any, output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass overwrite=True.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.clear()
    require_pyplot().close(fig)


def sampled_depth_axis(depth: np.ndarray | None, row_count: int, max_rows: int) -> np.ndarray:
    if depth is None:
        values = np.arange(row_count, dtype=np.float32)
    else:
        values = np.asarray(depth, dtype=np.float32).reshape(-1)
        if values.size != row_count:
            values = np.arange(row_count, dtype=np.float32)
    return values[sample_indices(row_count, min(max_rows, row_count))]


def sampled_image(values: np.ndarray, *, max_rows: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D image array, observed shape {array.shape}.")
    return array[sample_indices(array.shape[0], min(max_rows, array.shape[0]))]


def sample_indices(length: int, target: int) -> np.ndarray:
    if length <= target:
        return np.arange(length)
    return np.linspace(0, length - 1, num=target).astype(int)


def finite_percentile_limits(
    values: np.ndarray,
    low: float = 1.0,
    high: float = 99.0,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def image_extent(
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> tuple[float, float, float, float]:
    x_values = np.asarray(x_axis, dtype=np.float32).reshape(-1)
    y_values = np.asarray(y_axis, dtype=np.float32).reshape(-1)
    if x_values.size == 0:
        x_values = np.array([0.0], dtype=np.float32)
    if y_values.size == 0:
        y_values = np.array([0.0], dtype=np.float32)
    return (
        float(np.nanmin(x_values)),
        float(np.nanmax(x_values)),
        float(np.nanmax(y_values)),
        float(np.nanmin(y_values)),
    )


def add_uncertain_row_spans(
    ax: Any,
    *,
    y_axis: np.ndarray,
    uncertain_rows: np.ndarray | None,
    color: str = "tab:red",
    alpha: float = 0.12,
) -> None:
    if uncertain_rows is None:
        return
    mask = np.asarray(uncertain_rows, dtype=bool).reshape(-1)
    y_values = np.asarray(y_axis, dtype=np.float32).reshape(-1)
    if mask.size != y_values.size:
        return
    if y_values.size > 1:
        half_step = float(np.nanmedian(np.abs(np.diff(y_values)))) / 2.0
    else:
        half_step = 0.5
    start: int | None = None
    for index, active in enumerate(mask.tolist() + [False]):
        if active and start is None:
            start = index
        if not active and start is not None:
            stop = index - 1
            y0 = float(min(y_values[start], y_values[stop]) - half_step)
            y1 = float(max(y_values[start], y_values[stop]) + half_step)
            ax.axhspan(y0, y1, color=color, alpha=alpha, linewidth=0)
            start = None
