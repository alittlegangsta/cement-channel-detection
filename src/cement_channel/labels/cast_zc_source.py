from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

RAW_ZC_FIELD_CANDIDATES = ("cast_zc", "raw_zc", "zc", "Zc")
FORBIDDEN_RAW_ZC_FIELDS = ("zc_ratio", "relative_drop", "relative_drop_plus")
DEPTH_FIELD_CANDIDATES = ("cast_depth", "depth")
AZIMUTH_FIELD_CANDIDATES = ("cast_azimuth_deg", "cast_azimuth", "azimuth_deg")


@dataclass(frozen=True)
class CastZcSourceReport:
    source_file: str
    source_field: str
    depth_field: str
    azimuth_field: str
    shape: tuple[int, int]
    finite_ratio: float
    depth_count: int
    azimuth_count: int
    depth_min: float
    depth_max: float
    azimuth_min: float
    azimuth_max: float
    warnings: list[str]
    no_final_labels: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CastZcSource:
    cast_zc: np.ndarray
    cast_depth: np.ndarray
    cast_azimuth_deg: np.ndarray
    report: CastZcSourceReport


class CastZcSourceError(RuntimeError):
    """Raised when a controlled raw CAST Zc source is unavailable or invalid."""


def load_controlled_cast_zc_source(
    source_paths: Sequence[Path | str],
    *,
    min_finite_ratio: float = 0.5,
) -> CastZcSource:
    """Load raw CAST Zc from controlled NPZ sources, never from raw MAT files."""

    attempted: list[str] = []
    missing_raw: list[str] = []
    for path_like in source_paths:
        path = Path(path_like)
        attempted.append(str(path))
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        try:
            return build_cast_zc_source_from_arrays(
                arrays,
                source_file=path,
                min_finite_ratio=min_finite_ratio,
            )
        except CastZcSourceError as exc:
            missing_raw.append(f"{path}: {exc}")
            continue
    details = "; ".join(missing_raw) if missing_raw else "no candidate source file exists"
    raise CastZcSourceError(
        "Controlled raw CAST Zc is unavailable; attempted "
        + ", ".join(attempted)
        + f" ({details}). Refusing to use zc_ratio or fabricate raw Zc."
    )


def build_cast_zc_source_from_arrays(
    arrays: dict[str, np.ndarray],
    *,
    source_file: Path | str,
    min_finite_ratio: float = 0.5,
) -> CastZcSource:
    source_field = _first_present(arrays, RAW_ZC_FIELD_CANDIDATES)
    if source_field is None:
        forbidden = [field for field in FORBIDDEN_RAW_ZC_FIELDS if field in arrays]
        if forbidden:
            raise CastZcSourceError(
                "NPZ has derived CAST fields "
                + ", ".join(forbidden)
                + " but no raw Zc field."
            )
        raise CastZcSourceError("NPZ does not contain a controlled raw CAST Zc field.")
    depth_field = _first_present(arrays, DEPTH_FIELD_CANDIDATES)
    azimuth_field = _first_present(arrays, AZIMUTH_FIELD_CANDIDATES)
    if depth_field is None:
        raise CastZcSourceError("NPZ missing CAST depth grid.")
    if azimuth_field is None:
        raise CastZcSourceError("NPZ missing CAST azimuth grid.")

    cast_zc = np.asarray(arrays[source_field], dtype=np.float32)
    cast_depth = np.asarray(arrays[depth_field], dtype=np.float32).reshape(-1)
    cast_azimuth = np.asarray(arrays[azimuth_field], dtype=np.float32).reshape(-1)
    warnings = _validate_cast_zc_arrays(
        cast_zc=cast_zc,
        cast_depth=cast_depth,
        cast_azimuth=cast_azimuth,
        min_finite_ratio=min_finite_ratio,
    )
    finite_ratio = float(np.mean(np.isfinite(cast_zc)))
    report = CastZcSourceReport(
        source_file=str(source_file),
        source_field=source_field,
        depth_field=depth_field,
        azimuth_field=azimuth_field,
        shape=(int(cast_zc.shape[0]), int(cast_zc.shape[1])),
        finite_ratio=finite_ratio,
        depth_count=int(cast_depth.size),
        azimuth_count=int(cast_azimuth.size),
        depth_min=float(np.min(cast_depth)),
        depth_max=float(np.max(cast_depth)),
        azimuth_min=float(np.min(cast_azimuth)),
        azimuth_max=float(np.max(cast_azimuth)),
        warnings=warnings,
    )
    return CastZcSource(
        cast_zc=cast_zc,
        cast_depth=cast_depth,
        cast_azimuth_deg=cast_azimuth,
        report=report,
    )


def _validate_cast_zc_arrays(
    *,
    cast_zc: np.ndarray,
    cast_depth: np.ndarray,
    cast_azimuth: np.ndarray,
    min_finite_ratio: float,
) -> list[str]:
    if cast_zc.ndim != 2:
        raise CastZcSourceError("Raw CAST Zc must have shape [depth, azimuth].")
    if cast_depth.ndim != 1 or cast_depth.size != cast_zc.shape[0]:
        raise CastZcSourceError(
            "CAST depth grid length must match raw Zc depth dimension."
        )
    if cast_azimuth.ndim != 1 or cast_azimuth.size != cast_zc.shape[1]:
        raise CastZcSourceError(
            "CAST azimuth grid length must match raw Zc azimuth dimension."
        )
    if not np.all(np.isfinite(cast_depth)):
        raise CastZcSourceError("CAST depth grid contains non-finite values.")
    if not np.all(np.isfinite(cast_azimuth)):
        raise CastZcSourceError("CAST azimuth grid contains non-finite values.")
    if cast_depth.size < 2:
        raise CastZcSourceError("CAST depth grid must contain at least two samples.")
    finite_ratio = float(np.mean(np.isfinite(cast_zc)))
    if finite_ratio < min_finite_ratio:
        raise CastZcSourceError(
            f"Raw CAST Zc finite ratio {finite_ratio:.3f} is below {min_finite_ratio:.3f}."
        )
    depth_diff = np.diff(cast_depth.astype(np.float64))
    if np.any(depth_diff == 0.0) or not (
        np.all(depth_diff > 0.0) or np.all(depth_diff < 0.0)
    ):
        raise CastZcSourceError("CAST depth grid must be strictly monotonic.")
    if np.any(cast_azimuth < 0.0) or np.any(cast_azimuth >= 360.0):
        raise CastZcSourceError("CAST azimuth grid must be in [0, 360).")
    if np.unique(cast_azimuth).size != cast_azimuth.size:
        raise CastZcSourceError("CAST azimuth grid contains duplicate values.")

    warnings: list[str] = []
    azimuth_step = np.diff(np.sort(cast_azimuth.astype(np.float64)))
    if azimuth_step.size and not np.allclose(azimuth_step, np.median(azimuth_step)):
        warnings.append("CAST azimuth grid is not uniformly spaced.")
    if cast_zc.shape[1] != 180:
        warnings.append("CAST azimuth count is not 180; downstream reports must note this.")
    return warnings


def _first_present(arrays: dict[str, np.ndarray], candidates: Sequence[str]) -> str | None:
    for key in candidates:
        if key in arrays:
            return key
    return None
