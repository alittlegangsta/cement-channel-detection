from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

AlignmentMode = Literal[
    "r7_reference_depth",
    "receiver_depth_shifted",
    "source_receiver_midpoint",
    "source_receiver_interval",
]
DepthAxisSign = Literal[1, -1]

DEFAULT_ALIGNMENT_MODES: tuple[AlignmentMode, ...] = (
    "r7_reference_depth",
    "receiver_depth_shifted",
    "source_receiver_midpoint",
    "source_receiver_interval",
)


@dataclass(frozen=True)
class ReceiverGeometry:
    """Sign-aware XSI source/receiver geometry relative to the R7 reference line."""

    reference_receiver_index: int = 7
    receiver_count: int = 13
    receiver_spacing_ft: float = 0.5
    r1_source_distance_ft: float = 1.0
    receiver_offsets_relative_to_reference_ft: tuple[float, ...] | None = None
    source_offset_relative_to_reference_ft: float | None = None
    depth_axis_sign: int | str = "audit_both"
    sign_convention_status: str = "requires_audit"
    alignment_modes: tuple[AlignmentMode, ...] = DEFAULT_ALIGNMENT_MODES

    def __post_init__(self) -> None:
        offsets = self.receiver_offsets_relative_to_reference_ft
        if offsets is None:
            offsets = tuple(
                (index + 1 - self.reference_receiver_index) * self.receiver_spacing_ft
                for index in range(self.receiver_count)
            )
            object.__setattr__(self, "receiver_offsets_relative_to_reference_ft", offsets)
        if len(offsets) != self.receiver_count:
            raise ValueError(
                "receiver_offsets_relative_to_reference_ft length must match receiver_count."
            )
        if not np.isclose(offsets[self.reference_receiver_index - 1], 0.0):
            raise ValueError("Reference receiver offset must be zero.")
        if self.source_offset_relative_to_reference_ft is None:
            source_offset = offsets[0] - self.r1_source_distance_ft
            object.__setattr__(self, "source_offset_relative_to_reference_ft", source_offset)
        if self.depth_axis_sign not in (1, -1, "audit_both"):
            raise ValueError("depth_axis_sign must be +1, -1, or 'audit_both'.")
        invalid_modes = [
            mode for mode in self.alignment_modes if mode not in DEFAULT_ALIGNMENT_MODES
        ]
        if invalid_modes:
            raise ValueError("Unsupported alignment mode(s): " + ", ".join(invalid_modes))

    @property
    def receiver_labels(self) -> tuple[str, ...]:
        return tuple(f"R{index}" for index in range(1, self.receiver_count + 1))

    @property
    def receiver_offsets_ft(self) -> np.ndarray:
        return np.asarray(self.receiver_offsets_relative_to_reference_ft, dtype=np.float32)

    @property
    def source_offset_ft(self) -> float:
        return float(self.source_offset_relative_to_reference_ft)

    @property
    def audit_signs(self) -> tuple[DepthAxisSign, ...]:
        if self.depth_axis_sign == "audit_both":
            return (1, -1)
        return (1 if int(self.depth_axis_sign) > 0 else -1,)

    def reference_depths(self, reference_depth: np.ndarray | float) -> np.ndarray:
        return np.asarray(reference_depth, dtype=np.float32).reshape(-1)

    def source_depths(
        self,
        reference_depth: np.ndarray | float,
        *,
        sign: DepthAxisSign,
    ) -> np.ndarray:
        return self.reference_depths(reference_depth) + float(sign) * self.source_offset_ft

    def receiver_depths(
        self,
        reference_depth: np.ndarray | float,
        *,
        sign: DepthAxisSign,
    ) -> np.ndarray:
        reference = self.reference_depths(reference_depth)
        return reference[:, None] + float(sign) * self.receiver_offsets_ft[None, :]

    def source_receiver_midpoints(
        self,
        reference_depth: np.ndarray | float,
        *,
        sign: DepthAxisSign,
    ) -> np.ndarray:
        source = self.source_depths(reference_depth, sign=sign)[:, None]
        receivers = self.receiver_depths(reference_depth, sign=sign)
        return ((source + receivers) * 0.5).astype(np.float32)

    def source_receiver_interval_bounds(
        self,
        reference_depth: np.ndarray | float,
        *,
        sign: DepthAxisSign,
        padding_ft: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        source = self.source_depths(reference_depth, sign=sign)[:, None]
        receivers = self.receiver_depths(reference_depth, sign=sign)
        lower = np.minimum(source, receivers) - float(padding_ft)
        upper = np.maximum(source, receivers) + float(padding_ft)
        return lower.astype(np.float32), upper.astype(np.float32)

    def target_depths_for_mode(
        self,
        reference_depth: np.ndarray | float,
        *,
        mode: AlignmentMode,
        sign: DepthAxisSign,
    ) -> np.ndarray:
        reference = self.reference_depths(reference_depth)
        if mode == "r7_reference_depth":
            return reference[:, None].repeat(self.receiver_count, axis=1).astype(np.float32)
        if mode == "receiver_depth_shifted":
            return self.receiver_depths(reference, sign=sign)
        if mode == "source_receiver_midpoint":
            return self.source_receiver_midpoints(reference, sign=sign)
        if mode == "source_receiver_interval":
            lower, upper = self.source_receiver_interval_bounds(reference, sign=sign)
            return ((lower + upper) * 0.5).astype(np.float32)
        raise ValueError(f"Unsupported alignment mode: {mode}")

    def geometry_table(
        self,
        reference_depth: np.ndarray | float,
        *,
        sign: DepthAxisSign,
        interval_padding_ft: float = 0.0,
    ) -> dict[str, np.ndarray]:
        reference = self.reference_depths(reference_depth)
        source = self.source_depths(reference, sign=sign)
        receivers = self.receiver_depths(reference, sign=sign)
        midpoints = self.source_receiver_midpoints(reference, sign=sign)
        lower, upper = self.source_receiver_interval_bounds(
            reference,
            sign=sign,
            padding_ft=interval_padding_ft,
        )
        return {
            "reference_depth": reference.astype(np.float32),
            "source_depth": source.astype(np.float32),
            "receiver_depths": receivers.astype(np.float32),
            "midpoint_depths": midpoints.astype(np.float32),
            "interval_min_depths": lower.astype(np.float32),
            "interval_max_depths": upper.astype(np.float32),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ReceiverGeometry:
        geometry = config.get("xsi_geometry", config)
        offsets_map = geometry.get("receiver_offsets_relative_to_r7_ft") or geometry.get(
            "receiver_offsets_from_reference_ft"
        )
        offsets = None
        if isinstance(offsets_map, dict):
            receiver_count = int(geometry.get("receiver_count", 13))
            offsets = tuple(
                float(offsets_map[f"R{index}"]) for index in range(1, receiver_count + 1)
            )
        modes = tuple(
            str(mode) for mode in geometry.get("alignment_modes", DEFAULT_ALIGNMENT_MODES)
        )
        return cls(
            reference_receiver_index=int(geometry.get("reference_receiver_index", 7)),
            receiver_count=int(geometry.get("receiver_count", 13)),
            receiver_spacing_ft=float(geometry.get("receiver_spacing_ft", 0.5)),
            r1_source_distance_ft=float(
                geometry.get(
                    "r1_source_distance_ft",
                    geometry.get("source_to_receiver1_ft", 1.0),
                )
            ),
            receiver_offsets_relative_to_reference_ft=offsets,
            source_offset_relative_to_reference_ft=float(
                geometry.get(
                    "source_offset_relative_to_r7_ft",
                    geometry.get("source_offset_relative_to_reference_ft", -4.0),
                )
            ),
            depth_axis_sign=geometry.get("depth_axis_sign", "audit_both"),
            sign_convention_status=str(
                geometry.get("sign_convention_status", "requires_audit")
            ),
            alignment_modes=modes,  # type: ignore[arg-type]
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> ReceiverGeometry:
        with Path(path).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            raise ValueError("XSI geometry config must be a mapping.")
        return cls.from_config(config)


def audit_both_signs(geometry: ReceiverGeometry | None = None) -> tuple[DepthAxisSign, ...]:
    return (geometry or ReceiverGeometry()).audit_signs
