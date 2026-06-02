from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class DepthRegimeReviewPolicyError(ValueError):
    """Raised when the formal review policy config is unsafe or incomplete."""


@dataclass(frozen=True)
class BroadReviewRegime:
    id: str
    depth_min_ft: float
    depth_max_ft: float
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundaryReviewZone:
    zone_id: str
    depth_ft: float
    priority: str
    support_count: int
    half_width_ft: float

    @property
    def depth_min_ft(self) -> float:
        return self.depth_ft - self.half_width_ft

    @property
    def depth_max_ft(self) -> float:
        return self.depth_ft + self.half_width_ft

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["depth_min_ft"] = self.depth_min_ft
        data["depth_max_ft"] = self.depth_max_ft
        data["purpose"] = "manual_review_only"
        return data


@dataclass(frozen=True)
class DepthRegimeReviewPolicy:
    schema_version: str
    config_version: str
    status: str
    formal_cv_split_change_approved: bool
    regime_specific_modeling_approved: bool
    final_labels_approved: bool
    production_claims_approved: bool
    broad_review_regimes: tuple[BroadReviewRegime, ...]
    priority_boundary_review_zones: tuple[BoundaryReviewZone, ...]
    boundary_review_half_width_ft: float
    boundary_review_half_width_policy: str
    target_view_policy: dict[str, str]
    morphology_policy: dict[str, Any]
    forbidden_automatic_actions: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_version": self.config_version,
            "review_policy": {
                "status": self.status,
                "formal_cv_split_change_approved": self.formal_cv_split_change_approved,
                "regime_specific_modeling_approved": self.regime_specific_modeling_approved,
                "final_labels_approved": self.final_labels_approved,
                "production_claims_approved": self.production_claims_approved,
            },
            "broad_review_regimes": [
                regime.to_dict() for regime in self.broad_review_regimes
            ],
            "priority_boundary_review_zones": [
                zone.to_dict() for zone in self.priority_boundary_review_zones
            ],
            "boundary_review_half_width_ft": self.boundary_review_half_width_ft,
            "boundary_review_half_width_policy": self.boundary_review_half_width_policy,
            "target_view_policy": dict(self.target_view_policy),
            "morphology_policy": dict(self.morphology_policy),
            "forbidden_automatic_actions": dict(self.forbidden_automatic_actions),
        }

    def regime_for_depth(self, depth_ft: float) -> BroadReviewRegime | None:
        for regime in self.broad_review_regimes:
            if regime.depth_min_ft <= depth_ft <= regime.depth_max_ft:
                return regime
        return None

    def boundary_zone_for_depth(self, depth_ft: float) -> BoundaryReviewZone | None:
        candidates = [
            zone
            for zone in self.priority_boundary_review_zones
            if zone.depth_min_ft <= depth_ft <= zone.depth_max_ft
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda zone: abs(depth_ft - zone.depth_ft))[0]


def load_depth_regime_review_policy(path: Path | str) -> DepthRegimeReviewPolicy:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Depth-regime review policy config does not exist: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DepthRegimeReviewPolicyError(
            f"Depth-regime review policy config must be a YAML mapping: {config_path}"
        )
    return parse_depth_regime_review_policy(data)


def parse_depth_regime_review_policy(data: dict[str, Any]) -> DepthRegimeReviewPolicy:
    review_policy = _as_dict(data.get("review_policy"))
    half_width = _as_float(
        data.get("boundary_review_half_width_ft"),
        "boundary_review_half_width_ft",
    )
    regimes = tuple(_parse_regimes(data.get("broad_review_regimes")))
    zones = tuple(_parse_zones(data.get("priority_boundary_review_zones"), half_width))
    target_view_policy = _as_dict(data.get("target_view_policy"))
    morphology_policy = _as_dict(data.get("morphology_policy"))
    forbidden = _as_dict(data.get("forbidden_automatic_actions"))

    policy = DepthRegimeReviewPolicy(
        schema_version=str(data.get("schema_version", "")),
        config_version=str(data.get("config_version", "")),
        status=str(review_policy.get("status", "")),
        formal_cv_split_change_approved=bool(
            review_policy.get("formal_cv_split_change_approved")
        ),
        regime_specific_modeling_approved=bool(
            review_policy.get("regime_specific_modeling_approved")
        ),
        final_labels_approved=bool(review_policy.get("final_labels_approved")),
        production_claims_approved=bool(review_policy.get("production_claims_approved")),
        broad_review_regimes=regimes,
        priority_boundary_review_zones=zones,
        boundary_review_half_width_ft=half_width,
        boundary_review_half_width_policy=str(
            data.get("boundary_review_half_width_policy", "")
        ),
        target_view_policy={str(key): str(value) for key, value in target_view_policy.items()},
        morphology_policy=dict(morphology_policy),
        forbidden_automatic_actions={str(key): bool(value) for key, value in forbidden.items()},
    )
    validate_depth_regime_review_policy(policy)
    return policy


def validate_depth_regime_review_policy(policy: DepthRegimeReviewPolicy) -> None:
    if policy.status != "human_approved_for_review_only":
        raise DepthRegimeReviewPolicyError(
            "Depth-regime policy must be human_approved_for_review_only."
        )
    if policy.formal_cv_split_change_approved:
        raise DepthRegimeReviewPolicyError("Formal CV split change is not approved.")
    if policy.regime_specific_modeling_approved:
        raise DepthRegimeReviewPolicyError("Regime-specific modeling is not approved.")
    if policy.final_labels_approved:
        raise DepthRegimeReviewPolicyError("Final labels are not approved.")
    if policy.production_claims_approved:
        raise DepthRegimeReviewPolicyError("Production claims are not approved.")
    if not (0.0 < policy.boundary_review_half_width_ft <= 20.0):
        raise DepthRegimeReviewPolicyError(
            "Boundary review half-width must be a conservative review-only value <= 20 ft."
        )
    if len(policy.broad_review_regimes) != 3:
        raise DepthRegimeReviewPolicyError("Exactly three broad review regimes are required.")
    if len(policy.priority_boundary_review_zones) != 7:
        raise DepthRegimeReviewPolicyError("Exactly seven boundary review zones are required.")
    _validate_regime_order(policy.broad_review_regimes)
    _validate_target_view_policy(policy.target_view_policy)
    if policy.morphology_policy.get("use_existing_arrays_for_review") is not True:
        raise DepthRegimeReviewPolicyError("Existing morphology arrays must be review-enabled.")
    if policy.morphology_policy.get("morphology_target_redesign_approved") is not False:
        raise DepthRegimeReviewPolicyError("Morphology target redesign is not approved.")
    _validate_forbidden_actions(policy.forbidden_automatic_actions)


def _parse_regimes(value: Any) -> list[BroadReviewRegime]:
    if not isinstance(value, list):
        raise DepthRegimeReviewPolicyError("broad_review_regimes must be a list.")
    regimes = []
    for row in value:
        item = _as_dict(row)
        regimes.append(
            BroadReviewRegime(
                id=str(item.get("id", "")),
                depth_min_ft=_as_float(item.get("depth_min_ft"), "regime.depth_min_ft"),
                depth_max_ft=_as_float(item.get("depth_max_ft"), "regime.depth_max_ft"),
                purpose=str(item.get("purpose", "")),
            )
        )
    return regimes


def _parse_zones(value: Any, half_width_ft: float) -> list[BoundaryReviewZone]:
    if not isinstance(value, list):
        raise DepthRegimeReviewPolicyError("priority_boundary_review_zones must be a list.")
    zones = []
    seen: set[str] = set()
    for row in value:
        item = _as_dict(row)
        depth_ft = _as_float(item.get("depth_ft"), "boundary.depth_ft")
        priority = str(item.get("priority", ""))
        zone_id = _zone_id(depth_ft)
        if zone_id in seen:
            raise DepthRegimeReviewPolicyError(f"Duplicate boundary review zone: {zone_id}")
        seen.add(zone_id)
        zones.append(
            BoundaryReviewZone(
                zone_id=zone_id,
                depth_ft=depth_ft,
                priority=priority,
                support_count=int(item.get("support_count", 0)),
                half_width_ft=half_width_ft,
            )
        )
    return zones


def _validate_regime_order(regimes: tuple[BroadReviewRegime, ...]) -> None:
    previous_max: float | None = None
    for regime in regimes:
        if regime.purpose != "descriptive_review_only":
            raise DepthRegimeReviewPolicyError(
                f"Regime {regime.id} must be descriptive_review_only."
            )
        if regime.depth_max_ft <= regime.depth_min_ft:
            raise DepthRegimeReviewPolicyError(f"Regime {regime.id} has invalid bounds.")
        if previous_max is not None and regime.depth_min_ft < previous_max:
            raise DepthRegimeReviewPolicyError("Broad review regimes must be ordered.")
        previous_max = regime.depth_max_ft


def _validate_target_view_policy(policy: dict[str, str]) -> None:
    expected = {
        "receiver_mean": "conservative_reference",
        "receiver_p90": "robust_candidate_for_human_review",
        "receiver_max": "sensitive_audit_only",
        "full_360_fraction": "auxiliary_coverage_view",
        "receiver_std": "disagreement_audit",
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise DepthRegimeReviewPolicyError(
                f"target_view_policy.{key} must be {value}."
            )


def _validate_forbidden_actions(actions: dict[str, bool]) -> None:
    required_true = {
        "formal_cv_split_change",
        "regime_specific_model_training",
        "primary_target_view_change",
        "receiver_p90_primary_adoption",
        "morphology_target_creation",
        "label_definition_change",
        "feature_addition",
        "preprocessing_change",
        "waveform_read",
        "final_label_generation",
        "mvp4c",
        "stc",
        "apes",
        "deep_learning",
    }
    missing = sorted(key for key in required_true if actions.get(key) is not True)
    if missing:
        raise DepthRegimeReviewPolicyError(
            "Forbidden automatic actions must be explicitly true: " + ", ".join(missing)
        )


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise DepthRegimeReviewPolicyError("Expected YAML mapping.")


def _as_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DepthRegimeReviewPolicyError(f"{field_name} must be numeric.") from exc
    if not parsed == parsed:
        raise DepthRegimeReviewPolicyError(f"{field_name} must be finite.")
    return parsed


def _zone_id(depth_ft: float) -> str:
    return f"boundary_{depth_ft:.2f}_ft".replace(".", "p")
