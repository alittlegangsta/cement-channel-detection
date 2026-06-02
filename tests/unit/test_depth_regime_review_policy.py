from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cement_channel.evaluation.depth_regime_review_policy import (
    DepthRegimeReviewPolicyError,
    load_depth_regime_review_policy,
    parse_depth_regime_review_policy,
)


def test_depth_regime_review_policy_example_is_review_only() -> None:
    policy = load_depth_regime_review_policy(
        "configs/geometry_regression_depth_regime_review.example.yaml"
    )

    assert policy.status == "human_approved_for_review_only"
    assert policy.formal_cv_split_change_approved is False
    assert policy.regime_specific_modeling_approved is False
    assert policy.final_labels_approved is False
    assert policy.boundary_review_half_width_ft == 15.0
    assert [regime.id for regime in policy.broad_review_regimes] == [
        "regime_a",
        "regime_b",
        "regime_c",
    ]
    assert policy.target_view_policy["receiver_p90"] == (
        "robust_candidate_for_human_review"
    )
    assert policy.morphology_policy["morphology_target_redesign_approved"] is False
    zone = policy.boundary_zone_for_depth(2583.0)
    assert zone is not None
    assert zone.priority == "primary"
    assert zone.depth_min_ft < 2582.79 < zone.depth_max_ft


def test_depth_regime_review_policy_rejects_cv_split_approval(tmp_path: Path) -> None:
    config = yaml.safe_load(
        Path("configs/geometry_regression_depth_regime_review.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["review_policy"]["formal_cv_split_change_approved"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(DepthRegimeReviewPolicyError, match="CV split"):
        load_depth_regime_review_policy(path)


def test_depth_regime_review_policy_rejects_nonconservative_boundary_width() -> None:
    config = yaml.safe_load(
        Path("configs/geometry_regression_depth_regime_review.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["boundary_review_half_width_ft"] = 30.0

    with pytest.raises(DepthRegimeReviewPolicyError, match="half-width"):
        parse_depth_regime_review_policy(config)

