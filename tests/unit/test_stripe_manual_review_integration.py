from __future__ import annotations

from cement_channel.qc.stripe_manual_review_integration import (
    StripeManualReviewConfig,
    augment_selected_interval,
    max_target_sensitivity,
)


def _event() -> dict[str, object]:
    return {
        "event_id": "cast_hstripe_0001",
        "event_family": "near_360_high_zc",
        "event_class": "multi-row band",
        "stripe_depth_ft": 100.5,
        "depth_min_ft": 100.0,
        "depth_max_ft": 101.0,
        "azimuth_coverage": 0.98,
        "row_count": 7,
    }


def test_stripe_manual_review_action_rules() -> None:
    cfg = StripeManualReviewConfig(heatmap_half_window_ft=20.0, near_distance_ft=3.0)

    crossing = augment_selected_interval(
        {"depth": "100.5"},
        stripe_events=[_event()],
        sensitivity_max=0.4,
        config=cfg,
    )
    near = augment_selected_interval(
        {"depth": "103.0"},
        stripe_events=[_event()],
        sensitivity_max=0.4,
        config=cfg,
    )
    window = augment_selected_interval(
        {"depth": "115.0"},
        stripe_events=[_event()],
        sensitivity_max=0.4,
        config=cfg,
    )
    far = augment_selected_interval(
        {"depth": "150.0"},
        stripe_events=[_event()],
        sensitivity_max=0.4,
        config=cfg,
    )

    assert crossing["stripe_crosses_review_depth"] is True
    assert crossing["recommended_review_action"] == "defer_pending_horizontal_stripe_qc"
    assert near["stripe_near_review_depth"] is True
    assert near["recommended_review_action"] == "defer_pending_horizontal_stripe_qc"
    assert window["stripe_in_heatmap_window"] is True
    assert window["recommended_review_action"] == "review_with_qc_note"
    assert far["recommended_review_action"] == "review_now"


def test_max_target_sensitivity_ignores_keep_all_and_missing_contract_rows() -> None:
    rows = [
        {
            "target": "receiver_mean_v1_reference",
            "scenario": "keep_all_rows",
            "recompute_status": "recomputed_audit_only",
            "max_abs_delta_vs_keep_all": "0.0",
        },
        {
            "target": "receiver_mean_v1_reference",
            "scenario": "exclude_stripe_rows_pm1",
            "recompute_status": "recomputed_audit_only",
            "max_abs_delta_vs_keep_all": "0.12",
        },
        {
            "target": "connected_channel_fraction",
            "scenario": "keep_all_rows",
            "recompute_status": "skipped_missing_recompute_contract",
            "max_abs_delta_vs_keep_all": "",
        },
    ]

    assert max_target_sensitivity(rows) == 0.12
