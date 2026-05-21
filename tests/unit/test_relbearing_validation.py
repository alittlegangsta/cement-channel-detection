from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cement_channel.alignment.relbearing_validation import (
    relbearing_config_dict,
    validate_relbearing_sign,
)


def _write_inputs(tmp_path: Path, *, cast_preview: np.ndarray | None = None) -> dict[str, Path]:
    preview_npz = tmp_path / "depth_resample_preview_v001.npz"
    small_summary = tmp_path / "small_slice_summary_v001.json"
    resample_report = tmp_path / "depth_resample_preview_report.json"
    np.savez_compressed(
        preview_npz,
        canonical_depth=np.array([100.0, 101.0, 102.0], dtype=np.float32),
        inc_deg_on_grid=np.array([0.5, 3.0, 6.0], dtype=np.float32),
        relbearing_deg_on_grid=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        small_slice_cast_zc_on_preview=(
            cast_preview if cast_preview is not None else np.empty((0, 0), dtype=np.float32)
        ),
    )
    small_summary.write_text(json.dumps({"warnings": []}), encoding="utf-8")
    resample_report.write_text(
        json.dumps(
            {"small_slice": {"status": "skipped_no_common_overlap"}, "warnings": [], "errors": []}
        ),
        encoding="utf-8",
    )
    return {
        "preview": preview_npz,
        "small_summary": small_summary,
        "resample_report": resample_report,
    }


def test_validate_relbearing_sign_reports_insufficient_evidence(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    report = validate_relbearing_sign(
        depth_resample_preview_npz=paths["preview"],
        small_slice_summary_json=paths["small_summary"],
        depth_resample_report_json=paths["resample_report"],
    )

    assert report.decision == "insufficient_evidence"
    assert report.selected_convention is None
    assert report.manual_confirmation_required is True
    assert report.mvp3_allowed_without_confirmation is False
    assert (
        report.convention_conclusion["relbearing_sign_status"]
        == "specification_preferred_plus_data_unresolved"
    )
    assert report.convention_conclusion["documentation_preferred_sign"] == "plus"
    assert report.convention_conclusion["primary_convention"] == "plus"
    assert report.convention_conclusion["ablation_convention"] == "minus"
    assert report.convention_conclusion["single_sign_alignment_approved"] is False
    assert (
        report.convention_conclusion["single_sign_alignment_approval_basis"]
        == "specification_only_not_data_confirmed"
    )
    assert all(metric.wrap_valid for metric in report.candidate_metrics.values())


def test_relbearing_config_requires_confirmation(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    report = validate_relbearing_sign(
        depth_resample_preview_npz=paths["preview"],
        small_slice_summary_json=paths["small_summary"],
        depth_resample_report_json=paths["resample_report"],
    )

    config = relbearing_config_dict(report)

    assert config["status"] == "specification_preferred_primary_data_unresolved"
    assert config["selected_convention"] == "plus_primary_minus_ablation"
    assert config["relbearing_sign_status"] == "specification_preferred_plus_data_unresolved"
    assert config["primary_convention"] == "plus"
    assert config["ablation_convention"] == "minus"
    assert config["documentation_preferred_sign"] == "plus"
    assert config["data_driven_validation"] == "insufficient_evidence"
    assert config["single_sign_alignment_approved"] is False
    assert config["approved_downstream_mode"] == "plus_primary_minus_ablation"
    assert config["manual_confirmations"]["side_a_offset_deg"] == 0.0
    assert config["manual_confirmations"]["xsi_side_order"] == "clockwise"
    assert config["manual_confirmations"]["cast_azimuth_direction"] == "normal"
    assert config["manual_confirmation_required"] is True
    assert (
        config["mvp3_allowed_without_data_confirmation"]
        == "plus_primary_minus_ablation_only"
    )
