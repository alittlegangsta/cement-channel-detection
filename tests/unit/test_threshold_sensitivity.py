from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cement_channel.labels.threshold_sensitivity import (
    run_threshold_sensitivity,
    write_threshold_sensitivity_outputs,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    input_npz = tmp_path / "cast_label_input_v001.npz"
    baseline_npz = tmp_path / "cast_zc_baseline_v001.npz"
    zc = np.full((4, 8), 10.0, dtype=np.float32)
    zc[1:3, 2:4] = 5.0
    zc[3, 0] = 0.2
    base = np.full_like(zc, 10.0)
    relative_drop = ((base - zc) / base).astype(np.float32)
    np.savez_compressed(
        input_npz,
        cast_depth=np.arange(4, dtype=np.float32) + 100.0,
        cast_azimuth_deg=np.arange(8, dtype=np.float32) * 45.0,
        cast_zc=zc,
        relbearing_deg=np.array([0.0, 45.0, 0.0, 0.0], dtype=np.float32),
        orientation_confidence=np.ones(4, dtype=np.float32),
        orientation_uncertain=np.zeros(4, dtype=bool),
    )
    np.savez_compressed(
        baseline_npz,
        zc_base=base,
        relative_drop=relative_drop,
        zc_ratio=(zc / base).astype(np.float32),
        baseline_valid=np.ones_like(zc, dtype=bool),
        finite_fraction=np.ones_like(zc, dtype=np.float32),
    )
    return input_npz, baseline_npz


def _label_config() -> dict:
    return {
        "threshold": {
            "relative_drop_alpha": 0.35,
            "zc_min_limit": "TODO_CONFIRM",
            "conservative_fallback_mrayl": 2.5,
            "candidate_coverage_warning_max": 1.0,
        },
        "severity": {
            "mild_min_drop": 0.30,
            "moderate_min_drop": 0.45,
            "severe_min_drop": 0.60,
        },
        "confidence": {
            "relative_drop_full_confidence": 0.70,
            "orientation_floor": 0.05,
            "bad_data_confidence": 0.0,
        },
        "bad_data": {
            "extreme_relative_drop_threshold": 0.95,
            "isolated_extreme_max_pixels": 3,
        },
        "audit": {
            "isolated_object_max_pixels": 2,
        },
    }


def test_threshold_sensitivity_computes_grid_metrics(tmp_path: Path) -> None:
    input_npz, baseline_npz = _write_inputs(tmp_path)

    report = run_threshold_sensitivity(
        cast_label_input_npz=input_npz,
        cast_baseline_npz=baseline_npz,
        label_config=_label_config(),
        alpha_grid=[0.30, 0.40],
        zc_min_limit_grid=[2.5],
        severity_threshold_sets={"default": [0.30, 0.45, 0.60]},
    )

    assert report.errors == []
    assert report.no_final_labels
    assert len(report.results) == 2
    first = report.results[0]
    assert first["plus_coverage"] is not None
    assert first["relative_drop_only_coverage"] > 0.0
    assert first["relative_drop_outlier_fraction"] > 0.0
    assert first["invalid_bad_zc_fraction"] > 0.0
    assert first["connected_component_count"] >= 1
    assert "severe" in first["severity_distribution"]


def test_threshold_sensitivity_writes_markdown_json_csv(tmp_path: Path) -> None:
    input_npz, baseline_npz = _write_inputs(tmp_path)
    report = run_threshold_sensitivity(
        cast_label_input_npz=input_npz,
        cast_baseline_npz=baseline_npz,
        label_config=_label_config(),
        alpha_grid=[0.35],
        zc_min_limit_grid=[2.5],
        severity_threshold_sets={"default": [0.30, 0.45, 0.60]},
    )

    write_threshold_sensitivity_outputs(
        report,
        output_report_md=tmp_path / "label_threshold_sensitivity_v001.md",
        output_report_json=tmp_path / "label_threshold_sensitivity_v001.json",
        output_report_csv=tmp_path / "label_threshold_sensitivity_v001.csv",
        overwrite=False,
    )

    assert "Threshold Sensitivity" in (tmp_path / "label_threshold_sensitivity_v001.md").read_text(
        encoding="utf-8"
    )
    data = json.loads((tmp_path / "label_threshold_sensitivity_v001.json").read_text())
    assert data["no_final_labels"] is True
    assert "alpha,zc_min_limit,severity_set" in (
        tmp_path / "label_threshold_sensitivity_v001.csv"
    ).read_text(encoding="utf-8")
