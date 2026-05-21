from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cement_channel.labels.cast_weak_label import generate_cast_weak_labels


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    input_npz = tmp_path / "cast_label_input_v001.npz"
    baseline_npz = tmp_path / "cast_zc_baseline_v001.npz"
    zc = np.full((3, 8), 10.0, dtype=np.float32)
    zc[1, 0] = 4.0
    base = np.full_like(zc, 10.0)
    relative_drop = (base - zc) / base
    np.savez_compressed(
        input_npz,
        cast_depth=np.array([100.0, 101.0, 102.0], dtype=np.float32),
        cast_azimuth_deg=np.arange(8, dtype=np.float32) * 45.0,
        cast_zc=zc,
        relbearing_deg=np.array([0.0, 90.0, 0.0], dtype=np.float32),
        orientation_confidence=np.array([1.0, 0.5, 1.0], dtype=np.float32),
        orientation_uncertain=np.array([False, True, False]),
    )
    np.savez_compressed(
        baseline_npz,
        zc_base=base,
        relative_drop=relative_drop.astype(np.float32),
        zc_ratio=(zc / base).astype(np.float32),
        baseline_valid=np.ones_like(zc, dtype=bool),
    )
    return input_npz, baseline_npz


def _label_config() -> dict:
    return {
        "threshold": {
            "relative_drop_alpha": 0.35,
            "zc_min_limit": "TODO_CONFIRM",
            "conservative_fallback_mrayl": 2.5,
            "require_confirmed_zc_min_limit": False,
            "candidate_coverage_warning_min": 0.001,
            "candidate_coverage_warning_max": 0.50,
            "candidate_coverage_blocking_min": 0.000001,
            "candidate_coverage_blocking_max": 0.80,
        },
        "severity": {
            "mild_min_drop": 0.30,
            "moderate_min_drop": 0.45,
            "severe_min_drop": 0.60,
        },
        "confidence": {
            "relative_drop_full_confidence": 0.70,
            "orientation_floor": 0.05,
        },
    }


def test_generate_cast_weak_labels_outputs_plus_and_minus_candidates(tmp_path: Path) -> None:
    input_npz, baseline_npz = _write_inputs(tmp_path)

    report, arrays = generate_cast_weak_labels(
        cast_label_input_npz=input_npz,
        cast_baseline_npz=baseline_npz,
        label_config=_label_config(),
    )

    assert report.errors == []
    assert report.no_final_labels
    assert arrays["presence_plus"][1, 2] == 1
    assert arrays["presence_minus_ablation"][1, 6] == 1
    assert arrays["severity_plus"][1, 2] == 3
    assert 0.0 < arrays["label_confidence_plus"][1, 2] < 0.1
    assert report.coverage["plus"] == report.coverage["minus_ablation"]
    assert "conservative fallback" in report.warnings[0]


def test_generate_cast_weak_labels_metadata_says_no_final_labels(tmp_path: Path) -> None:
    input_npz, baseline_npz = _write_inputs(tmp_path)

    _report, arrays = generate_cast_weak_labels(
        cast_label_input_npz=input_npz,
        cast_baseline_npz=baseline_npz,
        label_config=_label_config(),
    )

    metadata = json.loads(str(arrays["metadata_json"]))
    assert metadata["no_final_labels"] is True
    assert metadata["plus"]["label_source"] == "cast_weak_plus"
    assert metadata["minus_ablation"]["label_source"] == "cast_weak_minus_ablation"
    assert metadata["convention_status"] == "specification_preferred_plus_data_unresolved"
