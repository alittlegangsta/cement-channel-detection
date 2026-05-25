from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _sample_report(
    *,
    high_conf_candidates: int = 25,
    high_conf_non_candidates: int = 25,
    positive_weight_fraction: float = 0.5,
    sample_weight_max: float = 0.9,
) -> dict:
    return {
        "sample_table_version": "baseline_sample_table_v001",
        "shape": {"samples": 80, "features": 6, "transformed_features": 12},
        "counts": {
            "total_samples": 80,
            "candidate_count": 40,
            "non_candidate_count": 40,
            "high_confidence_candidate_count": high_conf_candidates,
            "high_confidence_non_candidate_count": high_conf_non_candidates,
            "positive_sample_weight_count": int(80 * positive_weight_fraction),
            "positive_sample_weight_fraction": positive_weight_fraction,
            "plus_minus_disagreement_fraction": 0.05,
        },
        "excluded_counts": {
            "exclude_nonfinite_feature": 0,
            "exclude_large_depth_match_error": 0,
            "zero_sample_weight": 40,
        },
        "transformed_feature_ranges": {
            "log1p_rms_energy": {
                "finite_ratio": 1.0,
                "min": 0.0,
                "max": 2.0,
                "mean": 1.0,
                "median": 1.0,
            },
            "robust_scaled_rms_energy": {
                "finite_ratio": 1.0,
                "min": -2.0,
                "max": 2.0,
                "mean": 0.0,
                "median": 0.0,
            },
        },
        "sample_weight": {
            "finite_ratio": 1.0,
            "min": 0.0,
            "max": sample_weight_max,
            "mean": 0.4,
            "median": 0.5,
        },
        "no_model_training": True,
        "no_final_labels": True,
        "warnings": [],
        "errors": [],
        "not_performed": [
            "model training",
            "train/test split",
            "production inference",
            "STC",
            "APES",
            "deep learning",
            "final label generation",
        ],
    }


def _diagnostics_report() -> dict:
    return {
        "diagnostics_version": "feature_preprocessing_diagnostics_v001",
        "figures": {
            "feature_hist_raw_vs_log": "01.png",
            "feature_hist_scaled": "02.png",
            "candidate_vs_non_candidate_by_feature": "03.png",
            "sample_weight_distribution": "04.png",
            "depth_match_error_distribution": "05.png",
        },
        "nonfinite_counts": {
            "transformed:log1p_rms_energy": {
                "total": 80,
                "nonfinite": 0,
                "finite_ratio": 1.0,
            },
            "transformed:robust_scaled_rms_energy": {
                "total": 80,
                "nonfinite": 0,
                "finite_ratio": 1.0,
            },
        },
        "sample_weight": {
            "finite_ratio": 1.0,
            "min": 0.0,
            "max": 0.9,
            "mean": 0.4,
            "median": 0.5,
        },
        "depth_match_error": {
            "finite_ratio": 1.0,
            "min": 0.0,
            "max": 0.1,
            "mean": 0.02,
            "median": 0.0,
        },
        "no_model_training": True,
        "no_final_labels": True,
        "warnings": [],
        "errors": [],
        "not_performed": [
            "model training",
            "train/test split",
            "production inference",
            "STC",
            "APES",
            "deep learning",
            "final label generation",
        ],
    }


def _mvp4a_gate_report() -> dict:
    return {
        "decision": "go",
        "mvp4b_allowed": True,
        "gate_conditions": {
            "high_confidence_subset_exists": True,
            "no_model_training": True,
            "no_final_labels": True,
        },
        "warnings": [],
        "errors": [],
    }


def _write_gate_inputs(reports_dir: Path, sample_report: dict | None = None) -> None:
    _write_json(
        reports_dir / "baseline_sample_table_report_v001.json",
        sample_report or _sample_report(),
    )
    _write_json(
        reports_dir / "feature_preprocessing_diagnostics_v001.json",
        _diagnostics_report(),
    )
    _write_json(reports_dir / "mvp4a_gate_report.json", _mvp4a_gate_report())


def test_mvp4b_stage1_gate_report_cli_allows_stage2_when_conditions_pass(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True)
    _write_gate_inputs(reports_dir)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(["data:", f"  root: {root_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06c_generate_mvp4b_stage1_gate_report.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "MVP-4B Stage 1 gate decision=go" in result.stdout
    report = json.loads(
        (reports_dir / "mvp4b_stage1_gate_report.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "go"
    assert report["mvp4b_stage2_allowed"] is True
    assert report["gate_conditions"]["no_model_training"] is True
    assert (reports_dir / "mvp4b_stage1_gate_report.md").exists()


def test_mvp4b_stage1_gate_report_cli_blocks_empty_high_confidence_class(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True)
    _write_gate_inputs(reports_dir, _sample_report(high_conf_candidates=0))
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(["data:", f"  root: {root_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06c_generate_mvp4b_stage1_gate_report.py",
            "--paths",
            str(paths_config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "decision=no_go" in result.stdout
    report = json.loads(
        (reports_dir / "mvp4b_stage1_gate_report.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "no_go"
    assert report["mvp4b_stage2_allowed"] is False
    assert any(
        "high-confidence candidate subset is empty" in issue
        for issue in report["blocking_issues"]
    )


def test_mvp4b_stage1_gate_report_cli_blocks_zero_sample_weights(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True)
    _write_gate_inputs(
        reports_dir,
        _sample_report(positive_weight_fraction=0.0, sample_weight_max=0.0),
    )
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(["data:", f"  root: {root_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06c_generate_mvp4b_stage1_gate_report.py",
            "--paths",
            str(paths_config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(
        (reports_dir / "mvp4b_stage1_gate_report.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "no_go"
    assert any(
        "sample_weight is invalid or all zero" in issue
        for issue in report["blocking_issues"]
    )
