from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_inputs(reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True)
    reports_dir.joinpath("geometry_aware_regression_labels_report_v001.json").write_text(
        json.dumps(
            {
                "raw_zc_available": True,
                "raw_zc_source": {
                    "source_file": "cast_label_input_v001.npz",
                    "source_field": "cast_zc",
                    "finite_ratio": 0.99,
                    "shape": [20, 180],
                },
                "geometry_sign_status": {
                    "depth_axis_sign": -1,
                    "sign_convention_status": "human_confirmed",
                    "relbearing_plus_minus_independent": True,
                },
                "kernel_summaries": [
                    {
                        "geometry_kernel": "r7_reference_point",
                        "zero_fraction": 0.45,
                        "nonzero_fraction": 0.55,
                        "distribution": {"median": 0.02, "p95": 0.20},
                    },
                    {
                        "geometry_kernel": "midpoint_window",
                        "zero_fraction": 0.35,
                        "nonzero_fraction": 0.65,
                        "distribution": {"median": 0.03, "p95": 0.28},
                    },
                    {
                        "geometry_kernel": "uniform_source_receiver_interval",
                        "zero_fraction": 0.0,
                        "nonzero_fraction": 1.0,
                        "distribution": {"median": 0.30, "p95": 0.30},
                    },
                    {
                        "geometry_kernel": "triangular_midpoint_weighted",
                        "zero_fraction": 0.30,
                        "nonzero_fraction": 0.70,
                        "distribution": {"median": 0.04, "p95": 0.35},
                    },
                ],
                "no_final_labels": True,
            }
        ),
        encoding="utf-8",
    )
    reports_dir.joinpath("geometry_regression_audit_v001.json").write_text(
        json.dumps(
            {
                "recommendation": "human_review_primary_candidate",
                "best_kernel": "triangular_midpoint_weighted",
                "warnings": [],
                "errors": [],
                "no_final_labels": True,
                "kernel_summaries": [
                    {
                        "geometry_kernel": "r7_reference_point",
                        "status": "runnable",
                        "real_minus_permutation_margin": 0.08,
                        "depends_on_5700_band": False,
                        "sample_support_collapse": False,
                        "target_distribution": {"median": 0.02, "p90": 0.18},
                    },
                    {
                        "geometry_kernel": "midpoint_window",
                        "status": "runnable",
                        "real_minus_permutation_margin": 0.10,
                        "depends_on_5700_band": False,
                        "sample_support_collapse": False,
                        "target_distribution": {"median": 0.03, "p90": 0.26},
                    },
                    {
                        "geometry_kernel": "uniform_source_receiver_interval",
                        "status": "runnable",
                        "real_minus_permutation_margin": 0.02,
                        "depends_on_5700_band": False,
                        "sample_support_collapse": False,
                        "target_distribution": {"median": 0.30, "p90": 0.30},
                    },
                    {
                        "geometry_kernel": "triangular_midpoint_weighted",
                        "status": "runnable",
                        "real_minus_permutation_margin": 0.14,
                        "depends_on_5700_band": False,
                        "sample_support_collapse": False,
                        "target_distribution": {"median": 0.04, "p90": 0.32},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    review_dir = reports_dir / "geometry_regression_manual_review_v001"
    review_dir.mkdir()
    review_dir.joinpath("geometry_regression_review_summary_v001.json").write_text(
        json.dumps(
            {
                "primary_kernel": "triangular_midpoint_weighted",
                "interval_count": 2,
                "figure_count": 10,
                "no_final_labels": True,
            }
        ),
        encoding="utf-8",
    )
    review_dir.joinpath("selected_interval_review_list.json").write_text(
        json.dumps(
            [
                {
                    "review_id": "high_fraction_interval_0001",
                    "review_type": "high_fraction_interval",
                    "target_fraction": 0.4,
                },
                {
                    "review_id": "kernel_sensitive_interval_0002",
                    "review_type": "kernel_sensitive_interval",
                    "target_fraction": 0.2,
                },
            ]
        ),
        encoding="utf-8",
    )


def test_geometry_regression_gate_cli_writes_conditional_go(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    _write_inputs(reports_dir)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(["data:", f"  root: {root_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06ak_generate_geometry_regression_gate.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "decision=conditional_go" in result.stdout
    output_json = reports_dir / "geometry_regression_gate_report.json"
    output_md = reports_dir / "geometry_regression_gate_report.md"
    assert output_json.exists()
    assert output_md.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["decision"] == "conditional_go"
    assert report["answers"]["human_confirmed_geometry_sign_fixed_minus_one"] is True
    assert report["answers"]["raw_zc_correctly_controlled"] is True
    assert report["answers"]["most_reasonable_kernel"] == "triangular_midpoint_weighted"
    assert "uniform_source_receiver_interval" in report["answers"]["audit_only_kernels"]
    assert report["answers"]["continuous_target_healthier_than_old_binary"] is True
    assert report["answers"]["manual_review_required"] is True
    assert report["mvp4c_allowed"] is False
    assert report["stc_allowed"] is False
    assert report["apes_allowed"] is False
    assert report["deep_learning_allowed"] is False
    assert report["final_labels_allowed"] is False
    assert report["production_claims_allowed"] is False
    assert report["next_step_must_wait_human_review"] is True
