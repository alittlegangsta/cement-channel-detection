from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_inputs(root_dir: Path) -> None:
    interim = root_dir / "interim"
    reports = root_dir / "reports"
    interim.mkdir(parents=True)
    reports.mkdir()
    depth = np.arange(8, dtype=np.float32)
    kernels = np.asarray(
        [
            "r7_reference_point",
            "midpoint_window",
            "uniform_source_receiver_interval",
            "triangular_midpoint_weighted",
        ]
    )
    base = np.linspace(0.0, 0.4, depth.size, dtype=np.float32)
    primary = np.repeat(
        np.stack([base, base + 0.01, base + 0.02, base + 0.03], axis=0)[:, :, None],
        13,
        axis=2,
    )
    np.savez_compressed(
        interim / "geometry_aware_regression_labels_v001.npz",
        depth=depth,
        geometry_kernel=kernels,
        receiver_index=np.arange(1, 14, dtype=np.int16),
        weighted_channel_fraction_zc_lt_2p5=primary,
        raw_channel_fraction_zc_lt_2p5=primary,
        relative_anomaly_fraction=np.zeros_like(primary),
        combined_channel_fraction=primary,
        largest_connected_component_fraction=primary,
        max_azimuth_channel_fraction=primary,
        depth_label_confidence=np.ones_like(primary),
        candidate_cell_count=np.ones(primary.shape, dtype=np.int32),
        total_cell_count=np.ones(primary.shape, dtype=np.int32),
        receiver_mean=np.mean(primary, axis=2),
        receiver_max=np.max(primary, axis=2),
        receiver_p90=np.percentile(primary, 90.0, axis=2).astype(np.float32),
        receiver_std=np.std(primary, axis=2),
        full_360_fraction=np.max(primary, axis=2),
        derived_positive_at_fraction_0p01=np.max(primary, axis=2) >= 0.01,
        derived_positive_at_fraction_0p05=np.max(primary, axis=2) >= 0.05,
        derived_positive_at_fraction_0p10=np.max(primary, axis=2) >= 0.10,
        primary_target=np.asarray("weighted_channel_fraction_zc_lt_2p5"),
        raw_zc_source_file=np.asarray("cast_label_input_v001.npz"),
        raw_zc_source_field=np.asarray("cast_zc"),
        raw_zc_finite_ratio=np.asarray(1.0, dtype=np.float32),
        no_final_labels=np.asarray(True),
    )
    features = np.column_stack([base, base**2]).astype(np.float32)
    np.savez_compressed(
        interim / "depth_level_xsi_features_v001.npz",
        depth=depth,
        depth_level_xsi_features=features,
        depth_level_xsi_feature_names=np.asarray(
            ["near_far_ratio_mean_early_energy", "receiver_max_early_energy"]
        ),
        no_final_labels=np.asarray(True),
    )
    reports.joinpath("geometry_aware_regression_labels_report_v001.json").write_text(
        json.dumps(
            {
                "primary_target": "weighted_channel_fraction_zc_lt_2p5",
                "raw_zc_source": {"source_file": "cast_label_input_v001.npz"},
                "geometry_sign_status": {"depth_axis_sign": -1},
                "warnings": [],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    reports.joinpath("geometry_aware_regression_labels_report_v001.md").write_text(
        "old report\n",
        encoding="utf-8",
    )


def test_geometry_regression_contract_qa_cli_writes_reports(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    _write_inputs(root_dir)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  interim: {root_dir / 'interim'}",
                f"  reports: {root_dir / 'reports'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/06ai_audit_geometry_regression.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06al_generate_geometry_regression_contract_qa.py",
            "--paths",
            str(paths_config),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "invariants_passed=True" in result.stdout
    inventory = root_dir / "reports" / "geometry_regression_contract_inventory_v001.json"
    invariants = root_dir / "reports" / "geometry_regression_contract_invariants_v001.json"
    refreshed_stage9 = root_dir / "reports" / "geometry_aware_regression_labels_report_v001.json"
    assert inventory.exists()
    assert invariants.exists()
    assert json.loads(invariants.read_text(encoding="utf-8"))["passed"] is True
    assert "target_view_summaries" in json.loads(refreshed_stage9.read_text(encoding="utf-8"))
