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
    depth = np.arange(40, dtype=np.float32)
    base = np.linspace(0.0, 1.0, depth.size, dtype=np.float32)
    receiver_max = np.stack(
        [base * 0.4, base * 0.55, base * 0.3 + 0.02, base * 0.75],
        axis=0,
    ).astype(np.float32)
    np.savez_compressed(
        interim / "geometry_aware_regression_labels_v001.npz",
        depth=depth,
        geometry_kernel=np.asarray(
            [
                "r7_reference_point",
                "midpoint_window",
                "uniform_source_receiver_interval",
                "triangular_midpoint_weighted",
            ]
        ),
        receiver_max=receiver_max,
        no_final_labels=np.asarray(True),
    )
    features = np.column_stack([base, np.sin(base * np.pi)]).astype(np.float32)
    np.savez_compressed(
        interim / "depth_level_xsi_features_v001.npz",
        depth=depth,
        depth_level_xsi_features=features,
        depth_level_xsi_feature_names=np.asarray(["energy", "sine_shape"]),
        no_final_labels=np.asarray(True),
    )


def test_audit_geometry_regression_cli_writes_outputs(tmp_path: Path) -> None:
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

    result = subprocess.run(
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

    assert "Geometry regression audit" in result.stdout
    output_json = root_dir / "reports" / "geometry_regression_audit_v001.json"
    output_csv = root_dir / "reports" / "geometry_regression_audit_v001.csv"
    assert output_json.exists()
    assert output_csv.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["no_model_training"] is True
    assert report["no_model_weights"] is True
    assert report["no_final_labels"] is True
    assert len(report["kernel_summaries"]) == 4
