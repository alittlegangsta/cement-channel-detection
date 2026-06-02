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
    np.savez_compressed(
        interim / "cast_label_input_v001.npz",
        cast_depth=np.asarray([0.0, 1.0], dtype=np.float32),
        cast_azimuth_deg=np.asarray([0.0, 120.0, 240.0], dtype=np.float32),
        cast_zc=np.asarray(
            [
                [-1.0, 1.0, 3.0],
                [1.0, 3.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )
    np.savez_compressed(
        interim / "geometry_aware_regression_labels_v001.npz",
        depth=np.asarray([0.0, 1.0], dtype=np.float32),
        geometry_kernel=np.asarray(["r7_reference_point"]),
        receiver_index=np.asarray([1], dtype=np.int16),
        weighted_channel_fraction_zc_lt_2p5=np.asarray(
            [[[2.0 / 3.0], [1.0 / 3.0]]],
            dtype=np.float32,
        ),
        interval_min_depth=np.asarray([[[0.0], [1.0]]], dtype=np.float32),
        interval_max_depth=np.asarray([[[0.0], [1.0]]], dtype=np.float32),
        midpoint_depth=np.asarray([[[0.0], [1.0]]], dtype=np.float32),
        total_cell_count=np.asarray([[[3], [3]]], dtype=np.int32),
    )


def test_cast_zc_physical_qc_cli_writes_reports(tmp_path: Path) -> None:
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
            "scripts/06an_audit_cast_zc_physical_qc.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "negative_significant=True" in result.stdout
    output_json = root_dir / "reports" / "cast_zc_physical_qc_v001.json"
    output_md = root_dir / "reports" / "cast_zc_physical_qc_v001.md"
    output_csv = root_dir / "reports" / "cast_zc_physical_qc_v001.csv"
    assert output_json.exists()
    assert output_md.exists()
    assert output_csv.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["global_cell_counts"]["zc_lt_0_count"] == 1
    assert report["no_formal_mask_applied"] is True
