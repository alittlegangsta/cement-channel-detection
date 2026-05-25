from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat


def _write_raw(raw_dir: Path) -> None:
    receiver_dir = raw_dir / "XSILMR"
    receiver_dir.mkdir(parents=True)
    depth = np.arange(20, dtype=np.float64) * 0.5 + 1000.0
    savemat(
        raw_dir / "CAST.mat",
        {"CAST": {"Depth": depth, "Zc": np.ones((180, 20), dtype=np.float32)}},
        do_compression=True,
    )
    savemat(
        raw_dir / "D2_XSI_RelBearing_Inclination.mat",
        {
            "Depth_inc": depth.copy(),
            "Inc": np.linspace(5, 10, 20).astype(np.float32),
            "RelBearing": np.linspace(0, 90, 20).astype(np.float32),
        },
        do_compression=True,
    )
    for receiver in range(1, 3):
        savemat(
            receiver_dir / f"XSILMR{receiver:02d}.mat",
            {
                f"XSILMR{receiver:02d}": {
                    "Depth": depth.copy(),
                    "Tad": np.array([[10.0]], dtype=np.float32),
                }
            },
            do_compression=True,
        )


def _write_mapping(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "mapping_version: raw_variable_mapping_v001",
                "cast:",
                "  file: CAST.mat",
                "  zc_variable: CAST.Zc",
                "  depth_variable: CAST.Depth",
                "  depth_source_shape_order: [depth]",
                "  depth_unit: unknown_to_verify",
                "pose:",
                "  file: D2_XSI_RelBearing_Inclination.mat",
                "  depth_variable: Depth_inc",
                "  inclination_variable: Inc",
                "  relbearing_variable: RelBearing",
                "  source_shape_order: [depth]",
                "  depth_unit: unknown_to_verify",
                "xsi:",
                "  receiver_dir: XSILMR",
                "  expected_receiver_files: 2",
                "  depth_variable_pattern: XSILMR{receiver:02d}.Depth",
                "  depth_source_shape_order: [depth]",
                "  depth_unit: unknown_to_verify",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_audit_depth_axes_cli_outputs_reports(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    reports_dir = tmp_path / "reports"
    raw_dir.mkdir()
    reports_dir.mkdir()
    _write_raw(raw_dir)
    mapping = tmp_path / "raw_variable_mapping.yaml"
    paths_config = tmp_path / "paths.yaml"
    output_md = reports_dir / "depth_axis_audit_report.md"
    output_json = reports_dir / "depth_axis_audit_report.json"
    _write_mapping(mapping)
    paths_config.write_text(
        "\n".join(["data:", f"  raw: {raw_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/03a_audit_depth_axes.py",
            "--paths",
            str(paths_config),
            "--mapping",
            str(mapping),
            "--output-report-md",
            str(output_md),
            "--output-report-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Depth axis audit decision=conditional_go" in result.stdout
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["decision"] == "conditional_go"
    assert report["cast_depth"]["length"] == 20
    assert report["receiver_consistency"]["receiver_count"] == 2
    assert output_md.exists()
