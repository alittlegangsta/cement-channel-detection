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
    depth = np.arange(12, dtype=np.float64) + 100.0
    savemat(
        raw_dir / "CAST.mat",
        {"CAST": {"Depth": depth, "Zc": np.ones((180, 12), dtype=np.float32)}},
        do_compression=True,
    )
    savemat(
        raw_dir / "D2_XSI_RelBearing_Inclination.mat",
        {
            "Depth_inc": depth.copy(),
            "Inc": np.linspace(1, 6, 12).astype(np.float32),
            "RelBearing": np.linspace(10, 30, 12).astype(np.float32),
        },
        do_compression=True,
    )
    for receiver in range(1, 3):
        savemat(
            receiver_dir / f"XSILMR{receiver:02d}.mat",
            {f"XSILMR{receiver:02d}": {"Depth": depth.copy()}},
            do_compression=True,
        )


def _write_mapping(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "mapping_version: raw_variable_mapping_v001",
                "cast:",
                "  file: CAST.mat",
                "  depth_variable: CAST.Depth",
                "  depth_source_shape_order: [depth]",
                "pose:",
                "  file: D2_XSI_RelBearing_Inclination.mat",
                "  depth_variable: Depth_inc",
                "  inclination_variable: Inc",
                "  relbearing_variable: RelBearing",
                "  source_shape_order: [depth]",
                "xsi:",
                "  receiver_dir: XSILMR",
                "  expected_receiver_files: 2",
                "  depth_variable_pattern: XSILMR{receiver:02d}.Depth",
                "  depth_source_shape_order: [depth]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_read_depth_only_cli_outputs_npz_and_summary(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    raw_dir.mkdir()
    interim_dir.mkdir()
    _write_raw(raw_dir)
    mapping = tmp_path / "raw_variable_mapping.yaml"
    paths_config = tmp_path / "paths.yaml"
    output_npz = interim_dir / "depth_only_v001.npz"
    output_summary = interim_dir / "depth_only_summary_v001.json"
    _write_mapping(mapping)
    paths_config.write_text(
        "\n".join(["data:", f"  raw: {raw_dir}", f"  interim: {interim_dir}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/03c_read_depth_only.py",
            "--paths",
            str(paths_config),
            "--mapping",
            str(mapping),
            "--output-npz",
            str(output_npz),
            "--output-summary-json",
            str(output_summary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Depth-only read arrays=5" in result.stdout
    summary = json.loads(output_summary.read_text(encoding="utf-8"))
    assert summary["arrays"]["inc_deg"]["shape"] == [12]
    assert not summary["errors"]
    with np.load(output_npz) as data:
        assert data["cast_depth"].shape == (12,)
        assert data["xsi_depth_by_receiver"].shape == (2, 12)
        assert data["inc_deg"].shape == (12,)
        assert data["relbearing_deg"].shape == (12,)
