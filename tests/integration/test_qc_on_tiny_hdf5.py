from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def _write_tiny_hdf5(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("/aligned/xsi_waveform", data=np.ones((3, 13, 8, 32), dtype=np.float32))
        h5.create_dataset("/aligned/cast_zc", data=np.ones((3, 180), dtype=np.float32))
        h5.create_dataset("/pose/inc_deg", data=np.array([10.0, 11.0, 12.0], dtype=np.float32))
        h5.create_dataset(
            "/pose/rel_bearing_deg",
            data=np.array([100.0, 101.0, 102.0], dtype=np.float32),
        )


def test_qc_on_tiny_hdf5_outputs_summary(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "tiny.h5"
    output_dir = tmp_path / "qc_mvp1"
    paths_config = tmp_path / "paths.yaml"
    _write_tiny_hdf5(hdf5_path)
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  reports: {tmp_path}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/02_run_qc.py",
            "--paths",
            str(paths_config),
            "--input-hdf5",
            str(hdf5_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Initial QC" in result.stdout
    summary_json = output_dir / "qc_summary_v001.json"
    summary_md = output_dir / "qc_summary_v001.md"
    assert summary_json.exists()
    assert summary_md.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["status"] in {"passed", "passed_with_warnings"}
    assert summary["results"]["xsi_waveform"]["shape"] == [3, 13, 8, 32]
    assert summary["errors"] == []
