from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def _write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "schema_v001",
                "data_version": "data_v001",
                "created_at": "2026-05-20T00:00:00+00:00",
                "wells": [
                    {
                        "well_id": "D2",
                        "counts": {"cast": 1, "pose": 1, "xsi_receiver": 13},
                        "actual_xsi_receiver_files": 13,
                        "expected_xsi_receiver_files": 13,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_tiny_hdf5(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("/aligned/xsi_waveform", data=np.ones((3, 13, 8, 4), dtype=np.float32))
        h5.create_dataset("/aligned/cast_zc", data=np.ones((3, 180), dtype=np.float32))
        h5.create_dataset("/axis/depth", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("/axis/time_sample_index", data=np.arange(4, dtype=np.int32))
        h5.create_dataset("/axis/xsi_side_azimuth_deg", data=np.arange(8, dtype=np.float32))
        h5.create_dataset("/axis/cast_azimuth_deg", data=np.arange(180, dtype=np.float32))
        h5.create_dataset("/pose/inc_deg", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("/pose/rel_bearing_deg", data=np.arange(3, dtype=np.float32))
        meta = h5.create_group("metadata")
        dtype = h5py.string_dtype(encoding="utf-8")
        for key, value in {
            "schema_version": "schema_v001",
            "data_version": "data_v001",
            "mapping_version": "raw_variable_mapping_v001",
            "source_files": "{}",
            "created_at": "2026-05-20T00:00:00+00:00",
        }.items():
            meta.create_dataset(key, data=value, dtype=dtype)


def _write_common_inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "manifest": tmp_path / "data_manifest_v001.json",
        "metadata": tmp_path / "mat_metadata_v001.json",
        "probe": tmp_path / "mat_struct_probe_v001.json",
        "mapping": tmp_path / "raw_variable_mapping.yaml",
        "slice": tmp_path / "small_slice_summary_v001.json",
        "hdf5": tmp_path / "tiny.h5",
        "qc": tmp_path / "qc_summary_v001.json",
        "config": tmp_path / "paths.yaml",
    }
    _write_manifest(paths["manifest"])
    paths["metadata"].write_text(
        json.dumps({"files": [{"path": "/tmp/a.mat"}], "warnings": []}),
        encoding="utf-8",
    )
    paths["probe"].write_text(
        json.dumps({"summary": {"file_error_count": 0, "can_probe_count": 1}, "warnings": []}),
        encoding="utf-8",
    )
    paths["mapping"].write_text(
        "\n".join(
            [
                "mapping_version: raw_variable_mapping_v001",
                "human_review:",
                "  required: false",
                "  remaining_uncertainties:",
                "    - Confirm depth units.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    paths["slice"].write_text(
        json.dumps({"errors": [], "warnings": ["time unit unknown"]}),
        encoding="utf-8",
    )
    _write_tiny_hdf5(paths["hdf5"])
    paths["qc"].write_text(
        json.dumps({"status": "passed", "errors": [], "warnings": []}),
        encoding="utf-8",
    )
    paths["config"].write_text(
        "\n".join(
            [
                "data:",
                f"  manifests: {tmp_path}",
                f"  reports: {tmp_path}",
                f"  interim: {tmp_path}",
                f"  processed: {tmp_path}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


def test_mvp1_gate_report_conditional_go(tmp_path: Path) -> None:
    paths = _write_common_inputs(tmp_path)
    output_md = tmp_path / "mvp1_gate_report.md"
    output_json = tmp_path / "mvp1_gate_report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/02b_generate_mvp1_gate_report.py",
            "--paths",
            str(paths["config"]),
            "--manifest",
            str(paths["manifest"]),
            "--mat-metadata",
            str(paths["metadata"]),
            "--struct-probe",
            str(paths["probe"]),
            "--mapping",
            str(paths["mapping"]),
            "--small-slice-summary",
            str(paths["slice"]),
            "--tiny-hdf5",
            str(paths["hdf5"]),
            "--qc-summary",
            str(paths["qc"]),
            "--output-report-md",
            str(output_md),
            "--output-report-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "decision=conditional_go" in result.stdout
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["decision"] == "conditional_go"
    assert output_md.exists()


def test_mvp1_gate_report_no_go_on_blocking_issue(tmp_path: Path) -> None:
    paths = _write_common_inputs(tmp_path)
    paths["qc"].write_text(
        json.dumps({"status": "failed", "errors": ["bad"], "warnings": []}),
        encoding="utf-8",
    )
    output_md = tmp_path / "mvp1_gate_report.md"
    output_json = tmp_path / "mvp1_gate_report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/02b_generate_mvp1_gate_report.py",
            "--paths",
            str(paths["config"]),
            "--manifest",
            str(paths["manifest"]),
            "--mat-metadata",
            str(paths["metadata"]),
            "--struct-probe",
            str(paths["probe"]),
            "--mapping",
            str(paths["mapping"]),
            "--small-slice-summary",
            str(paths["slice"]),
            "--tiny-hdf5",
            str(paths["hdf5"]),
            "--qc-summary",
            str(paths["qc"]),
            "--output-report-md",
            str(output_md),
            "--output-report-json",
            str(output_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["decision"] == "no_go"
    assert report["blocking_issues"]
