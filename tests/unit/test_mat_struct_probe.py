from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from cement_channel.data.mat_struct_probe import (
    probe_file_structs,
    probe_structs_from_metadata,
    role_hint_for_field_path,
)


def _write_struct_mat(path: Path) -> None:
    savemat(
        path,
        {
            "CAST": {
                "Zc": np.arange(12, dtype=float).reshape(3, 4),
                "depth": np.array([1.0, 2.0, 3.0]),
                "azimuth": np.array([0.0, 90.0, 180.0, 270.0]),
                "nested": {"waveform": np.ones((3, 8))},
            }
        },
    )


def _file_metadata(path: Path) -> dict:
    return {
        "path": str(path),
        "filename": path.name,
        "file_role": "cast",
        "receiver_index": None,
        "can_open": True,
        "mat_format": "matlab_v5_or_v7",
        "variables": [
            {
                "name": "CAST",
                "shape": [1, 1],
                "dtype_or_class": "struct",
                "role_hint": "unknown",
            }
        ],
        "warnings": [],
        "errors": [],
    }


def test_struct_field_probe_identifies_fields(tmp_path: Path) -> None:
    mat_path = tmp_path / "cast_struct.mat"
    _write_struct_mat(mat_path)

    result = probe_file_structs(_file_metadata(mat_path), max_field_depth=2)

    field_paths = {field.field_path: field for field in result.fields}
    assert result.can_probe
    assert "CAST.Zc" in field_paths
    assert field_paths["CAST.Zc"].role_hint == "cast_zc_candidate"
    assert "CAST.depth" in field_paths
    assert field_paths["CAST.depth"].role_hint == "depth_candidate"
    assert "CAST.azimuth" in field_paths
    assert field_paths["CAST.azimuth"].role_hint == "cast_azimuth_candidate"


def test_role_hint_for_field_path() -> None:
    assert role_hint_for_field_path("pose.Inc", [1, 10]) == "inclination_candidate"
    assert role_hint_for_field_path("pose.RelBearing", [1, 10]) == "relbearing_candidate"
    assert role_hint_for_field_path("cast.Zc", [10, 180]) == "cast_zc_candidate"
    assert role_hint_for_field_path("xsi.time", [1024]) == "xsi_time_candidate"
    assert role_hint_for_field_path("xsi.waveform", [10, 1024]) == "xsi_waveform_candidate"


def test_max_files_and_field_depth_are_respected(tmp_path: Path) -> None:
    mat_path = tmp_path / "cast_struct.mat"
    _write_struct_mat(mat_path)
    metadata = {"files": [_file_metadata(mat_path), _file_metadata(mat_path)]}

    result = probe_structs_from_metadata(
        metadata,
        metadata_json_path=tmp_path / "mat_metadata_v001.json",
        max_files=1,
        max_field_depth=1,
    )

    assert result.summary["file_count"] == 1
    field_paths = [field.field_path for field in result.files[0].fields]
    assert "CAST.nested.waveform" not in field_paths


def test_corrupt_mat_file_records_error(tmp_path: Path) -> None:
    mat_path = tmp_path / "broken.mat"
    mat_path.write_text("not a real mat file", encoding="utf-8")

    result = probe_file_structs(_file_metadata(mat_path))

    assert not result.can_probe
    assert result.errors


def test_probe_result_is_json_serializable(tmp_path: Path) -> None:
    mat_path = tmp_path / "cast_struct.mat"
    _write_struct_mat(mat_path)
    result = probe_structs_from_metadata(
        {"files": [_file_metadata(mat_path)]},
        metadata_json_path=tmp_path / "metadata.json",
    )

    json.dumps(result.to_dict())
