from __future__ import annotations

from copy import deepcopy

from cement_channel.data.raw_mapping import audit_raw_metadata


def _variable(name: str, shape: list[int], role_hint: str = "unknown") -> dict:
    return {
        "name": name,
        "shape": shape,
        "dtype_or_class": "double",
        "is_numeric": True,
        "element_count": 1,
        "role_hint": role_hint,
    }


def _file(
    filename: str,
    file_role: str,
    variables: list[dict],
    receiver_index: int | None = None,
) -> dict:
    return {
        "path": f"/tmp/{filename}",
        "filename": filename,
        "file_role": file_role,
        "receiver_index": receiver_index,
        "can_open": True,
        "mat_format": "matlab_v5_or_v7",
        "variables": variables,
        "warnings": [],
        "errors": [],
    }


def _metadata(receiver_count: int = 13, inconsistent_shape: bool = False) -> dict:
    files = [
        _file(
            "CAST.mat",
            "cast",
            [
                _variable("Zc", [4, 180], "cast_zc_candidate"),
                _variable("depth", [4, 1], "depth_candidate"),
                _variable("impedance_map", [4, 180]),
                _variable("cast_zc", [4, 180]),
            ],
        ),
        _file(
            "D2_XSI_RelBearing_Inclination.mat",
            "pose",
            [
                _variable("MD", [4, 1], "depth_candidate"),
                _variable("Inc", [4, 1], "inclination_candidate"),
                _variable("Inclination", [4, 1]),
                _variable("RelBearing", [4, 1], "relbearing_candidate"),
            ],
        ),
    ]
    for receiver_index in range(1, receiver_count + 1):
        waveform_shape = [4, 1024]
        if inconsistent_shape and receiver_index == receiver_count:
            waveform_shape = [5, 1024]
        files.append(
            _file(
                f"XSILMR{receiver_index:02d}.mat",
                "xsi_receiver",
                [
                    _variable("waveform", waveform_shape, "xsi_waveform_candidate"),
                    _variable("data", waveform_shape),
                    _variable("XSI", waveform_shape),
                    _variable("depth", [4, 1], "depth_candidate"),
                    _variable("time", [1024]),
                ],
                receiver_index=receiver_index,
            )
        )
    return {
        "metadata_version": "mat_metadata_v001",
        "schema_version": "schema_v001",
        "data_version": "data_v001",
        "files": files,
        "warnings": [],
    }


def test_candidate_variable_identification() -> None:
    result = audit_raw_metadata(_metadata(), metadata_json_path="/tmp/mat_metadata_v001.json")

    assert result.status == "pass"
    assert {candidate.variable_name for candidate in result.candidates["cast_zc_candidates"]} >= {
        "Zc",
        "impedance_map",
        "cast_zc",
    }
    assert {
        candidate.variable_name for candidate in result.candidates["cast_depth_candidates"]
    } >= {"depth"}
    assert {
        candidate.variable_name for candidate in result.candidates["inclination_candidates"]
    } >= {
        "Inc",
        "Inclination",
    }
    assert {
        candidate.variable_name for candidate in result.candidates["relbearing_candidates"]
    } >= {"RelBearing"}
    assert {
        candidate.variable_name for candidate in result.candidates["xsi_waveform_candidates"]
    } >= {
        "waveform",
        "data",
        "XSI",
    }


def test_complete_receivers_have_no_receiver_warning() -> None:
    result = audit_raw_metadata(_metadata(), metadata_json_path="/tmp/mat_metadata_v001.json")

    assert result.xsi_warnings == []


def test_missing_receiver_warns() -> None:
    result = audit_raw_metadata(
        _metadata(receiver_count=12), metadata_json_path="/tmp/missing.json"
    )

    assert any("receiver indexes are incomplete" in warning for warning in result.xsi_warnings)


def test_inconsistent_receiver_shape_warns() -> None:
    result = audit_raw_metadata(
        _metadata(inconsistent_shape=True),
        metadata_json_path="/tmp/inconsistent.json",
    )

    assert any("inconsistent shapes" in warning for warning in result.xsi_warnings)


def test_file_metadata_errors_are_summarized_without_crash() -> None:
    metadata = deepcopy(_metadata())
    metadata["files"][0]["can_open"] = False
    metadata["files"][0]["variables"] = []
    metadata["files"][0]["errors"] = ["failed to read MAT metadata"]

    result = audit_raw_metadata(metadata, metadata_json_path="/tmp/errors.json")

    assert result.status == "warning"
    assert result.statistics["files_with_errors"] == 1
    assert any("Some MAT files reported metadata errors" in warning for warning in result.warnings)
