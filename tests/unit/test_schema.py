from __future__ import annotations

from copy import deepcopy

from cement_channel.data.schema import (
    EXPECTED_CAST_AZIMUTH_COUNT,
    EXPECTED_RECEIVER_COUNT,
    EXPECTED_SIDE_COUNT,
    EXPECTED_TIME_SAMPLE_COUNT,
    LEGAL_PRESENCE_LABELS,
    LEGAL_SEVERITY_LABELS,
    validate_manifest_basic,
)


def _valid_manifest() -> dict:
    files = [
        {"file_role": "cast", "filename": "CAST.fake_mat"},
        {"file_role": "pose", "filename": "D2_XSI_RelBearing_Inclination.fake_mat"},
    ]
    files.extend(
        {"file_role": "xsi_receiver", "filename": f"XSILMR{receiver_index:02d}.fake_mat"}
        for receiver_index in range(1, 14)
    )
    return {
        "schema_version": "schema_v001",
        "data_version": "data_v001",
        "created_at": "2026-05-19T00:00:00+00:00",
        "raw_layout": {"expected_xsi_receiver_files": 13},
        "wells": [
            {
                "well_id": "D2",
                "expected_xsi_receiver_files": 13,
                "files": files,
            }
        ],
    }


def test_schema_constants() -> None:
    assert EXPECTED_RECEIVER_COUNT == 13
    assert EXPECTED_SIDE_COUNT == 8
    assert EXPECTED_TIME_SAMPLE_COUNT == 1024
    assert EXPECTED_CAST_AZIMUTH_COUNT == 180
    assert LEGAL_PRESENCE_LABELS == {-1, 0, 1}
    assert LEGAL_SEVERITY_LABELS == {-1, 0, 1, 2, 3}


def test_validate_manifest_basic_valid() -> None:
    result = validate_manifest_basic(_valid_manifest())

    assert result.is_valid
    assert result.errors == []
    assert result.warnings == []


def test_validate_manifest_basic_empty_wells_error() -> None:
    manifest = _valid_manifest()
    manifest["wells"] = []

    result = validate_manifest_basic(manifest)

    assert not result.is_valid
    assert any("at least one well" in error for error in result.errors)


def test_validate_manifest_basic_receiver_missing_warning() -> None:
    manifest = deepcopy(_valid_manifest())
    manifest["wells"][0]["files"] = [
        file_record
        for file_record in manifest["wells"][0]["files"]
        if file_record["filename"] != "XSILMR13.fake_mat"
    ]

    result = validate_manifest_basic(manifest)

    assert result.is_valid
    assert any("receiver count mismatch" in warning for warning in result.warnings)
