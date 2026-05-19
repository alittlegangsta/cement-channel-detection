from __future__ import annotations

from pathlib import Path

import pytest

from cement_channel.data.manifest import (
    ManifestBuildError,
    build_manifest,
    classify_single_well_file,
    parse_receiver_index,
)

FIXTURE_RAW = Path("tests/fixtures/tiny_sample/raw")


def _single_well_config(raw_dir: Path, expected_receivers: int = 13) -> dict:
    return {
        "schema_version": "schema_v001",
        "data": {
            "raw": str(raw_dir),
            "manifests": str(raw_dir.parent / "manifests"),
        },
        "raw_layout": {
            "organization": "single_well_flat",
            "well_id": "D2",
            "cast_files": ["CAST.fake_mat"],
            "pose_files": ["D2_XSI_RelBearing_Inclination.fake_mat"],
            "xsi_receiver_dir": "XSILMR",
            "xsi_receiver_file_patterns": ["XSILMR*.fake_mat"],
            "expected_xsi_receiver_files": expected_receivers,
        },
    }


def test_parse_receiver_index() -> None:
    assert parse_receiver_index("XSILMR01.fake_mat") == 1
    assert parse_receiver_index("XSILMR13.mat") == 13
    assert parse_receiver_index("CAST.fake_mat") is None


def test_classify_single_well_file_role() -> None:
    layout = _single_well_config(FIXTURE_RAW)["raw_layout"]

    cast_match = classify_single_well_file(FIXTURE_RAW / "CAST.fake_mat", FIXTURE_RAW, layout)
    pose_match = classify_single_well_file(
        FIXTURE_RAW / "D2_XSI_RelBearing_Inclination.fake_mat", FIXTURE_RAW, layout
    )
    receiver_match = classify_single_well_file(
        FIXTURE_RAW / "XSILMR" / "XSILMR01.fake_mat", FIXTURE_RAW, layout
    )

    assert cast_match is not None
    assert cast_match.file_role == "cast"
    assert pose_match is not None
    assert pose_match.file_role == "pose"
    assert receiver_match is not None
    assert receiver_match.file_role == "xsi_receiver"
    assert receiver_match.receiver_index == 1


def test_scan_single_well_flat_fixture() -> None:
    manifest = build_manifest(_single_well_config(FIXTURE_RAW))

    assert manifest["summary"]["file_count"] == 15
    assert manifest["summary"]["files_by_role"] == {
        "cast": 1,
        "pose": 1,
        "xsi_receiver": 13,
    }
    assert manifest["summary"]["warning_count"] == 0
    receiver_indices = [
        record["receiver_index"]
        for record in manifest["files"]
        if record["file_role"] == "xsi_receiver"
    ]
    assert receiver_indices == list(range(1, 14))


def test_missing_receiver_file_warning(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    receiver_dir = raw_dir / "XSILMR"
    receiver_dir.mkdir(parents=True)
    (raw_dir / "CAST.fake_mat").write_text("fake\n", encoding="utf-8")
    (raw_dir / "D2_XSI_RelBearing_Inclination.fake_mat").write_text("fake\n", encoding="utf-8")
    for receiver_index in range(1, 13):
        (receiver_dir / f"XSILMR{receiver_index:02d}.fake_mat").write_text(
            "fake\n", encoding="utf-8"
        )

    manifest = build_manifest(_single_well_config(raw_dir))

    assert manifest["summary"]["xsi_receiver_file_count"] == 12
    assert any(warning["code"] == "xsi_receiver_count_mismatch" for warning in manifest["warnings"])


def test_raw_dir_override() -> None:
    config = _single_well_config(Path("does-not-exist"))

    manifest = build_manifest(config, raw_dir_override=FIXTURE_RAW)

    assert manifest["raw_dir"] == str(FIXTURE_RAW)
    assert manifest["summary"]["file_count"] == 15


def test_missing_raw_dir_is_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ManifestBuildError, match="Raw directory does not exist"):
        build_manifest(_single_well_config(tmp_path / "missing_raw"))
