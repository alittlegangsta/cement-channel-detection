from __future__ import annotations

import json

from cement_channel.data.mapping_suggester import (
    TODO_CONFIRM,
    format_mapping_draft_yaml,
    suggest_raw_variable_mapping,
)


def _field(field_path: str, shape: list[int], role_hint: str) -> dict:
    return {
        "top_variable": field_path.split(".")[0],
        "field_path": field_path,
        "shape": shape,
        "dtype_or_class": "float32",
        "role_hint": role_hint,
        "element_count": 1,
        "preview_stats": {},
    }


def _file(
    filename: str,
    file_role: str,
    fields: list[dict],
    receiver_index: int | None = None,
) -> dict:
    return {
        "path": f"/tmp/{filename}",
        "filename": filename,
        "file_role": file_role,
        "receiver_index": receiver_index,
        "can_probe": True,
        "mat_format": "matlab_v5_or_v7",
        "probed_variables": [],
        "fields": fields,
        "warnings": [],
        "errors": [],
    }


def _probe() -> dict:
    xsi_fields = [
        _field("XSILMR01.Depth", [1, 4], "depth_candidate"),
        _field("XSILMR01.Tad", [1, 1], "xsi_time_candidate"),
    ]
    xsi_fields.extend(
        _field(f"XSILMR01.WaveRng01Side{side}", [1024, 4], "xsi_waveform_candidate")
        for side in "ABCDEFGH"
    )
    return {
        "probe_version": "mat_struct_probe_v001",
        "files": [
            _file(
                "CAST.mat",
                "cast",
                [
                    _field("CAST.Zc", [180, 4], "cast_zc_candidate"),
                    _field("CAST.Depth", [1, 4], "depth_candidate"),
                ],
            ),
            _file(
                "D2_XSI_RelBearing_Inclination.mat",
                "pose",
                [
                    _field("Depth_inc", [1, 4], "depth_candidate"),
                    _field("Inc", [1, 4], "inclination_candidate"),
                    _field("RelBearing", [1, 4], "relbearing_candidate"),
                ],
            ),
            _file("XSILMR01.mat", "xsi_receiver", xsi_fields, receiver_index=1),
        ],
        "warnings": [],
    }


def test_suggests_core_raw_mapping_variables() -> None:
    result = suggest_raw_variable_mapping(
        _probe(),
        struct_probe_json_path="/tmp/mat_struct_probe_v001.json",
        well_id="D2",
    )

    recommendations = result.recommendations
    assert recommendations["cast.zc_variable"].variable_path == "CAST.Zc"
    assert recommendations["cast.depth_variable"].variable_path == "CAST.Depth"
    assert recommendations["pose.inclination_variable"].variable_path == "Inc"
    assert recommendations["pose.relbearing_variable"].variable_path == "RelBearing"
    assert (
        recommendations["xsi.waveform_variable"].variable_path
        == "XSILMR01.WaveRng01Side{A-H}"
    )
    assert recommendations["xsi.depth_variable"].variable_path == "XSILMR01.Depth"
    assert recommendations["xsi.time_variable"].variable_path == "XSILMR01.Tad"
    assert result.human_review_required


def test_uncertain_missing_mapping_uses_todo_confirm() -> None:
    probe = {"files": [_file("CAST.mat", "cast", [_field("CAST.Unknown", [1], "unknown")])]}

    result = suggest_raw_variable_mapping(
        probe,
        struct_probe_json_path="/tmp/probe.json",
        well_id="D2",
    )

    assert result.recommendations["cast.zc_variable"].variable_path == TODO_CONFIRM
    assert any("No reliable recommendation" in warning for warning in result.warnings)


def test_draft_yaml_is_parseable_and_marks_human_review() -> None:
    result = suggest_raw_variable_mapping(
        _probe(),
        struct_probe_json_path="/tmp/probe.json",
        well_id="D2",
    )
    draft_yaml = format_mapping_draft_yaml(result)

    import yaml

    parsed = yaml.safe_load(draft_yaml)
    assert parsed["status"] == "draft_requires_human_review"
    assert parsed["cast"]["zc_variable"] == "CAST.Zc"
    assert parsed["xsi"]["waveform_variable"] == "XSILMR01.WaveRng01Side{A-H}"
    assert parsed["human_review"]["required"] is True


def test_suggestion_result_is_json_serializable() -> None:
    result = suggest_raw_variable_mapping(
        _probe(),
        struct_probe_json_path="/tmp/probe.json",
        well_id="D2",
    )

    json.dumps(result.to_dict())
