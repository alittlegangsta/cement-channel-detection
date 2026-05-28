from __future__ import annotations

from pathlib import Path

import yaml


def test_xsi_geometry_example_records_manual_confirmations() -> None:
    config = yaml.safe_load(
        Path("configs/xsi_geometry.example.yaml").read_text(encoding="utf-8")
    )
    geometry = config["xsi_geometry"]

    assert geometry["receiver_count"] == 13
    assert geometry["reference_receiver_index"] == 7
    assert geometry["receiver_spacing_ft"] == 0.5
    assert geometry["r1_source_distance_ft"] == 1.0
    assert geometry["source_offset_relative_to_r7_ft"] == -4.0
    assert geometry["depth_axis_sign"] == "audit_both"
    assert geometry["sign_convention_status"] == "requires_audit"
    assert geometry["alignment_modes"] == [
        "r7_reference_depth",
        "receiver_depth_shifted",
        "source_receiver_midpoint",
        "source_receiver_interval",
    ]
    assert geometry["receiver_offsets_relative_to_r7_ft"] == {
        "R1": -3.0,
        "R2": -2.5,
        "R3": -2.0,
        "R4": -1.5,
        "R5": -1.0,
        "R6": -0.5,
        "R7": 0.0,
        "R8": 0.5,
        "R9": 1.0,
        "R10": 1.5,
        "R11": 2.0,
        "R12": 2.5,
        "R13": 3.0,
    }

    side = config["side_geometry"]
    assert side["side_a_aligned_with_cast_0deg"] is True
    assert side["side_a_offset_deg"] == 0.0
    assert side["side_a_offset_status"] == "manually_confirmed"
    assert side["xsi_side_order"] == "clockwise"
    assert side["xsi_side_order_status"] == "manually_confirmed"

    cast = config["cast_azimuth"]
    assert cast["cast_azimuth_direction"] == "normal"
    assert cast["cast_azimuth_direction_status"] == "manually_confirmed"
    assert cast["column_1_deg"] == 0.0
    assert cast["column_2_deg"] == 2.0
    assert cast["last_column_deg"] == 358.0

    relbearing = config["relbearing_convention"]
    assert (
        relbearing["relbearing_sign_status"]
        == "specification_preferred_plus_data_unresolved"
    )
    assert relbearing["primary_convention"] == "plus"
    assert relbearing["ablation_convention"] == "minus"
    assert relbearing["data_driven_validation"] == "insufficient_evidence"
    assert relbearing["single_sign_alignment_approved"] is False
