from __future__ import annotations

from pathlib import Path

import yaml


def test_xsi_geometry_example_records_manual_confirmations() -> None:
    config = yaml.safe_load(
        Path("configs/xsi_geometry.example.yaml").read_text(encoding="utf-8")
    )

    assert config["receiver_count"] == 13
    assert config["reference_receiver_index"] == 7
    assert config["receiver_spacing_ft"] == 0.5
    assert config["source_to_receiver1_ft"] == 1.0
    assert config["source_to_reference_receiver_ft"] == 4.0
    assert config["receiver_offsets_from_reference_ft"] == {
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
