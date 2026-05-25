from __future__ import annotations

from cement_channel.alignment.depth_grid import depth_grid_config_dict, propose_depth_grid


def _audit_report() -> dict:
    return {
        "decision": "conditional_go",
        "depth_unit": "unknown_to_verify",
        "warnings": ["depth unit is unknown_to_verify"],
        "no_go_blockers": [],
        "common_overlap_interval": {"min": 100.0, "max": 110.0, "length": 10.0},
        "cast_depth": {"median_step": 0.25},
        "pose_depth": {"median_step": 0.5},
        "xsi_depth_by_receiver": {
            "receiver_01": {"median_step": 1.0},
            "receiver_02": {"median_step": 1.0},
        },
    }


def test_propose_depth_grid_uses_coarsest_step() -> None:
    proposal = propose_depth_grid(_audit_report(), source_audit_report="audit.json")

    assert proposal.decision == "conditional_go"
    assert proposal.depth_start == 100.0
    assert proposal.depth_stop == 110.0
    assert proposal.depth_step == 1.0
    assert proposal.sample_count == 11
    assert proposal.allow_extrapolation is False
    assert proposal.grid_order == "increasing"


def test_propose_depth_grid_blocks_missing_overlap() -> None:
    audit = _audit_report()
    audit["common_overlap_interval"] = {"min": 110.0, "max": 100.0, "length": -10.0}

    proposal = propose_depth_grid(audit, source_audit_report="audit.json")

    assert proposal.decision == "no_go"
    assert proposal.no_go_blockers


def test_depth_grid_config_marks_human_review_for_warnings() -> None:
    proposal = propose_depth_grid(_audit_report(), source_audit_report="audit.json")

    config = depth_grid_config_dict(proposal)

    assert config["canonical_depth_grid"]["depth_step"] == 1.0
    assert config["quality_gate"]["requires_human_review"] is True
    assert config["not_performed"]
