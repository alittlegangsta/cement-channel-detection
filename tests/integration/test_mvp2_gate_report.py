from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _conclusion(
    *,
    single_sign_alignment_approved: bool = False,
    status: str = "specification_preferred_plus_data_unresolved",
) -> dict[str, object]:
    return {
        "relbearing_sign_status": status,
        "documentation_preferred_sign": "plus",
        "primary_convention": "plus",
        "ablation_convention": "minus",
        "documentation_formula": "theta_aligned = (theta_raw + RelBearing) mod 360",
        "data_driven_validation": "insufficient_evidence",
        "single_sign_alignment_approved": single_sign_alignment_approved,
        "single_sign_alignment_approval_basis": "specification_only_not_data_confirmed",
        "approved_downstream_mode": "plus_primary_minus_ablation",
        "documentation_basis": (
            "Halliburton Relative Bearing documentation suggests plus under looking-downhole "
            "clockwise tool-key assumptions."
        ),
        "manual_confirmations": {
            "side_a_aligned_with_cast_0deg": True,
            "side_a_offset_deg": 0.0,
            "side_a_offset_status": "manually_confirmed",
            "xsi_side_order": "clockwise",
            "xsi_side_order_status": "manually_confirmed",
            "cast_azimuth_direction": "normal",
            "cast_azimuth_values": "0,2,4,...,358",
            "cast_azimuth_direction_status": "manually_confirmed",
        },
        "unresolved_assumptions": [
            "Data-driven plus/minus/no/random metrics are not sign-discriminative.",
        ],
    }


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    interim_dir = tmp_path / "interim"
    reports_dir = tmp_path / "reports"
    interim_dir.mkdir()
    reports_dir.mkdir()
    paths = {
        "config": tmp_path / "paths.yaml",
        "depth_axis_audit": reports_dir / "depth_axis_audit_report.json",
        "depth_grid": reports_dir / "depth_grid_proposal.json",
        "depth_only": interim_dir / "depth_only_summary_v001.json",
        "resample": reports_dir / "depth_resample_preview_report.json",
        "resample_overlap": reports_dir / "depth_resample_overlap_preview_report.json",
        "relbearing": reports_dir / "relbearing_sign_validation_report.json",
        "relbearing_overlap": reports_dir / "relbearing_sign_validation_overlap_report.json",
        "orientation": reports_dir / "orientation_confidence_report.json",
        "output_md": reports_dir / "mvp2_gate_report.md",
        "output_json": reports_dir / "mvp2_gate_report.json",
    }
    paths["config"].write_text(
        "\n".join(["data:", f"  interim: {interim_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )
    paths["depth_axis_audit"].write_text(
        json.dumps(
            {
                "decision": "conditional_go",
                "depth_unit": "unknown_to_verify",
                "common_overlap_interval": {"min": 100.0, "max": 110.0, "length": 10.0},
                "errors": [],
                "warnings": ["depth unit unknown"],
                "no_go_blockers": [],
            }
        ),
        encoding="utf-8",
    )
    paths["depth_grid"].write_text(
        json.dumps(
            {
                "decision": "conditional_go",
                "depth_start": 100.0,
                "depth_stop": 110.0,
                "depth_step": 0.5,
                "sample_count": 21,
                "errors": [],
                "warnings": ["depth unit unknown"],
                "no_go_blockers": [],
            }
        ),
        encoding="utf-8",
    )
    paths["depth_only"].write_text(
        json.dumps(
            {
                "arrays": {
                    "cast_depth": {},
                    "xsi_depth_by_receiver": {},
                    "pose_depth": {},
                    "inc_deg": {},
                    "relbearing_deg": {},
                },
                "errors": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    paths["resample"].write_text(
        json.dumps(
            {
                "small_slice": {"status": "skipped_no_common_overlap"},
                "errors": [],
                "warnings": ["no initial overlap"],
            }
        ),
        encoding="utf-8",
    )
    paths["resample_overlap"].write_text(
        json.dumps(
            {
                "small_slice": {"status": "completed"},
                "arrays": {
                    "small_slice_cast_zc_on_preview": {},
                    "small_slice_xsi_waveform_on_preview": {},
                },
                "errors": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    relbearing_report = {
        "decision": "insufficient_evidence",
        "selected_convention": None,
        "convention_conclusion": _conclusion(),
        "errors": [],
        "warnings": [],
    }
    paths["relbearing"].write_text(json.dumps(relbearing_report), encoding="utf-8")
    paths["relbearing_overlap"].write_text(json.dumps(relbearing_report), encoding="utf-8")
    paths["orientation"].write_text(
        json.dumps(
            {
                "arrays": {
                    "orientation_confidence": {},
                    "orientation_uncertain": {},
                    "low_inc_mask": {},
                },
                "relbearing_sign_dependency": "independent_of_plus_minus_convention",
                "low_inclination_ratio": 0.25,
                "stable_inclination_ratio": 0.5,
                "errors": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    return paths


def test_mvp2_gate_report_conditional_go_for_documentation_plus(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/03h_generate_mvp2_gate_report.py",
            "--paths",
            str(paths["config"]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "decision=conditional_go" in result.stdout
    report = json.loads(paths["output_json"].read_text(encoding="utf-8"))
    assert report["decision"] == "conditional_go"
    assert report["documentation_preferred_convention"] == "plus"
    assert report["relbearing_convention"]["relbearing_sign_status"] == (
        "specification_preferred_plus_data_unresolved"
    )
    assert report["primary_convention"] == "plus"
    assert report["ablation_convention"] == "minus"
    assert report["data_driven_validation"] == "insufficient_evidence"
    assert report["single_sign_alignment_approved"] is False
    assert (
        report["single_sign_alignment_approval_basis"]
        == "specification_only_not_data_confirmed"
    )
    assert report["approved_downstream_mode"] == "plus_primary_minus_ablation"
    assert report["manual_confirmations"]["side_a_offset_deg"] == 0.0
    assert report["manual_confirmations"]["side_a_offset_status"] == "manually_confirmed"
    assert report["manual_confirmations"]["xsi_side_order"] == "clockwise"
    assert report["manual_confirmations"]["cast_azimuth_direction"] == "normal"
    assert report["xsi_geometry"]["receiver_count"] == 13
    assert report["xsi_geometry"]["reference_receiver_index"] == 7
    assert paths["output_md"].exists()


def test_mvp2_gate_report_normalizes_legacy_documentation_status(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    legacy_report = {
        "decision": "insufficient_evidence",
        "selected_convention": None,
        "convention_conclusion": _conclusion(
            status="documentation_preferred_plus_data_unresolved"
        ),
        "errors": [],
        "warnings": [],
    }
    paths["relbearing_overlap"].write_text(json.dumps(legacy_report), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/03h_generate_mvp2_gate_report.py",
            "--paths",
            str(paths["config"]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(paths["output_json"].read_text(encoding="utf-8"))
    assert report["decision"] == "conditional_go"
    assert report["relbearing_convention"]["relbearing_sign_status"] == (
        "specification_preferred_plus_data_unresolved"
    )


def test_mvp2_gate_report_no_go_on_blocking_errors(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    paths["resample_overlap"].write_text(
        json.dumps(
            {
                "small_slice": {"status": "skipped_no_common_overlap"},
                "arrays": {},
                "errors": ["no overlap"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/03h_generate_mvp2_gate_report.py",
            "--paths",
            str(paths["config"]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(paths["output_json"].read_text(encoding="utf-8"))
    assert report["decision"] == "no_go"
    assert report["blocking_issues"]


def test_mvp2_gate_report_no_go_if_single_sign_alignment_approved(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    relbearing_report = {
        "decision": "insufficient_evidence",
        "selected_convention": None,
        "convention_conclusion": _conclusion(single_sign_alignment_approved=True),
        "errors": [],
        "warnings": [],
    }
    paths["relbearing_overlap"].write_text(json.dumps(relbearing_report), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/03h_generate_mvp2_gate_report.py",
            "--paths",
            str(paths["config"]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(paths["output_json"].read_text(encoding="utf-8"))
    assert report["decision"] == "no_go"
    assert report["single_sign_alignment_approved"] is True
