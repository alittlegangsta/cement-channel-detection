from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_generate_mvp4x_rapid_decision_cli_writes_stop_decision(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    flags = {
        "research_only": True,
        "exploratory_only": True,
        "weak_label_target": True,
        "no_final_labels": True,
        "no_ground_truth_claim": True,
        "no_production_claim": True,
    }
    _write_json(
        reports / "mvp4x_research_snapshot_report_v001.json",
        {"errors": [], "sample_count": 10, "feature_count": 2, "target_kernel": "k", **flags},
    )
    _write_json(
        reports / "mvp4x_existing_feature_baselines_v001.json",
        {
            "errors": [],
            "sklearn_available": False,
            "model_backend": "sklearn_unavailable_all_skipped",
            "decision": {"baseline_sufficient": False, "leakage_detected": False},
            "target_summaries": {"receiver_p90": {"finite_ratio": 1.0}},
            "warnings": ["sklearn unavailable"],
            **flags,
        },
    )
    _write_json(
        reports / "mvp4x_waveform_feature_report_v001.json",
        {
            "errors": [],
            "feature_version": "mvp4x_waveform_features_v001",
            "sample_count": 10,
            "receiver_count": 1,
            "side_count": 2,
            "depth_feature_count": 3,
            "chunk_count": 1,
            "peak_memory_bytes": 100,
            "finite_ratio": {},
            **flags,
        },
    )
    _write_json(
        reports / "mvp4x_enhanced_feature_baselines_v001.json",
        {
            "errors": [],
            "feature_sets": {
                "combined_features": {
                    "group_counts": {"existing:g": 2, "waveform:w": 3},
                    "feature_count": 5,
                }
            },
            "sub_reports": {},
            "best_result": None,
            "feature_group_ablation": {"status": "skipped_no_completed_models"},
            "permutation_importance": {"status": "skipped_no_completed_models"},
            "warnings": [],
            **flags,
        },
    )
    (reports / "mvp4x_rapid_iteration_log.md").write_text(
        "## iteration_001\n## iteration_002\n",
        encoding="utf-8",
    )
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        "\n".join(["data:", f"  reports: {reports}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07e_generate_mvp4x_rapid_decision.py",
            "--paths",
            str(paths),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "decision=exploratory_signal_insufficient_stop" in result.stdout
    output_json = reports / "mvp4x_rapid_decision.json"
    output_md = reports / "mvp4x_rapid_decision.md"
    assert output_json.exists()
    assert output_md.exists()
    decision = json.loads(output_json.read_text(encoding="utf-8"))
    assert decision["decision"] == "exploratory_signal_insufficient_stop"
    assert decision["answers"]["2_waveform_feature_extraction_triggered"] is True
    assert decision["answers"]["17_research_only_marking_complete"] is True
