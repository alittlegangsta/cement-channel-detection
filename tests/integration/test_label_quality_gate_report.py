from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_reports(reports_dir: Path, *, label_noise_likely: bool) -> None:
    reports_dir.mkdir(parents=True)
    (reports_dir / "label_quality_subsets_report_v001.json").write_text(
        json.dumps(
            {
                "subset_counts": {
                    "quality_strong_positive": {"sample_count": 120},
                    "quality_clear_negative": {"sample_count": 220},
                },
                "warnings": [],
                "errors": [],
                "no_final_labels": True,
                "no_stc": True,
                "no_apes": True,
                "no_deep_learning": True,
                "no_mvp4c": True,
            }
        ),
        encoding="utf-8",
    )
    delta = 0.08 if label_noise_likely else 0.01
    quality = 0.36 if label_noise_likely else 0.24
    (reports_dir / "subset_feature_separation_audit_v001.json").write_text(
        json.dumps(
            {
                "signal_enhancement": {
                    "quality_subset_best_abs_effect_size": quality,
                    "quality_minus_all_delta": delta,
                    "label_noise_likely": label_noise_likely,
                },
                "review_exclusion_sensitivity": {
                    "result_flip_exceeds_threshold": False,
                    "sign_flip_fraction": 0.0,
                },
                "label_noise_likely": label_noise_likely,
                "warnings": [],
                "errors": [],
                "no_final_labels": True,
                "no_stc": True,
                "no_apes": True,
                "no_deep_learning": True,
                "no_mvp4c": True,
            }
        ),
        encoding="utf-8",
    )
    (reports_dir / "receiver_feature_gate_report.json").write_text(
        json.dumps({"decision": "no_go"}),
        encoding="utf-8",
    )


def _write_paths(path: Path, root_dir: Path, reports_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  reports: {reports_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_label_quality_gate_allows_only_controlled_time_frequency_when_signal_improves(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    _write_reports(reports_dir, label_noise_likely=True)
    paths_config = tmp_path / "paths.yaml"
    _write_paths(paths_config, root_dir, reports_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06q_generate_label_quality_gate_report.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "decision=go" in result.stdout
    report = json.loads((reports_dir / "label_quality_gate_report.json").read_text())
    assert report["controlled_time_frequency_sanity_allowed"] is True
    assert report["mvp4c_allowed"] is False


def test_label_quality_gate_remains_no_go_when_subset_signal_is_weak(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    _write_reports(reports_dir, label_noise_likely=False)
    paths_config = tmp_path / "paths.yaml"
    _write_paths(paths_config, root_dir, reports_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06q_generate_label_quality_gate_report.py",
            "--paths",
            str(paths_config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "decision=no_go" in result.stdout
    report = json.loads((reports_dir / "label_quality_gate_report.json").read_text())
    assert report["controlled_time_frequency_sanity_allowed"] is False
    assert report["mvp4c_consideration_allowed"] is False
