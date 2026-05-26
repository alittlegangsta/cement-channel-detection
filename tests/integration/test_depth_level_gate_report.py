from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def _write_reports(reports_dir: Path) -> None:
    (reports_dir / "depth_level_labels_report_v001.json").write_text(
        json.dumps(
            {
                "positive_fraction": 0.5,
                "positive_count": 10,
                "negative_count": 10,
                "strong_positive_count": 4,
                "clear_negative_count": 6,
                "review_band_impact": {"positive_fraction_in_review_band": 0.0},
                "no_final_labels": True,
                "no_stc": True,
                "no_apes": True,
                "no_deep_learning": True,
                "no_mvp4c": True,
                "warnings": [],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    (reports_dir / "depth_level_xsi_features_report_v001.json").write_text(
        json.dumps(
            {
                "depth_feature_count": 8,
                "no_final_labels": True,
                "no_stc": True,
                "no_apes": True,
                "no_deep_learning": True,
                "no_mvp4c": True,
                "warnings": [],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    (reports_dir / "depth_level_separation_audit_v001.json").write_text(
        json.dumps(
            {
                "depth_level_separation_enhanced": True,
                "depth_vs_side_comparison": {
                    "depth_level_best_abs_effect_size": 0.8,
                    "side_level_best_abs_effect_size": 0.3,
                    "depth_minus_side_delta": 0.5,
                },
                "no_final_labels": True,
                "no_stc": True,
                "no_apes": True,
                "no_deep_learning": True,
                "no_mvp4c": True,
                "warnings": [],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )


def _write_config(path: Path) -> None:
    config = yaml.safe_load(Path("configs/depth_level_label.example.yaml").read_text())
    config["quality_policy"]["review_intervals"] = [
        {
            "name": "review_horizontal_severe_band_5700ft",
            "depth_min_ft": 5680.0,
            "depth_max_ft": 5720.0,
            "reason": "outside synthetic sample but required by schema",
            "apply_by_default": True,
        }
    ]
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_depth_level_gate_report_cli_allows_only_depth_baseline_sanity(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True)
    _write_reports(reports_dir)
    depth_config = tmp_path / "depth_level_label.yaml"
    _write_config(depth_config)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
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

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06u_generate_depth_level_gate_report.py",
            "--paths",
            str(paths_config),
            "--depth-level-config",
            str(depth_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "decision=conditional_go" in result.stdout
    output_json = reports_dir / "depth_level_gate_report_v001.json"
    output_md = reports_dir / "depth_level_gate_report_v001.md"
    assert output_json.exists()
    assert output_md.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["decision"] == "conditional_go"
    assert report["depth_level_baseline_sanity_allowed"] is True
    assert report["mvp4c_allowed"] is False
    assert report["stc_allowed"] is False
    assert report["final_labels_allowed"] is False
