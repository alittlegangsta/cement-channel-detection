from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _write_inputs(interim_dir: Path, reports_dir: Path) -> Path:
    n = 24
    has_channel = np.zeros(n, dtype=bool)
    has_channel[:12] = True
    strong = np.zeros(n, dtype=bool)
    strong[:8] = True
    clear = np.zeros(n, dtype=bool)
    clear[12:22] = True
    np.savez_compressed(
        interim_dir / "depth_level_labels_v001.npz",
        depth=np.arange(n, dtype=np.float32),
        depth_has_channel_any=has_channel,
        depth_strong_positive_mask=strong,
        depth_clear_negative_mask=clear,
        depth_review_band_mask=np.zeros(n, dtype=bool),
        depth_label_confidence=np.ones(n, dtype=np.float32),
        depth_orientation_confidence=np.ones(n, dtype=np.float32),
        depth_plus_minus_disagreement_fraction=np.zeros(n, dtype=np.float32),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    signal = np.linspace(0.0, 0.2, n, dtype=np.float32)
    signal[:12] += 2.0
    np.savez_compressed(
        interim_dir / "depth_level_xsi_features_v001.npz",
        depth=np.arange(n, dtype=np.float32),
        depth_level_xsi_features=np.column_stack([signal]).astype(np.float32),
        depth_level_xsi_feature_names=np.asarray(["side_mean_late_over_early_ratio"]),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    side_report = reports_dir / "subset_feature_separation_audit_v001.json"
    side_report.write_text(
        json.dumps({"signal_enhancement": {"all_candidate_best_abs_effect_size": 0.1}}),
        encoding="utf-8",
    )
    return side_report


def _write_config(path: Path) -> None:
    config = yaml.safe_load(Path("configs/depth_level_label.example.yaml").read_text())
    config["gate"]["sanity_effect_size_threshold"] = 0.2
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


def test_audit_depth_level_separation_cli_writes_reports_csv_and_figures(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    _write_inputs(interim_dir, reports_dir)
    depth_config = tmp_path / "depth_level_label.yaml"
    _write_config(depth_config)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  interim: {interim_dir}",
                f"  reports: {reports_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06t_audit_depth_level_separation.py",
            "--paths",
            str(paths_config),
            "--depth-level-config",
            str(depth_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Depth-level separation audit errors=0" in result.stdout
    output_json = reports_dir / "depth_level_separation_audit_v001.json"
    output_csv = reports_dir / "depth_level_separation_audit_v001.csv"
    figure_dir = reports_dir / "depth_level_separation_audit_v001"
    assert output_json.exists()
    assert output_csv.exists()
    assert (figure_dir / "depth_level_effect_size_heatmap.png").exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["report_version"] == "depth_level_separation_audit_v001"
    assert report["depth_level_separation_enhanced"] is True
    assert report["no_model_training"] is True
