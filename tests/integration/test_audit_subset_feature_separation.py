from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _write_sample_and_subsets(interim_dir: Path) -> None:
    n = 40
    label = np.zeros(n, dtype=np.int8)
    label[:20] = 1
    side_feature = np.zeros(n, dtype=np.float32)
    side_feature[:20] = 0.1
    receiver_feature = np.zeros(n, dtype=np.float32)
    receiver_feature[:10] = np.linspace(2.5, 3.5, 10, dtype=np.float32)
    receiver_feature[10:20] = 0.2
    receiver_feature[20:34] = np.linspace(0.0, 0.2, 14, dtype=np.float32)
    quality_pos = np.zeros(n, dtype=bool)
    quality_pos[:10] = True
    quality_neg = np.zeros(n, dtype=bool)
    quality_neg[20:34] = True
    strong = np.zeros(n, dtype=bool)
    strong[:14] = True
    clear = np.zeros(n, dtype=bool)
    clear[20:] = True
    np.savez_compressed(
        interim_dir / "baseline_sample_table_receiver_enhanced_v001.npz",
        label_presence_plus=label,
        transformed_features=np.column_stack([side_feature, receiver_feature]).astype(
            np.float32
        ),
        transformed_feature_names=np.asarray(
            [
                "per_depth_side_z_late_over_early_ratio",
                "robust_scaled_receiver_far_minus_near_late_over_early_ratio",
            ]
        ),
        receiver_transformed_feature_names_added=np.asarray(
            ["robust_scaled_receiver_far_minus_near_late_over_early_ratio"]
        ),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
    )
    np.savez_compressed(
        interim_dir / "label_quality_subsets_v001.npz",
        disagreement_free_mask=np.ones(n, dtype=bool),
        high_confidence_orientation_mask=np.ones(n, dtype=bool),
        connected_object_only_mask=strong,
        review_exclusion_mask=np.zeros(n, dtype=bool),
        strong_positive_mask=strong,
        clear_negative_mask=clear,
        quality_strong_positive_mask=quality_pos,
        quality_clear_negative_mask=quality_neg,
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
    )


def _write_config(path: Path) -> None:
    config = yaml.safe_load(Path("configs/mvp4b_label_quality_subsets.example.yaml").read_text())
    config["gate"]["signal_enhancement_effect_size_delta"] = 0.01
    config["gate"]["strong_signal_effect_size_threshold"] = 0.1
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_audit_subset_feature_separation_cli_writes_reports_and_figures(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    _write_sample_and_subsets(interim_dir)
    config_path = tmp_path / "label_quality.yaml"
    _write_config(config_path)
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
            "scripts/06p_audit_subset_feature_separation.py",
            "--paths",
            str(paths_config),
            "--label-quality-config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Subset feature audit errors=0" in result.stdout
    output_json = reports_dir / "subset_feature_separation_audit_v001.json"
    output_csv = reports_dir / "subset_feature_separation_audit_v001.csv"
    figure_dir = reports_dir / "subset_feature_separation_audit_v001"
    assert output_json.exists()
    assert output_csv.exists()
    assert (figure_dir / "subset_feature_effect_size_heatmap.png").exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["report_version"] == "subset_feature_separation_audit_v001"
    assert report["label_noise_likely"] is True
    assert report["no_final_labels"] is True
