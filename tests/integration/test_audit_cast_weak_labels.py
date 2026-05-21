from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_audit_cast_weak_labels_cli_outputs_reports(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    labels_dir = root_dir / "labels"
    reports_dir = root_dir / "reports"
    labels_dir.mkdir(parents=True)
    reports_dir.mkdir()
    presence = np.zeros((4, 8), dtype=np.int8)
    presence[1:3, 1:3] = 1
    np.savez_compressed(
        labels_dir / "cast_weak_label_candidates_v001.npz",
        cast_depth=np.arange(4, dtype=np.float32),
        cast_azimuth_aligned_deg=np.arange(8, dtype=np.float32) * 45.0,
        presence_plus=presence,
        severity_plus=np.where(presence == 1, 2, 0).astype(np.int8),
        label_confidence_plus=np.where(presence == 1, 0.8, 0.2).astype(np.float32),
        presence_minus_ablation=presence,
        severity_minus_ablation=np.where(presence == 1, 2, 0).astype(np.int8),
        label_confidence_minus_ablation=np.where(presence == 1, 0.8, 0.2).astype(
            np.float32
        ),
        no_final_labels=np.asarray(True),
    )
    paths_config = tmp_path / "paths.yaml"
    label_config = tmp_path / "label.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  labels: {labels_dir}",
                f"  reports: {reports_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    label_config.write_text(
        "\n".join(
            [
                "threshold:",
                "  candidate_coverage_warning_max: 0.5",
                "audit:",
                "  max_plus_minus_disagreement_warning: 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/04d_audit_cast_weak_labels.py",
            "--paths",
            str(paths_config),
            "--label-config",
            str(label_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CAST weak-label audit errors=0" in result.stdout
    output_json = reports_dir / "cast_weak_label_audit_v001.json"
    output_md = reports_dir / "cast_weak_label_audit_v001.md"
    assert output_md.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["no_final_labels"] is True
    assert report["components"]["plus"]["component_count"] == 1
