from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_run_label_threshold_sensitivity_cli_outputs_reports(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    zc = np.full((4, 8), 10.0, dtype=np.float32)
    zc[1:3, 2:4] = 5.0
    zc[3, 0] = 0.2
    base = np.full_like(zc, 10.0)
    relative_drop = ((base - zc) / base).astype(np.float32)
    np.savez_compressed(
        interim_dir / "cast_label_input_v001.npz",
        cast_depth=np.arange(4, dtype=np.float32) + 100.0,
        cast_azimuth_deg=np.arange(8, dtype=np.float32) * 45.0,
        cast_zc=zc,
        relbearing_deg=np.zeros(4, dtype=np.float32),
        orientation_confidence=np.ones(4, dtype=np.float32),
        orientation_uncertain=np.zeros(4, dtype=bool),
    )
    np.savez_compressed(
        interim_dir / "cast_zc_baseline_v001.npz",
        zc_base=base,
        relative_drop=relative_drop,
        zc_ratio=(zc / base).astype(np.float32),
        baseline_valid=np.ones_like(zc, dtype=bool),
        finite_fraction=np.ones_like(zc, dtype=np.float32),
    )
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
    label_config = tmp_path / "label.yaml"
    label_config.write_text(
        "\n".join(
            [
                "threshold:",
                "  relative_drop_alpha: 0.35",
                "  zc_min_limit: TODO_CONFIRM",
                "  conservative_fallback_mrayl: 2.5",
                "  candidate_coverage_warning_max: 1.0",
                "severity:",
                "  mild_min_drop: 0.30",
                "  moderate_min_drop: 0.45",
                "  severe_min_drop: 0.60",
                "confidence:",
                "  relative_drop_full_confidence: 0.70",
                "  bad_data_confidence: 0.0",
                "bad_data:",
                "  extreme_relative_drop_threshold: 0.95",
                "audit:",
                "  isolated_object_max_pixels: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/04g_run_label_threshold_sensitivity.py",
            "--paths",
            str(paths_config),
            "--label-config",
            str(label_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Label threshold sensitivity errors=0" in result.stdout
    summary = json.loads(
        (reports_dir / "label_threshold_sensitivity_v001.json").read_text(encoding="utf-8")
    )
    assert summary["no_final_labels"] is True
    assert len(summary["results"]) == 27
    assert (reports_dir / "label_threshold_sensitivity_v001.md").exists()
    assert (reports_dir / "label_threshold_sensitivity_v001.csv").exists()
