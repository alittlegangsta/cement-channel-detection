from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_generate_cast_weak_labels_cli_outputs_candidates(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    labels_dir = root_dir / "labels"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    labels_dir.mkdir()
    reports_dir.mkdir()
    zc = np.full((4, 8), 10.0, dtype=np.float32)
    zc[1, 0] = 4.0
    base = np.full_like(zc, 10.0)
    relative_drop = (base - zc) / base
    np.savez_compressed(
        interim_dir / "cast_label_input_v001.npz",
        cast_depth=np.arange(4, dtype=np.float32),
        cast_azimuth_deg=np.arange(8, dtype=np.float32) * 45.0,
        cast_zc=zc,
        relbearing_deg=np.zeros(4, dtype=np.float32),
        orientation_confidence=np.ones(4, dtype=np.float32),
        orientation_uncertain=np.zeros(4, dtype=bool),
    )
    np.savez_compressed(
        interim_dir / "cast_zc_baseline_v001.npz",
        zc_base=base,
        relative_drop=relative_drop.astype(np.float32),
        zc_ratio=(zc / base).astype(np.float32),
        baseline_valid=np.ones_like(zc, dtype=bool),
    )
    paths_config = tmp_path / "paths.yaml"
    label_config = tmp_path / "label.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  interim: {interim_dir}",
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
                "  relative_drop_alpha: 0.35",
                "  zc_min_limit: TODO_CONFIRM",
                "  conservative_fallback_mrayl: 2.5",
                "  require_confirmed_zc_min_limit: false",
                "  candidate_coverage_warning_max: 0.5",
                "severity:",
                "  mild_min_drop: 0.30",
                "  moderate_min_drop: 0.45",
                "  severe_min_drop: 0.60",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/04c_generate_cast_weak_labels.py",
            "--paths",
            str(paths_config),
            "--label-config",
            str(label_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CAST weak-label candidates errors=0" in result.stdout
    output_npz = labels_dir / "cast_weak_label_candidates_v001.npz"
    output_json = reports_dir / "cast_weak_label_candidates_report_v001.json"
    output_md = reports_dir / "cast_weak_label_candidates_report_v001.md"
    assert output_npz.exists()
    assert output_md.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["no_final_labels"] is True
    assert report["threshold"]["zc_min_limit_status"] == "requires_human_threshold_confirmation"
    with np.load(output_npz) as data:
        assert data["presence_plus"].shape == (4, 8)
        assert data["presence_plus"][1, 0] == 1
        assert bool(data["no_final_labels"])
