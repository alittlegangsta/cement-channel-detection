from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_generate_label_review_figures_cli_outputs_review_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    labels_dir = root_dir / "labels"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    labels_dir.mkdir()
    reports_dir.mkdir()
    zc = np.full((6, 8), 10.0, dtype=np.float32)
    zc[2:4, 2:4] = 4.0
    base = np.full_like(zc, 10.0)
    presence = np.zeros_like(zc, dtype=np.int8)
    presence[2:4, 2:4] = 1
    np.savez_compressed(interim_dir / "cast_label_input_v001.npz", cast_zc=zc)
    np.savez_compressed(
        interim_dir / "cast_zc_baseline_v001.npz",
        zc_base=base,
        relative_drop=(base - zc) / base,
    )
    np.savez_compressed(
        labels_dir / "cast_weak_label_candidates_v001.npz",
        presence_plus=presence,
        presence_minus_ablation=presence,
        label_confidence_plus=np.where(presence == 1, 0.8, 0.1).astype(np.float32),
        severity_plus=np.where(presence == 1, 2, 0).astype(np.int8),
        no_final_labels=np.asarray(True),
    )
    paths_config = tmp_path / "paths.yaml"
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

    result = subprocess.run(
        [
            sys.executable,
            "scripts/04e_generate_label_review_figures.py",
            "--paths",
            str(paths_config),
            "--max-depth-pixels",
            "10",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Label review figures errors=0" in result.stdout
    review_dir = reports_dir / "label_review_v001"
    summary = json.loads((review_dir / "label_review_summary_v001.json").read_text())
    assert summary["no_final_labels"] is True
    assert (review_dir / "01_cast_zc_raw.png").read_bytes().startswith(b"\x89PNG")
    assert (review_dir / "review_summary_template.md").exists()
