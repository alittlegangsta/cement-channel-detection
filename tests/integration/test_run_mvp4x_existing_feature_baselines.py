from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_snapshot(interim: Path) -> None:
    depth = np.arange(15, dtype=np.float32)
    target = depth / 15.0
    np.savez_compressed(
        interim / "mvp4x_research_snapshot_v001.npz",
        depth=depth,
        xsi_features=np.column_stack([depth, np.cos(depth)]).astype(np.float32),
        xsi_feature_names=np.asarray(["side_mean_energy", "receiver_std_rms"]),
        model_feature_mask=np.asarray([True, True]),
        receiver_p90=target.astype(np.float32),
        receiver_mean=target.astype(np.float32),
        receiver_max=target.astype(np.float32),
        broad_regime_id=np.asarray(["A"] * 5 + ["B"] * 5 + ["C"] * 5),
        saturation_platform_flag=np.zeros(15, dtype=bool),
        special_5680_flag=np.zeros(15, dtype=bool),
        low_orientation_confidence_flag=np.zeros(15, dtype=bool),
        any_special_flag=np.zeros(15, dtype=bool),
        research_only=np.asarray(True),
        exploratory_only=np.asarray(True),
        weak_label_target=np.asarray(True),
        no_final_labels=np.asarray(True),
        no_ground_truth_claim=np.asarray(True),
        no_production_claim=np.asarray(True),
    )


def test_run_existing_feature_baselines_cli_writes_reports(tmp_path: Path) -> None:
    root = tmp_path / "data"
    interim = root / "interim"
    reports = root / "reports"
    interim.mkdir(parents=True)
    reports.mkdir()
    _write_snapshot(interim)
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root}",
                f"  interim: {interim}",
                f"  reports: {reports}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07b_run_mvp4x_existing_feature_baselines.py",
            "--paths",
            str(paths),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "MVP-4X existing-feature baselines" in result.stdout
    output_json = reports / "mvp4x_existing_feature_baselines_v001.json"
    output_csv = reports / "mvp4x_existing_feature_baselines_v001.csv"
    assert output_json.exists()
    assert output_csv.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["research_only"] is True
    assert report["derived_binary_audit_only"] is True
    assert report["no_production_claim"] is True
    assert output_csv.read_text(encoding="utf-8").splitlines()[0].startswith("csv_version")
