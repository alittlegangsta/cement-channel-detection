from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def test_error_analysis_cli_writes_reports(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for name in ("interim", "features", "reports", "labels", "manifests"):
        (data / name).mkdir(parents=True)
    sample_count = 7108
    regimes = np.asarray(["A"] * 2370 + ["B"] * 2369 + ["C"] * 2369)
    depth = np.linspace(2350.0, 5760.0, sample_count, dtype=np.float32)
    x = np.linspace(0.0, 1.0, sample_count, dtype=np.float32).reshape(-1, 1)
    y = x.reshape(-1)
    flags = {
        "research_only": np.asarray(True),
        "exploratory_only": np.asarray(True),
        "weak_label_target": np.asarray(True),
        "no_final_labels": np.asarray(True),
        "no_ground_truth_claim": np.asarray(True),
        "no_production_claim": np.asarray(True),
    }
    np.savez_compressed(
        data / "interim/mvp4x_research_snapshot_v001.npz",
        depth=depth,
        xsi_features=np.tile(x, (1, 80)).astype(np.float32),
        xsi_feature_names=np.asarray([f"x{i}" for i in range(80)]),
        xsi_feature_group=np.asarray(["side_std"] * 80),
        target_kernel=np.asarray("triangular_midpoint_weighted"),
        broad_regime_id=regimes,
        orientation_confidence=np.ones(sample_count, dtype=np.float32),
        saturation_platform_flag=np.zeros(sample_count, dtype=bool),
        special_5680_flag=np.zeros(sample_count, dtype=bool),
        any_special_flag=np.zeros(sample_count, dtype=bool),
        receiver_mean=y,
        receiver_p90=y,
        receiver_max=y,
        full_360_fraction=y,
        receiver_std=y,
        **flags,
    )
    np.savez_compressed(
        data / "features/mvp4x_waveform_features_v001.npz",
        waveform_depth_features=np.ones((sample_count, 342), dtype=np.float32),
        waveform_depth_feature_names=np.asarray([f"w{i}" for i in range(342)]),
        waveform_depth_feature_group=np.asarray(["wave"] * 342),
        **flags,
    )
    cohort_names = np.asarray(["regime_b_all"])
    cohort_masks = (regimes == "B").reshape(1, -1)
    np.savez_compressed(
        data / "interim/mvp4x_regime_policy_v001.npz",
        cohort_names=cohort_names,
        cohort_masks=cohort_masks,
        metadata_json=np.asarray(json.dumps({"policy_version": "test"})),
        **flags,
    )
    report_flags = {key: bool(value.item()) for key, value in flags.items()}
    candidate = {
        "feature_set": "existing_features_only",
        "target": "receiver_mean",
        "model": "Ridge",
    }
    (data / "reports/mvp4x_regime_robustness_v001.json").write_text(
        json.dumps(
            {
                "candidate_rows": [{"cohort": "regime_b_all", "candidate": candidate}],
                **report_flags,
            }
        ),
        encoding="utf-8",
    )
    (data / "reports/mvp4x_ranking_audit_v001.json").write_text(
        json.dumps({"summary": {"ranking_stable_cohorts": ["regime_b_all"]}, **report_flags}),
        encoding="utf-8",
    )
    (data / "reports/mvp4x_stratified_decision.json").write_text(
        json.dumps({"answers": {"12_domain_shift_exists": True}, **report_flags}),
        encoding="utf-8",
    )
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        yaml.safe_dump(
            {
                "data": {
                    name: str(data / name)
                    for name in ("interim", "features", "reports", "labels", "manifests")
                }
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "policy.yaml"
    config.write_text(yaml.safe_dump({"robustness": {"n_contiguous_folds": 3}}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07o_analyze_mvp4x_regime_errors.py",
            "--paths",
            str(paths),
            "--policy-config",
            str(config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((data / "reports/mvp4x_regime_error_analysis_v001.json").read_text())
    assert report["research_only"]
    assert len(report["rows"]) == 1
