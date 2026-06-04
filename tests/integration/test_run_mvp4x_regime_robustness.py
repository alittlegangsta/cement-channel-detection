from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def test_run_regime_robustness_cli_writes_reports(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for name in ("interim", "features", "reports", "labels", "manifests"):
        (data / name).mkdir(parents=True)
    sample_count = 7108
    rng = np.random.default_rng(4)
    regimes = np.asarray(["A"] * 2370 + ["B"] * 2369 + ["C"] * 2369)
    low = np.zeros(sample_count, dtype=bool)
    low[:2370] = True
    low[2370 : 2370 + 1326] = True
    depth = np.linspace(2350.0, 5760.0, sample_count, dtype=np.float32)
    x = np.column_stack(
        [
            np.linspace(0.0, 1.0, sample_count),
            rng.normal(0.0, 0.01, sample_count),
        ]
    ).astype(np.float32)
    target = (0.1 * x[:, 0] + rng.normal(0.0, 0.001, sample_count)).astype(np.float32)
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
        xsi_features=np.tile(x, (1, 40)).astype(np.float32),
        xsi_feature_names=np.asarray([f"x{i}" for i in range(80)]),
        xsi_feature_group=np.asarray(["side_std"] * 80),
        target_kernel=np.asarray("triangular_midpoint_weighted"),
        broad_regime_id=regimes,
        low_orientation_confidence_flag=low,
        saturation_platform_flag=np.zeros(sample_count, dtype=bool),
        special_5680_flag=np.zeros(sample_count, dtype=bool),
        any_special_flag=np.zeros(sample_count, dtype=bool),
        receiver_mean=target,
        receiver_p90=target,
        receiver_max=target,
        full_360_fraction=target,
        receiver_std=target,
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
    (data / "reports/mvp4x_stratified_baselines_v001.json").write_text(
        json.dumps(
            {
                "model_matrix": {
                    "best_by_cohort": {
                        "regime_b_all": {
                            "feature_set": "existing_features_only",
                            "target": "receiver_mean",
                            "model": "Ridge",
                            "spearman": 0.2,
                            "mae": 0.1,
                            "r2": -0.1,
                            "stable_positive_spearman_folds": 3,
                        }
                    }
                },
                **report_flags,
            }
        ),
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
    config.write_text(
        yaml.safe_dump(
            {
                "robustness": {
                    "candidate_cohorts": ["regime_b_all"],
                    "repeated_seeds": [11],
                    "blocked_gap_ft": [10.0],
                    "permutation_count": 1,
                    "bootstrap_repeats": 5,
                    "n_contiguous_folds": 3,
                    "ridge_alpha": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07m_run_mvp4x_regime_robustness.py",
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
    output_json = data / "reports/mvp4x_regime_robustness_v001.json"
    output_csv = data / "reports/mvp4x_regime_robustness_v001.csv"
    assert output_json.exists()
    assert output_csv.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["research_only"]
    assert report["candidate_count"] == 1
