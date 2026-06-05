from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def test_generate_screening_scores_cli_writes_oof_outputs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for name in ("interim", "features", "reports", "labels", "manifests"):
        (data / name).mkdir(parents=True)
    sample_count = 7108
    regimes = np.asarray(["A"] * 2370 + ["B"] * 2369 + ["C"] * 2369)
    low = np.zeros(sample_count, dtype=bool)
    low[:2370] = True
    low[2370 : 2370 + 1326] = True
    features = np.column_stack(
        [
            np.linspace(0.0, 1.0, sample_count, dtype=np.float32),
            np.ones((sample_count, 79), dtype=np.float32),
        ]
    )
    target = features[:, 0] * 0.1
    np.savez_compressed(
        data / "interim/mvp4x_research_snapshot_v001.npz",
        depth=np.linspace(2350.0, 5760.0, sample_count, dtype=np.float32),
        xsi_features=features,
        xsi_feature_names=np.asarray([f"f{i}" for i in range(80)]),
        model_feature_mask=np.ones(80, dtype=bool),
        target_kernel=np.asarray("triangular_midpoint_weighted"),
        broad_regime_id=regimes,
        low_orientation_confidence_flag=low,
        saturation_platform_flag=np.zeros(sample_count, dtype=bool),
        transition_2582_flag=np.zeros(sample_count, dtype=bool),
        transition_4219_flag=np.zeros(sample_count, dtype=bool),
        special_5680_flag=np.zeros(sample_count, dtype=bool),
        any_special_flag=np.zeros(sample_count, dtype=bool),
        receiver_mean=target,
        receiver_p90=target,
        receiver_max=target,
        research_only=np.asarray(True),
        exploratory_only=np.asarray(True),
        weak_label_target=np.asarray(True),
        no_final_labels=np.asarray(True),
        no_ground_truth_claim=np.asarray(True),
        no_production_claim=np.asarray(True),
    )
    names = np.asarray(
        [
            "pooled_bc_all",
            "pooled_bc_high_orientation",
            "regime_b_high_orientation",
            "regime_c_all",
            "regime_c_high_orientation",
            "regime_a_all",
            "regime_b_all",
            "low_orientation_outside_supported_cohorts",
        ]
    )
    masks = []
    regime_b = regimes == "B"
    regime_c = regimes == "C"
    high = ~low
    for name in names:
        if name == "pooled_bc_all":
            masks.append(regime_b | regime_c)
        elif name == "pooled_bc_high_orientation":
            masks.append((regime_b | regime_c) & high)
        elif name == "regime_b_high_orientation":
            masks.append(regime_b & high)
        elif name == "regime_c_all":
            masks.append(regime_c)
        elif name == "regime_c_high_orientation":
            masks.append(regime_c & high)
        elif name == "regime_a_all":
            masks.append(regimes == "A")
        elif name == "regime_b_all":
            masks.append(regime_b)
        else:
            masks.append(low & (regimes == "A"))
    np.savez_compressed(
        data / "interim/mvp4x_screening_policy_v001.npz",
        cohort_names=names,
        cohort_masks=np.vstack(masks),
    )
    policy = {
        "research_only": True,
        "exploratory_only": True,
        "weak_label_target": True,
        "no_final_labels": True,
        "no_ground_truth_claim": True,
        "no_production_claim": True,
        "not_validated_for_deployment": True,
    }
    (data / "reports/mvp4x_screening_policy_v001.json").write_text(
        json.dumps(policy),
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

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07r_generate_mvp4x_screening_scores.py",
            "--paths",
            str(paths),
            "--score-config",
            "configs/mvp4x_screening_baseline.example.yaml",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((data / "reports/mvp4x_screening_scores_oof_v001.json").read_text())
    assert report["not_validated_for_deployment"]
    assert report["sample_counts"]["total"] == sample_count
    assert (data / "reports/mvp4x_screening_scores_oof_v001.csv").exists()
    scores = np.load(data / "interim/mvp4x_screening_scores_oof_v001.npz")
    assert scores["score"].shape == (sample_count,)
