from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def test_build_screening_policy_cli_writes_outputs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for name in ("interim", "features", "reports", "labels", "manifests"):
        (data / name).mkdir(parents=True)
    sample_count = 7108
    regimes = np.asarray(["A"] * 2370 + ["B"] * 2369 + ["C"] * 2369)
    low = np.zeros(sample_count, dtype=bool)
    low[:2370] = True
    low[2370 : 2370 + 1326] = True
    np_flags = {
        "research_only": np.asarray(True),
        "exploratory_only": np.asarray(True),
        "weak_label_target": np.asarray(True),
        "no_final_labels": np.asarray(True),
        "no_ground_truth_claim": np.asarray(True),
        "no_production_claim": np.asarray(True),
    }
    report_flags = {key: bool(value.item()) for key, value in np_flags.items()}
    np.savez_compressed(
        data / "interim/mvp4x_research_snapshot_v001.npz",
        depth=np.linspace(2350.0, 5760.0, sample_count, dtype=np.float32),
        xsi_features=np.ones((sample_count, 80), dtype=np.float32),
        target_kernel=np.asarray("triangular_midpoint_weighted"),
        broad_regime_id=regimes,
        low_orientation_confidence_flag=low,
        any_special_flag=np.zeros(sample_count, dtype=bool),
        receiver_mean=np.linspace(0.0, 0.2, sample_count, dtype=np.float32),
        receiver_p90=np.linspace(0.0, 0.2, sample_count, dtype=np.float32),
        receiver_max=np.linspace(0.0, 0.2, sample_count, dtype=np.float32),
        full_360_fraction=np.zeros(sample_count, dtype=np.float32),
        receiver_std=np.ones(sample_count, dtype=np.float32) * 0.01,
        **np_flags,
    )
    np.savez_compressed(
        data / "features/mvp4x_waveform_features_v001.npz",
        waveform_depth_features=np.ones((sample_count, 342), dtype=np.float32),
        **np_flags,
    )
    for filename in (
        "mvp4x_regime_policy_v001.json",
        "mvp4x_regime_robustness_v001.json",
        "mvp4x_ranking_audit_v001.json",
        "mvp4x_regime_error_analysis_v001.json",
    ):
        (data / "reports" / filename).write_text(json.dumps(report_flags), encoding="utf-8")
    (data / "reports/mvp4x_screening_baseline_v001.json").write_text(
        json.dumps(
            {
                "supported_cohorts": ["pooled_bc_high_orientation"],
                "unsupported_cohorts": ["regime_b_all"],
                "domain_shift": True,
                **report_flags,
            }
        ),
        encoding="utf-8",
    )
    (data / "reports/mvp4x_regime_policy_decision.json").write_text(
        json.dumps(
            {
                "decision": "research_screening_baseline_domain_shift_limited",
                "answers": {"18_stable_features": []},
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

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07q_build_mvp4x_screening_policy.py",
            "--paths",
            str(paths),
            "--policy-config",
            "configs/mvp4x_screening_baseline.example.yaml",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output_json = data / "reports/mvp4x_screening_policy_v001.json"
    output_npz = data / "interim/mvp4x_screening_policy_v001.npz"
    assert output_json.exists()
    assert output_npz.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["not_validated_for_deployment"]
    assert report["primary_research_score"]["model"] == "Ridge"
