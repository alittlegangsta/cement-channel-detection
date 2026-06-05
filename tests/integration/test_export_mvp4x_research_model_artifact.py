from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml


def test_export_research_model_artifact_cli_writes_manifest(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for name in ("interim", "features", "reports", "labels", "manifests"):
        (data / name).mkdir(parents=True)
    sample_count = 7108
    low = np.zeros(sample_count, dtype=bool)
    low[:2370] = True
    features = np.ones((sample_count, 80), dtype=np.float32)
    target = np.linspace(0.0, 0.2, sample_count, dtype=np.float32)
    np.savez_compressed(
        data / "interim/mvp4x_research_snapshot_v001.npz",
        depth=np.linspace(2350.0, 5760.0, sample_count, dtype=np.float32),
        xsi_features=features,
        xsi_feature_names=np.asarray([f"f{i}" for i in range(80)]),
        model_feature_mask=np.ones(80, dtype=bool),
        target_kernel=np.asarray("triangular_midpoint_weighted"),
        receiver_mean=target,
        research_only=np.asarray(True),
        exploratory_only=np.asarray(True),
        weak_label_target=np.asarray(True),
        no_final_labels=np.asarray(True),
        no_ground_truth_claim=np.asarray(True),
        no_production_claim=np.asarray(True),
    )
    names = np.asarray(["pooled_bc_high_orientation"])
    masks = np.ones((1, sample_count), dtype=bool)
    np.savez_compressed(
        data / "interim/mvp4x_screening_policy_v001.npz",
        cohort_names=names,
        cohort_masks=masks,
    )
    flags = {
        "research_only": True,
        "exploratory_only": True,
        "weak_label_target": True,
        "no_final_labels": True,
        "no_ground_truth_claim": True,
        "no_production_claim": True,
    }
    policy_flags = {**flags, "not_validated_for_deployment": True}
    (data / "reports/mvp4x_screening_policy_v001.json").write_text(
        json.dumps({"unsupported_or_audit_only_cohorts": [], **policy_flags}),
        encoding="utf-8",
    )
    (data / "reports/mvp4x_screening_scores_oof_v001.json").write_text(
        json.dumps({"leakage_checks": {}, "gap_stability": {}, **policy_flags}),
        encoding="utf-8",
    )
    (data / "reports/mvp4x_regime_robustness_v001.json").write_text(
        json.dumps({"candidate_rows": [], **flags}),
        encoding="utf-8",
    )
    (data / "reports/mvp4x_ranking_audit_v001.json").write_text(
        json.dumps(flags),
        encoding="utf-8",
    )
    (data / "reports/mvp4x_regime_error_analysis_v001.json").write_text(
        json.dumps({"summary": {"domain_shift_warning": True}, **flags}),
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
            "scripts/07s_export_mvp4x_research_model_artifact.py",
            "--paths",
            str(paths),
            "--artifact-config",
            "configs/mvp4x_screening_baseline.example.yaml",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        (data / "manifests/mvp4x_research_screening_model_v001.json").read_text()
    )
    assert manifest["not_validated_for_deployment"]
    artifact = joblib.load(data / "features/mvp4x_research_screening_model_v001.joblib")
    assert artifact["not_for_deployment"]
