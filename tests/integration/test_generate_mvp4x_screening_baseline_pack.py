from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_screening_pack_cli_writes_decision_and_review(tmp_path: Path) -> None:
    data = tmp_path / "data"
    reports = data / "reports"
    for name in ("interim", "features", "reports", "labels", "manifests"):
        (data / name).mkdir(parents=True)
    flags = {
        "research_only": True,
        "exploratory_only": True,
        "weak_label_target": True,
        "no_final_labels": True,
        "no_ground_truth_claim": True,
        "no_production_claim": True,
    }
    cohort = "pooled_bc_high_orientation"
    robustness = {
        "screening_support": {
            "supported_screening_cohorts": [cohort],
            "unsupported_screening_cohorts": [],
        },
        "candidate_rows": [
            {
                "cohort": cohort,
                "sample_count": 10,
                "repeated_cv": {"metrics": {"spearman": {"mean": 0.2}, "r2": {"mean": -0.1}}},
                "bootstrap_ci": {"metrics": {"spearman": {"ci_lower": 0.1}}},
                "permutation": {
                    "margins": {"global": 0.2, "within_depth_bin": 0.2, "block": 0.2}
                },
                "blocked_gap": {
                    "gap_10_ft": {"metrics": {"spearman": {"mean": 0.2}}},
                    "gap_25_ft": {"metrics": {"spearman": {"mean": 0.2}}},
                },
                "special_band_sensitivity": {"dependency_flag": False},
            }
        ],
        **flags,
    }
    ranking = {
        "summary": {"ranking_stable_cohorts": [cohort], "ranking_unstable_cohorts": []},
        "rows": [
            {
                "cohort": cohort,
                "ranking": {"top_k": {"top_10pct": {"lift": 2.0}}},
                "derived_ordinal_audit": {"macro_f1": 0.4},
            }
        ],
        **flags,
    }
    error = {
        "summary": {
            "absolute_calibration_insufficient": True,
            "max_abs_calibration_bias": 0.08,
            "domain_shift_warning": True,
            "transfer_limits": {"b_to_c_spearman": 0.05, "c_to_b_spearman": 0.1},
            "top_stable_features": [
                {"feature_group": "existing:side_std", "mean_spearman_drop": 0.1}
            ],
        },
        "rows": [
            {
                "cohort": cohort,
                "residual_vs_depth": 0.1,
                "residual_summary": {"mean": 0.0},
            }
        ],
        **flags,
    }
    stratified = {
        "answers": {
            "4_b_to_c_transfer": 0.05,
            "5_c_to_b_transfer": 0.1,
            "12_domain_shift_exists": True,
        },
        **flags,
    }
    (reports / "mvp4x_regime_policy_v001.json").write_text(json.dumps(flags), encoding="utf-8")
    (reports / "mvp4x_regime_robustness_v001.json").write_text(
        json.dumps(robustness),
        encoding="utf-8",
    )
    (reports / "mvp4x_ranking_audit_v001.json").write_text(json.dumps(ranking), encoding="utf-8")
    (reports / "mvp4x_regime_error_analysis_v001.json").write_text(
        json.dumps(error),
        encoding="utf-8",
    )
    (reports / "mvp4x_stratified_decision.json").write_text(
        json.dumps(stratified),
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
            "scripts/07p_generate_mvp4x_screening_baseline_pack.py",
            "--paths",
            str(paths),
            "--overwrite",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    decision = json.loads((reports / "mvp4x_regime_policy_decision.json").read_text())
    screening = json.loads((reports / "mvp4x_screening_baseline_v001.json").read_text())
    assert decision["decision"] == "research_screening_baseline_domain_shift_limited"
    assert screening["research_only"]
    assert (reports / "mvp4x_regime_policy_review_v001/review_summary.md").exists()
