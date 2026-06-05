from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def test_generate_screening_review_cli_writes_decision(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for name in ("interim", "features", "reports", "labels", "manifests"):
        (data / name).mkdir(parents=True)
    sample_count = 7108
    np.savez_compressed(
        data / "interim/mvp4x_research_snapshot_v001.npz",
        depth=np.linspace(2350.0, 5760.0, sample_count, dtype=np.float32),
        morphology_min_zc=np.ones(sample_count, dtype=np.float32),
        orientation_confidence=np.ones(sample_count, dtype=np.float32),
        low_orientation_confidence_flag=np.zeros(sample_count, dtype=bool),
        inclination_deg=np.ones(sample_count, dtype=np.float32),
    )
    np.savez_compressed(
        data / "interim/mvp4x_screening_scores_oof_v001.npz",
        score=np.ones(sample_count, dtype=np.float32),
    )
    rows = []
    for index in range(sample_count):
        regime = "A" if index < 2370 else ("B" if index < 4739 else "C")
        rows.append(
            {
                "depth": float(index),
                "regime_id": regime,
                "orientation_cohort": "low_orientation" if regime == "A" else "high_orientation",
                "supported_cohort": "P1" if regime != "A" else "",
                "score": 0.5 if regime != "A" else "",
                "score_rank_percentile": 0.9 if regime != "A" else "",
                "target_receiver_mean": 0.1,
                "residual": 0.1 if regime != "A" else "",
                "score_stability_std": 0.01 if regime != "A" else "",
                "special_band_flags": json.dumps({"any_special": index == 5000}),
            }
        )
    with (data / "reports/mvp4x_screening_scores_oof_v001.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    flags = {
        "research_only": True,
        "exploratory_only": True,
        "weak_label_target": True,
        "no_final_labels": True,
        "no_ground_truth_claim": True,
        "no_production_claim": True,
    }
    extended = {**flags, "not_validated_for_deployment": True}
    (data / "reports/mvp4x_screening_scores_oof_v001.json").write_text(
        json.dumps(
            {
                "sample_counts": {"supported_scored": 1},
                "gap_stability": {"warning_count": 0},
                "rank_metrics": {"top_k": {}},
                **extended,
            }
        ),
        encoding="utf-8",
    )
    (data / "reports/mvp4x_screening_policy_v001.json").write_text(
        json.dumps(
            {
                "supported_screening_cohorts": ["pooled_bc_high_orientation"],
                "unsupported_or_audit_only_cohorts": ["regime_a_all"],
                **extended,
            }
        ),
        encoding="utf-8",
    )
    candidate = {
        "cohort": "pooled_bc_high_orientation",
        "sample_count": 10,
        "repeated_cv": {"metrics": {"spearman": {"mean": 0.2}, "r2": {"mean": -0.1}}},
        "bootstrap_ci": {"metrics": {"spearman": {"ci_lower": 0.1}}},
        "blocked_gap": {},
        "permutation": {"margins": {"global": 0.2}},
        "special_band_sensitivity": {"dependency_flag": False},
    }
    (data / "reports/mvp4x_regime_robustness_v001.json").write_text(
        json.dumps({"candidate_rows": [candidate], **flags}),
        encoding="utf-8",
    )
    (data / "reports/mvp4x_ranking_audit_v001.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "cohort": "pooled_bc_high_orientation",
                        "ranking": {"top_k": {"top_10pct": {"lift": 2.0}}},
                        "derived_ordinal_audit": {"macro_f1": 0.4},
                    }
                ],
                **flags,
            }
        ),
        encoding="utf-8",
    )
    (data / "reports/mvp4x_regime_error_analysis_v001.json").write_text(
        json.dumps({"summary": {"domain_shift_warning": True}, **flags}),
        encoding="utf-8",
    )
    (data / "manifests/mvp4x_research_screening_model_v001.json").write_text(
        json.dumps({"not_validated_for_deployment": True, "forbidden_use": [], **extended}),
        encoding="utf-8",
    )
    (data / "reports/mvp4x_regime_policy_decision.json").write_text(
        json.dumps(
            {
                "decision": "research_screening_baseline_domain_shift_limited",
                **flags,
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
            "scripts/07t_generate_mvp4x_screening_review.py",
            "--paths",
            str(paths),
            "--review-config",
            "configs/mvp4x_screening_baseline.example.yaml",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    decision = json.loads(
        (data / "reports/mvp4x_screening_consolidation_decision.json").read_text()
    )
    assert decision["decision"] == "request_formal_regime_policy_approval"
    assert (data / "reports/mvp4x_screening_consolidation_review_v001/review_summary.md").exists()
