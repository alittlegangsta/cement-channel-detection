from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _baseline_report(*, margin: float = 0.08, usable: bool = True) -> dict:
    check = {
        "target_variant": "high_confidence_positive_vs_clear_negative",
        "model_type": "logistic_regression",
        "real_balanced_accuracy": 0.58,
        "permutation_balanced_accuracy": 0.50,
        "balanced_accuracy_margin": margin,
        "required_margin": 0.03,
        "predicted_positive_rate": 0.57,
        "degenerate_prediction": False,
        "stable_fold_count": 3,
        "stable_fold_min_count": 2,
        "stable_folds_pass": True,
        "permutation_lower_than_real": True,
        "passes_margin": margin >= 0.03,
        "usable": usable,
    }
    return {
        "report_version": "depth_level_baseline_v001",
        "best_result": check if usable else None,
        "usable_target_variants": (
            ["high_confidence_positive_vs_clear_negative"] if usable else []
        ),
        "production_training": False,
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_mvp4c": True,
        "warnings": [
            "strong_positive_vs_clear_negative: too few samples for baseline sanity."
        ],
        "errors": [],
    }


def _review_summary() -> dict:
    return {
        "review_version": "depth_level_baseline_review_v001",
        "figures": {"a": "a.png", "b": "b.png"},
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_mvp4c": True,
        "no_production_model": True,
        "warnings": [],
        "errors": [],
    }


def _write_inputs(reports_dir: Path, *, margin: float = 0.08, usable: bool = True) -> None:
    _write_json(
        reports_dir / "depth_level_baseline_report_v001.json",
        _baseline_report(margin=margin, usable=usable),
    )
    _write_json(
        reports_dir
        / "depth_level_baseline_review_v001"
        / "depth_level_baseline_review_summary_v001.json",
        _review_summary(),
    )


def _write_paths_config(path: Path, reports_dir: Path) -> None:
    root_dir = reports_dir.parent
    path.write_text(
        "\n".join(["data:", f"  root: {root_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )


def test_depth_level_baseline_gate_allows_controlled_feature_refinement(
    tmp_path: Path,
) -> None:
    reports_dir = tmp_path / "data" / "reports"
    _write_inputs(reports_dir)
    paths_config = tmp_path / "paths.yaml"
    _write_paths_config(paths_config, reports_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06x_generate_depth_level_baseline_gate.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "decision=conditional_go" in result.stdout
    report = json.loads(
        (reports_dir / "depth_level_baseline_gate_v001.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "conditional_go"
    assert report["controlled_depth_level_feature_refinement_allowed"] is True
    assert report["mvp4c_allowed"] is False
    assert report["stc_allowed"] is False
    assert report["deep_learning_allowed"] is False
    assert report["final_labels_allowed"] is False


def test_depth_level_baseline_gate_blocks_failed_margin(tmp_path: Path) -> None:
    reports_dir = tmp_path / "data" / "reports"
    _write_inputs(reports_dir, margin=0.01)
    paths_config = tmp_path / "paths.yaml"
    _write_paths_config(paths_config, reports_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06x_generate_depth_level_baseline_gate.py",
            "--paths",
            str(paths_config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(
        (reports_dir / "depth_level_baseline_gate_v001.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "no_go"
    assert report["controlled_depth_level_feature_refinement_allowed"] is False
    assert any("margin" in issue for issue in report["blocking_issues"])


def test_depth_level_baseline_gate_blocks_no_usable_variant(tmp_path: Path) -> None:
    reports_dir = tmp_path / "data" / "reports"
    _write_inputs(reports_dir, usable=False)
    paths_config = tmp_path / "paths.yaml"
    _write_paths_config(paths_config, reports_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06x_generate_depth_level_baseline_gate.py",
            "--paths",
            str(paths_config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(
        (reports_dir / "depth_level_baseline_gate_v001.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "no_go"
    assert any("no target variant" in issue for issue in report["blocking_issues"])
