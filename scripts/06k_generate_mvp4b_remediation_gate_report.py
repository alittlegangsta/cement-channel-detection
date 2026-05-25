from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402


class MVP4BRemediationGateError(RuntimeError):
    """Raised when the MVP-4B remediation gate cannot be generated safely."""


MVP4B_REMEDIATION_GATE_VERSION = "mvp4b_remediation_gate_v001"
MIN_REMEDIATION_MARGIN = 0.03
MAX_CANDIDATE_WEIGHT_FRACTION = 0.60
MIN_FOLDS_ABOVE_PERMUTATION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MVP-4B remediation gate report.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--baseline-failure-diagnostics", default=None)
    parser.add_argument("--sample-weight-policy-report", default=None)
    parser.add_argument("--enhanced-feature-report", default=None)
    parser.add_argument("--baseline-remediation-ablation", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        paths = _resolve_paths(config, args)
        for key, path in paths.items():
            if key.startswith("output_"):
                _ensure_path_within(config, path, key="reports", action="write")
            else:
                _ensure_path_within(config, path, key="reports", action="read")
        report = build_remediation_gate_report(paths)
        markdown = format_remediation_gate_markdown(report)
        if not args.dry_run:
            _write_outputs(
                report,
                markdown,
                output_md=paths["output_report_md"],
                output_json=paths["output_report_json"],
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        MVP4BRemediationGateError,
        OSError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "MVP-4B remediation gate "
        f"decision={report['decision']}; "
        f"blocking_issues={len(report['blocking_issues'])}; "
        f"warnings={len(report['warnings'])}; "
        "mvp4c_consideration_allowed="
        f"{report['mvp4c_consideration_allowed']}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {paths['output_report_md']}")
        print(f"Wrote JSON report: {paths['output_report_json']}")
    return 1 if report["decision"] == "no_go" else 0


def build_remediation_gate_report(paths: dict[str, Path]) -> dict[str, Any]:
    failure = _read_json(paths["baseline_failure_diagnostics"])
    weights = _read_json(paths["sample_weight_policy_report"])
    features = _read_json(paths["enhanced_feature_report"])
    ablation = _read_json(paths["baseline_remediation_ablation"])
    blocking: list[str] = []
    warnings: list[str] = []

    weight_summary = _as_dict(weights.get("policy_summary"))
    capped_summary = _as_dict(weight_summary.get("capped_class_balanced_confidence"))
    capped_fraction = _as_float(capped_summary.get("candidate_effective_weight_fraction"))
    if capped_fraction is None or capped_fraction > MAX_CANDIDATE_WEIGHT_FRACTION:
        blocking.append(
            "sample_weight_policy: capped class-balanced candidate effective "
            f"weight fraction invalid: {capped_fraction}."
        )
    if weights.get("errors"):
        blocking.extend(f"sample_weight_policy: {message}" for message in weights["errors"])
    warnings.extend(f"sample_weight_policy: {message}" for message in weights.get("warnings", []))

    if _as_float(features.get("enhanced_transformed_feature_finite_ratio")) != 1.0:
        blocking.append("enhanced_features: transformed features are not fully finite.")
    if features.get("used_label_information_for_features") is not False:
        blocking.append("enhanced_features: feature report does not prove label-free features.")
    if features.get("errors"):
        blocking.extend(f"enhanced_features: {message}" for message in features["errors"])
    warnings.extend(f"enhanced_features: {message}" for message in features.get("warnings", []))

    best = _as_dict(ablation.get("best_non_degenerate_scenario"))
    best_margin = _as_float(best.get("real_minus_permutation_margin"))
    class_balanced_success = bool(
        ablation.get("class_balanced_non_degenerate_above_permutation")
    )
    best_class_balanced = best.get("weight_policy") in {
        "class_balanced_confidence",
        "capped_class_balanced_confidence",
    }
    if not class_balanced_success:
        blocking.append(
            "ablation: no class-balanced non-degenerate baseline met the configured "
            "permutation margin."
        )
    if not best or not best_class_balanced:
        blocking.append("ablation: best non-degenerate scenario is not class-balanced.")
    if best_margin is None or best_margin < MIN_REMEDIATION_MARGIN:
        blocking.append(
            "ablation: best real-minus-permutation margin below remediation gate "
            f"threshold {MIN_REMEDIATION_MARGIN}: {best_margin}."
        )
    if bool(best.get("degenerate_prediction")):
        blocking.append("ablation: best scenario has degenerate predictions.")
    if int(best.get("folds_above_permutation") or 0) < MIN_FOLDS_ABOVE_PERMUTATION:
        blocking.append("ablation: result is not supported by enough depth-block folds.")
    if bool(ablation.get("only_confidence_only_effective")):
        blocking.append("ablation: only confidence-only weighting appears effective.")
    if ablation.get("errors"):
        blocking.extend(f"ablation: {message}" for message in ablation["errors"])
    warnings.extend(f"ablation: {message}" for message in ablation.get("warnings", []))

    no_final_labels = all(
        [
            _data_bool(failure, "no_final_labels"),
            _data_bool(weights, "no_final_labels"),
            _data_bool(features, "no_final_labels"),
            _data_bool(ablation, "no_final_labels"),
        ]
    )
    no_forbidden = all(
        [
            _data_bool(weights, "no_deep_learning"),
            _data_bool(weights, "no_stc"),
            _data_bool(weights, "no_apes"),
            _data_bool(features, "no_deep_learning"),
            _data_bool(features, "no_stc"),
            _data_bool(features, "no_apes"),
            _data_bool(ablation, "no_deep_learning"),
            _data_bool(ablation, "no_stc"),
            _data_bool(ablation, "no_apes"),
            _data_bool(ablation, "no_mvp4c"),
        ]
    )
    if not no_final_labels:
        blocking.append("guardrails: one or more reports do not preserve no_final_labels.")
    if not no_forbidden:
        blocking.append(
            "guardrails: one or more reports permit MVP-4C, STC, APES, or deep learning."
        )

    decision = _decision(blocking, warnings)
    allowed = decision in {"go", "conditional_go"} and not blocking
    return {
        "gate_version": MVP4B_REMEDIATION_GATE_VERSION,
        "stage": "MVP-4B-R",
        "task": "baseline_remediation_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: str(path) for key, path in paths.items() if not key.startswith("output_")},
        "gate_thresholds": {
            "min_real_minus_permutation_margin": MIN_REMEDIATION_MARGIN,
            "max_candidate_weight_fraction": MAX_CANDIDATE_WEIGHT_FRACTION,
            "min_folds_above_permutation": MIN_FOLDS_ABOVE_PERMUTATION,
        },
        "evidence": {
            "previous_no_go_confirmed": bool(failure.get("no_go_confirmed")),
            "previous_no_go_reason_classes": failure.get("no_go_reason_classes", []),
            "capped_candidate_effective_weight_fraction": capped_fraction,
            "enhanced_feature_finite_ratio": features.get(
                "enhanced_transformed_feature_finite_ratio"
            ),
            "best_non_degenerate_scenario": best,
            "class_balanced_non_degenerate_above_permutation": class_balanced_success,
            "ablation_no_go_reasons": ablation.get("no_go_reasons", []),
            "no_final_labels": no_final_labels,
            "no_stc_apes_deep_learning_mvp4c": no_forbidden,
        },
        "blocking_issues": blocking,
        "warnings": warnings,
        "decision": decision,
        "mvp4c_consideration_allowed": allowed,
        "mvp4c_allowed": allowed,
        "next_stage_allowed": (
            "MVP-4C consideration only; no STC/APES/deep learning unless separately approved"
            if allowed
            else None
        ),
        "recommendation": (
            "continue no-go and return to label/feature design review"
            if not allowed
            else "review remediation evidence before any MVP-4C planning"
        ),
        "not_performed": [
            "final label generation",
            "ground truth claim",
            "production training",
            "model weight export",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C implementation",
        ],
    }


def format_remediation_gate_markdown(report: dict[str, Any]) -> str:
    evidence = _as_dict(report.get("evidence"))
    best = _as_dict(evidence.get("best_non_degenerate_scenario"))
    lines = [
        "# MVP-4B-R Remediation Gate Report",
        "",
        "This gate decides whether the previous simple-baseline no-go has been "
        "remediated. It does not authorize STC, APES, deep learning, final "
        "labels, or production modeling.",
        "",
        "## Decision",
        "",
        f"- decision: `{report['decision']}`",
        f"- mvp4c_consideration_allowed: `{report['mvp4c_consideration_allowed']}`",
        f"- recommendation: {report['recommendation']}",
        "",
        "## Best Non-Degenerate Scenario",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- scenario_name: `{best.get('scenario_name')}`",
                f"- model_type: `{best.get('model_type')}`",
                f"- feature_set: `{best.get('feature_set')}`",
                f"- weight_policy: `{best.get('weight_policy')}`",
                f"- balanced_accuracy: {best.get('balanced_accuracy')}",
                f"- permutation_balanced_accuracy: {best.get('permutation_balanced_accuracy')}",
                f"- real_minus_permutation_margin: {best.get('real_minus_permutation_margin')}",
                f"- predicted_positive_rate: {best.get('predicted_positive_rate')}",
                f"- folds_above_permutation: {best.get('folds_above_permutation')}",
            ]
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Blocking Issues", ""])
    lines.extend(_message_lines(report["blocking_issues"]))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report["warnings"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _resolve_paths(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    reports = Path(str(_as_dict(config.get("data")).get("reports", "")))
    if not str(reports):
        raise MVP4BRemediationGateError("data.reports is not configured.")
    return {
        "baseline_failure_diagnostics": Path(
            args.baseline_failure_diagnostics
            or reports
            / "baseline_failure_diagnostics_v001.json"
        ),
        "sample_weight_policy_report": Path(
            args.sample_weight_policy_report or reports / "sample_weight_policy_report_v001.json"
        ),
        "enhanced_feature_report": Path(
            args.enhanced_feature_report or reports / "enhanced_feature_report_v001.json"
        ),
        "baseline_remediation_ablation": Path(
            args.baseline_remediation_ablation
            or reports
            / "baseline_remediation_ablation_v001.json"
        ),
        "output_report_md": Path(
            args.output_report_md or reports / "mvp4b_remediation_gate_report.md"
        ),
        "output_report_json": Path(
            args.output_report_json or reports / "mvp4b_remediation_gate_report.json"
        ),
    }


def _write_outputs(
    report: dict[str, Any],
    markdown: str,
    *,
    output_md: Path,
    output_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MVP4BRemediationGateError(f"Required report does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MVP4BRemediationGateError(f"Report must be a JSON object: {path}")
    return data


def _ensure_path_within(
    config: dict[str, Any],
    path: Path,
    *,
    key: str,
    action: str,
) -> None:
    data = _as_dict(config.get("data"))
    root = Path(str(data.get(key, ""))).resolve()
    if not str(root):
        raise MVP4BRemediationGateError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise MVP4BRemediationGateError(
            f"Refusing to {action} remediation gate path outside data.{key}: {path}"
        ) from exc


def _decision(blocking: list[str], warnings: list[str]) -> str:
    if blocking:
        return "no_go"
    if warnings:
        return "conditional_go"
    return "go"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError):
        return None


def _data_bool(data: dict[str, Any], key: str) -> bool:
    return bool(data.get(key)) is True


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


if __name__ == "__main__":
    raise SystemExit(main())
