from __future__ import annotations

import argparse
import csv
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
from cement_channel.features.receiver_feature_schema import (  # noqa: E402
    MVP4B_RECEIVER_ABLATION_VERSION,
    load_receiver_feature_config,
)
from cement_channel.training.baseline_schema import load_baseline_config  # noqa: E402
from cement_channel.training.simple_baseline import (  # noqa: E402
    RemediationAblationScenario,
    run_baseline_remediation_ablation,
)


class ReceiverFeatureAblationCliError(RuntimeError):
    """Raised when receiver feature ablation cannot run safely."""


RECEIVER_FEATURE_SCENARIOS = (
    RemediationAblationScenario(
        "side_enhanced_include_disagreement",
        "side_level_enhanced_only",
        "capped_class_balanced_confidence",
        "include",
        0.5,
    ),
    RemediationAblationScenario(
        "side_enhanced_exclude_disagreement",
        "side_level_enhanced_only",
        "capped_class_balanced_confidence",
        "exclude",
        0.5,
    ),
    RemediationAblationScenario(
        "receiver_only_include_disagreement",
        "receiver_derived_only",
        "capped_class_balanced_confidence",
        "include",
        0.5,
    ),
    RemediationAblationScenario(
        "receiver_only_exclude_disagreement",
        "receiver_derived_only",
        "capped_class_balanced_confidence",
        "exclude",
        0.5,
    ),
    RemediationAblationScenario(
        "side_plus_receiver_include_disagreement",
        "side_plus_receiver",
        "capped_class_balanced_confidence",
        "include",
        0.5,
    ),
    RemediationAblationScenario(
        "side_plus_receiver_exclude_disagreement",
        "side_plus_receiver",
        "capped_class_balanced_confidence",
        "exclude",
        0.5,
    ),
    RemediationAblationScenario(
        "receiver_late_ratio_include_disagreement",
        "receiver_late_over_early_only",
        "capped_class_balanced_confidence",
        "include",
        0.5,
    ),
    RemediationAblationScenario(
        "receiver_late_ratio_exclude_disagreement",
        "receiver_late_over_early_only",
        "capped_class_balanced_confidence",
        "exclude",
        0.5,
    ),
    RemediationAblationScenario(
        "receiver_far_near_include_disagreement",
        "receiver_far_near_only",
        "capped_class_balanced_confidence",
        "include",
        0.5,
    ),
    RemediationAblationScenario(
        "receiver_far_near_exclude_disagreement",
        "receiver_far_near_only",
        "capped_class_balanced_confidence",
        "exclude",
        0.5,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MVP-4B-R2 receiver feature ablations.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--baseline-config",
        default="configs/mvp4b_simple_baseline.example.yaml",
    )
    parser.add_argument(
        "--receiver-config",
        default="configs/mvp4b_receiver_features.example.yaml",
    )
    parser.add_argument("--sample-table-npz", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths_config = load_paths_config(args.paths_config)
        sample_npz = _resolve_interim_path(
            paths_config,
            args.sample_table_npz,
            "baseline_sample_table_receiver_enhanced_v001.npz",
        )
        output_md = _resolve_report_path(
            paths_config,
            args.output_report_md,
            "receiver_feature_ablation_v001.md",
        )
        output_json = _resolve_report_path(
            paths_config,
            args.output_report_json,
            "receiver_feature_ablation_v001.json",
        )
        output_csv = _resolve_report_path(
            paths_config,
            args.output_csv,
            "receiver_feature_ablation_v001.csv",
        )
        _ensure_path_within(paths_config, sample_npz, key="interim", action="read")
        _ensure_path_within(paths_config, output_md, key="reports", action="write")
        _ensure_path_within(paths_config, output_json, key="reports", action="write")
        _ensure_path_within(paths_config, output_csv, key="reports", action="write")
        baseline_config = load_baseline_config(args.baseline_config)
        receiver_config = load_receiver_feature_config(args.receiver_config)
        arrays = _load_npz(sample_npz)
        base_report = run_baseline_remediation_ablation(
            arrays=arrays,
            baseline_config=baseline_config,
            inputs={
                "sample_table_npz": str(sample_npz),
                "baseline_config_path": str(args.baseline_config),
                "receiver_config_path": str(args.receiver_config),
            },
            scenarios=RECEIVER_FEATURE_SCENARIOS,
        )
        report = _receiver_ablation_report(
            base_report.to_dict(),
            required_margin=receiver_config.required_margin_over_permutation,
            required_folds=receiver_config.required_folds_above_permutation,
        )
        if not args.dry_run:
            _write_outputs(
                report,
                output_md=output_md,
                output_json=output_json,
                output_csv=output_csv,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        ReceiverFeatureAblationCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Receiver feature ablation "
        f"errors={len(report['errors'])}; "
        f"warnings={len(report['warnings'])}; "
        f"scenarios={report['scenario_count']}; "
        f"best_margin={report['best_non_degenerate_margin']}; "
        f"decision_suggestion={report['decision_suggestion']}."
    )
    if args.dry_run:
        print("Dry run: no Markdown/JSON/CSV outputs written.")
    else:
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
        print(f"Wrote CSV report: {output_csv}")
    return 1 if report["errors"] else 0


def _receiver_ablation_report(
    base_report: dict[str, Any],
    *,
    required_margin: float,
    required_folds: int,
) -> dict[str, Any]:
    best = _as_dict(base_report.get("best_non_degenerate_scenario"))
    best_margin = _as_float(base_report.get("best_non_degenerate_margin"))
    best_is_receiver = str(best.get("feature_set", "")).startswith("receiver") or (
        best.get("feature_set") == "side_plus_receiver"
    )
    passes = (
        best_margin is not None
        and best_margin >= required_margin
        and best_is_receiver
        and not bool(best.get("degenerate_prediction"))
        and int(best.get("folds_above_permutation") or 0) >= required_folds
        and not bool(best.get("leakage_suspected"))
    )
    report = dict(base_report)
    report["report_version"] = MVP4B_RECEIVER_ABLATION_VERSION
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["required_margin_over_permutation"] = required_margin
    report["required_folds_above_permutation"] = required_folds
    report["feature_sets_compared"] = sorted(
        {str(row["feature_set"]) for row in report.get("summary_rows", [])}
    )
    report["receiver_feature_ablation_passes"] = passes
    report["decision_suggestion"] = "conditional_go_candidate" if passes else "no_go"
    report["mvp4c_allowed"] = False
    report["no_mvp4c"] = True
    report["not_performed"] = list(
        dict.fromkeys(
            [
                *report.get("not_performed", []),
                "MVP-4C implementation",
                "STC",
                "APES",
                "deep learning",
                "final label generation",
            ]
        )
    )
    if not passes:
        reasons = list(report.get("no_go_reasons", []))
        if best_margin is None or best_margin < required_margin:
            reasons.append("receiver_margin_below_threshold")
        if not best_is_receiver:
            reasons.append("best_result_not_receiver_feature_based")
        report["no_go_reasons"] = sorted(set(reasons))
    return report


def _write_outputs(
    report: dict[str, Any],
    *,
    output_md: Path,
    output_json: Path,
    output_csv: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    _ensure_can_write(output_csv, overwrite=overwrite)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(_format_markdown(report), encoding="utf-8")
    _write_csv(report.get("summary_rows", []), output_csv)


def _format_markdown(report: dict[str, Any]) -> str:
    best = _as_dict(report.get("best_non_degenerate_scenario"))
    lines = [
        "# MVP-4B-R2 Receiver Feature Ablation",
        "",
        "These are weak-label sanity ablations, not formal model performance.",
        "",
        f"- report_version: `{report['report_version']}`",
        f"- decision_suggestion: `{report['decision_suggestion']}`",
        f"- required_margin_over_permutation: {report['required_margin_over_permutation']}",
        f"- no_mvp4c: `{report['no_mvp4c']}`",
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
                f"- balanced_accuracy: {best.get('balanced_accuracy')}",
                f"- permutation_balanced_accuracy: {best.get('permutation_balanced_accuracy')}",
                f"- real_minus_permutation_margin: {best.get('real_minus_permutation_margin')}",
                f"- predicted_positive_rate: {best.get('predicted_positive_rate')}",
            ]
        )
    else:
        lines.append("- none")
    lines.extend(["", "## No-Go Reasons", ""])
    lines.extend(_message_lines(report.get("no_go_reasons", [])))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.get("not_performed", [])))
    lines.append("")
    return "\n".join(lines)


def _write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise ReceiverFeatureAblationCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise ReceiverFeatureAblationCliError("data.reports is not configured.")


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
        raise ReceiverFeatureAblationCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ReceiverFeatureAblationCliError(
            f"Refusing to {action} receiver ablation path outside data.{key}: {path}"
        ) from exc


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


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


if __name__ == "__main__":
    raise SystemExit(main())
