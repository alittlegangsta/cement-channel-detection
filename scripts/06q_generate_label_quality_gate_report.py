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
from cement_channel.labels.label_quality_schema import (  # noqa: E402
    MVP4B_LABEL_QUALITY_GATE_VERSION,
    load_label_quality_config,
)


class LabelQualityGateError(RuntimeError):
    """Raised when the label-quality gate cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MVP-4B-R3 label-quality gate.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--label-quality-config",
        default="configs/mvp4b_label_quality_subsets.example.yaml",
    )
    parser.add_argument("--label-quality-subsets-report", default=None)
    parser.add_argument("--subset-feature-audit-report", default=None)
    parser.add_argument("--receiver-feature-gate-report", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths_config = load_paths_config(args.paths_config)
        paths = _resolve_paths(paths_config, args)
        for key, path in paths.items():
            if key.startswith("output_"):
                _ensure_path_within(paths_config, path, key="reports", action="write")
            else:
                _ensure_path_within(paths_config, path, key="reports", action="read")
        config = load_label_quality_config(args.label_quality_config)
        report = build_label_quality_gate_report(paths, config=config)
        markdown = format_label_quality_gate_markdown(report)
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
        LabelQualityGateError,
        OSError,
        json.JSONDecodeError,
        ValueError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Label-quality gate "
        f"decision={report['decision']}; "
        f"blocking_issues={len(report['blocking_issues'])}; "
        f"warnings={len(report['warnings'])}; "
        "controlled_time_frequency_sanity_allowed="
        f"{report['controlled_time_frequency_sanity_allowed']}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {paths['output_report_md']}")
        print(f"Wrote JSON report: {paths['output_report_json']}")
    return 1 if report["decision"] == "no_go" else 0


def build_label_quality_gate_report(
    paths: dict[str, Path],
    *,
    config: Any,
) -> dict[str, Any]:
    subsets = _read_json(paths["label_quality_subsets_report"])
    audit = _read_json(paths["subset_feature_audit_report"])
    receiver_gate = _read_json(paths["receiver_feature_gate_report"])
    blocking: list[str] = []
    warnings: list[str] = []

    strong_count = int(
        _as_dict(subsets.get("subset_counts", {}))
        .get("quality_strong_positive", {})
        .get("sample_count", 0)
    )
    clear_count = int(
        _as_dict(subsets.get("subset_counts", {}))
        .get("quality_clear_negative", {})
        .get("sample_count", 0)
    )
    if strong_count < config.quality_policy.min_subset_samples_per_class:
        blocking.append("quality_strong_positive subset is too small.")
    if clear_count < config.quality_policy.min_subset_samples_per_class:
        blocking.append("quality_clear_negative subset is too small.")
    if subsets.get("errors"):
        blocking.extend(f"subsets: {message}" for message in subsets["errors"])
    warnings.extend(f"subsets: {message}" for message in subsets.get("warnings", []))

    enhancement = _as_dict(audit.get("signal_enhancement"))
    label_noise_likely = bool(audit.get("label_noise_likely"))
    quality_best = _as_float(enhancement.get("quality_subset_best_abs_effect_size"))
    quality_delta = _as_float(enhancement.get("quality_minus_all_delta"))
    review_sensitivity = _as_dict(audit.get("review_exclusion_sensitivity"))
    review_flip = bool(review_sensitivity.get("result_flip_exceeds_threshold"))
    if audit.get("errors"):
        blocking.extend(f"audit: {message}" for message in audit["errors"])
    warnings.extend(f"audit: {message}" for message in audit.get("warnings", []))
    if review_flip:
        blocking.append("review exclusion around ~5700 ft causes result-flip sensitivity.")

    enhanced_enough = (
        label_noise_likely
        and quality_best is not None
        and quality_best >= config.gate.strong_signal_effect_size_threshold
        and quality_delta is not None
        and quality_delta >= config.gate.signal_enhancement_effect_size_delta
    )
    if not enhanced_enough:
        blocking.append(
            "label-quality subsets did not clearly enhance feature separation enough "
            "to justify controlled time-frequency sanity."
        )

    no_final_labels = all(
        [
            _data_bool(subsets, "no_final_labels"),
            _data_bool(audit, "no_final_labels"),
        ]
    )
    no_forbidden = all(
        [
            _data_bool(subsets, "no_stc"),
            _data_bool(subsets, "no_apes"),
            _data_bool(subsets, "no_deep_learning"),
            _data_bool(subsets, "no_mvp4c"),
            _data_bool(audit, "no_stc"),
            _data_bool(audit, "no_apes"),
            _data_bool(audit, "no_deep_learning"),
            _data_bool(audit, "no_mvp4c"),
        ]
    )
    if not no_final_labels:
        blocking.append("guardrails: reports do not preserve no_final_labels.")
    if not no_forbidden:
        blocking.append("guardrails: reports permit STC/APES/deep learning/MVP-4C.")

    decision = "no_go" if blocking else ("conditional_go" if warnings else "go")
    controlled_allowed = decision in {"go", "conditional_go"} and enhanced_enough
    return {
        "gate_version": MVP4B_LABEL_QUALITY_GATE_VERSION,
        "stage": "MVP-4B-R3",
        "task": "label_quality_subset_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: str(path) for key, path in paths.items() if not key.startswith("output_")},
        "previous_receiver_feature_gate_decision": receiver_gate.get("decision"),
        "thresholds": {
            "min_subset_samples_per_class": config.quality_policy.min_subset_samples_per_class,
            "signal_enhancement_effect_size_delta": (
                config.gate.signal_enhancement_effect_size_delta
            ),
            "strong_signal_effect_size_threshold": config.gate.strong_signal_effect_size_threshold,
            "max_result_flip_fraction_from_review_exclusion": (
                config.gate.max_result_flip_fraction_from_review_exclusion
            ),
        },
        "evidence": {
            "quality_strong_positive_count": strong_count,
            "quality_clear_negative_count": clear_count,
            "signal_enhancement": enhancement,
            "review_exclusion_sensitivity": review_sensitivity,
            "label_noise_likely": label_noise_likely,
            "no_final_labels": no_final_labels,
            "no_stc_apes_deep_learning_mvp4c": no_forbidden,
        },
        "blocking_issues": blocking,
        "warnings": warnings,
        "decision": decision,
        "controlled_time_frequency_sanity_allowed": controlled_allowed,
        "mvp4c_allowed": False,
        "mvp4c_consideration_allowed": False,
        "recommendation": (
            "label-quality subsets improved separation; consider controlled "
            "time-frequency sanity only, still not MVP-4C/STC/APES/deep learning"
            if controlled_allowed
            else "continue no-go; return to label definition review or manual annotation "
            "before adding more modeling complexity"
        ),
        "not_performed": [
            "final label generation",
            "ground truth claim",
            "model training",
            "production inference",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C implementation",
        ],
    }


def format_label_quality_gate_markdown(report: dict[str, Any]) -> str:
    evidence = _as_dict(report.get("evidence"))
    enhancement = _as_dict(evidence.get("signal_enhancement"))
    lines = [
        "# MVP-4B-R3 Label-Quality Gate Report",
        "",
        "This gate evaluates whether cleaner weak-label subsets explain the MVP-4B "
        "no-go. It does not authorize final labels, STC, APES, deep learning, or MVP-4C.",
        "",
        f"- decision: `{report['decision']}`",
        "- controlled_time_frequency_sanity_allowed: "
        f"`{report['controlled_time_frequency_sanity_allowed']}`",
        f"- mvp4c_allowed: `{report['mvp4c_allowed']}`",
        f"- recommendation: {report['recommendation']}",
        "",
        "## Evidence",
        "",
        f"- quality_strong_positive_count: {evidence.get('quality_strong_positive_count')}",
        f"- quality_clear_negative_count: {evidence.get('quality_clear_negative_count')}",
        f"- label_noise_likely: `{evidence.get('label_noise_likely')}`",
        f"- quality_subset_best_abs_effect_size: "
        f"{enhancement.get('quality_subset_best_abs_effect_size')}",
        f"- quality_minus_all_delta: {enhancement.get('quality_minus_all_delta')}",
        "",
        "## Blocking Issues",
        "",
    ]
    lines.extend(_message_lines(report["blocking_issues"]))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report["warnings"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _resolve_paths(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    return {
        "label_quality_subsets_report": _resolve_report_path(
            config,
            args.label_quality_subsets_report,
            "label_quality_subsets_report_v001.json",
        ),
        "subset_feature_audit_report": _resolve_report_path(
            config,
            args.subset_feature_audit_report,
            "subset_feature_separation_audit_v001.json",
        ),
        "receiver_feature_gate_report": _resolve_report_path(
            config,
            args.receiver_feature_gate_report,
            "receiver_feature_gate_report.json",
        ),
        "output_report_md": _resolve_report_path(
            config,
            args.output_report_md,
            "label_quality_gate_report.md",
        ),
        "output_report_json": _resolve_report_path(
            config,
            args.output_report_json,
            "label_quality_gate_report.json",
        ),
    }


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise LabelQualityGateError("data.reports is not configured.")


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
        raise LabelQualityGateError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise LabelQualityGateError(
            f"Refusing to {action} label-quality gate path outside data.{key}: {path}"
        ) from exc


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
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(markdown, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _data_bool(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).lower() == "true"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


if __name__ == "__main__":
    raise SystemExit(main())
