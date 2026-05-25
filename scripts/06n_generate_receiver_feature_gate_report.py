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
from cement_channel.features.receiver_feature_schema import (  # noqa: E402
    MVP4B_RECEIVER_GATE_VERSION,
    load_receiver_feature_config,
)


class ReceiverFeatureGateError(RuntimeError):
    """Raised when the receiver feature remediation gate cannot run safely."""


RECEIVER_FEATURE_SETS = {
    "receiver_derived_only",
    "side_plus_receiver",
    "receiver_late_over_early_only",
    "receiver_far_near_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate receiver feature remediation gate.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--receiver-config",
        default="configs/mvp4b_receiver_features.example.yaml",
    )
    parser.add_argument("--receiver-feature-report", default=None)
    parser.add_argument("--receiver-feature-ablation", default=None)
    parser.add_argument("--mvp4b-remediation-gate-report", default=None)
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
        receiver_config = load_receiver_feature_config(args.receiver_config)
        report = build_receiver_feature_gate_report(paths, receiver_config=receiver_config)
        markdown = format_receiver_feature_gate_markdown(report)
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
        ReceiverFeatureGateError,
        OSError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Receiver feature gate "
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


def build_receiver_feature_gate_report(
    paths: dict[str, Path],
    *,
    receiver_config: Any,
) -> dict[str, Any]:
    feature_report = _read_json(paths["receiver_feature_report"])
    ablation = _read_json(paths["receiver_feature_ablation"])
    remediation_gate = _read_json(paths["mvp4b_remediation_gate_report"])
    blocking: list[str] = []
    warnings: list[str] = []

    receiver_finite_ratio = _as_float(
        feature_report.get("finite_ratio", {}).get("transformed_receiver_features")
    )
    if receiver_finite_ratio != 1.0:
        blocking.append("receiver_features: transformed receiver features are not fully finite.")
    if feature_report.get("used_label_information_for_feature_construction") is not False:
        blocking.append("receiver_features: feature construction may have used labels.")
    if feature_report.get("errors"):
        blocking.extend(f"receiver_features: {message}" for message in feature_report["errors"])
    warnings.extend(
        f"receiver_features: {message}" for message in feature_report.get("warnings", [])
    )

    best_receiver = _best_receiver_row(ablation.get("summary_rows", []))
    best_margin = _as_float(_as_dict(best_receiver).get("real_minus_permutation_margin"))
    predicted_positive_rate = _as_float(_as_dict(best_receiver).get("predicted_positive_rate"))
    folds_above = int(_as_dict(best_receiver).get("folds_above_permutation") or 0)
    receiver_passes = bool(
        best_receiver
        and best_margin is not None
        and best_margin >= receiver_config.required_margin_over_permutation
        and not bool(best_receiver.get("degenerate_prediction"))
        and folds_above >= receiver_config.required_folds_above_permutation
        and not bool(best_receiver.get("leakage_suspected"))
        and predicted_positive_rate is not None
        and receiver_config.min_degenerate_positive_rate
        < predicted_positive_rate
        < receiver_config.max_degenerate_positive_rate
    )
    if not best_receiver:
        blocking.append("ablation: no receiver-derived non-degenerate result was available.")
    if best_margin is None or best_margin < receiver_config.required_margin_over_permutation:
        blocking.append(
            "ablation: best receiver-derived real-minus-permutation margin below "
            f"{receiver_config.required_margin_over_permutation}: {best_margin}."
        )
    if predicted_positive_rate is None or not (
        receiver_config.min_degenerate_positive_rate
        < predicted_positive_rate
        < receiver_config.max_degenerate_positive_rate
    ):
        blocking.append("ablation: best receiver-derived prediction rate is degenerate.")
    if folds_above < receiver_config.required_folds_above_permutation:
        blocking.append("ablation: best receiver-derived result lacks fold support.")
    if bool(_as_dict(best_receiver).get("leakage_suspected")):
        blocking.append("ablation: leakage suspected for receiver-derived result.")
    if ablation.get("errors"):
        blocking.extend(f"ablation: {message}" for message in ablation["errors"])
    warnings.extend(f"ablation: {message}" for message in ablation.get("warnings", []))

    no_final_labels = all(
        [
            _data_bool(feature_report, "no_final_labels"),
            _data_bool(ablation, "no_final_labels"),
        ]
    )
    no_forbidden = all(
        [
            _data_bool(feature_report, "no_stc"),
            _data_bool(feature_report, "no_apes"),
            _data_bool(feature_report, "no_deep_learning"),
            _data_bool(feature_report, "no_mvp4c"),
            _data_bool(ablation, "no_stc"),
            _data_bool(ablation, "no_apes"),
            _data_bool(ablation, "no_deep_learning"),
            _data_bool(ablation, "no_mvp4c"),
        ]
    )
    if not no_final_labels:
        blocking.append("guardrails: reports do not preserve no_final_labels.")
    if not no_forbidden:
        blocking.append("guardrails: reports permit STC/APES/deep learning/MVP-4C.")

    decision = "no_go" if blocking else ("conditional_go" if warnings else "go")
    allowed = decision in {"go", "conditional_go"} and receiver_passes and not blocking
    return {
        "gate_version": MVP4B_RECEIVER_GATE_VERSION,
        "stage": "MVP-4B-R2",
        "task": "receiver_feature_remediation_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: str(path) for key, path in paths.items() if not key.startswith("output_")},
        "previous_remediation_gate_decision": remediation_gate.get("decision"),
        "thresholds": {
            "required_margin_over_permutation": receiver_config.required_margin_over_permutation,
            "required_folds_above_permutation": receiver_config.required_folds_above_permutation,
            "min_degenerate_positive_rate": receiver_config.min_degenerate_positive_rate,
            "max_degenerate_positive_rate": receiver_config.max_degenerate_positive_rate,
        },
        "evidence": {
            "receiver_feature_finite_ratio": feature_report.get("finite_ratio"),
            "best_receiver_derived_result": best_receiver,
            "receiver_feature_ablation_passes": receiver_passes,
            "no_final_labels": no_final_labels,
            "no_stc_apes_deep_learning_mvp4c": no_forbidden,
        },
        "blocking_issues": blocking,
        "warnings": warnings,
        "decision": decision,
        "mvp4c_consideration_allowed": allowed,
        "mvp4c_allowed": allowed,
        "recommendation": (
            "receiver-derived features remain below gate; return to label refinement "
            "or controlled time-frequency feature sanity"
            if not allowed
            else "review receiver-derived evidence before any MVP-4C planning"
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


def format_receiver_feature_gate_markdown(report: dict[str, Any]) -> str:
    best = _as_dict(report["evidence"].get("best_receiver_derived_result"))
    lines = [
        "# MVP-4B-R2 Receiver Feature Gate Report",
        "",
        "This gate decides whether receiver-derived XSI features remediate the "
        "MVP-4B no-go. It does not authorize STC, APES, deep learning, final "
        "labels, or production modeling.",
        "",
        f"- decision: `{report['decision']}`",
        f"- mvp4c_consideration_allowed: `{report['mvp4c_consideration_allowed']}`",
        f"- recommendation: {report['recommendation']}",
        "",
        "## Best Receiver-Derived Result",
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


def _best_receiver_row(rows: Any) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    candidates = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("feature_set") in RECEIVER_FEATURE_SETS
        and not bool(row.get("degenerate_prediction"))
        and _as_float(row.get("real_minus_permutation_margin")) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["real_minus_permutation_margin"]))


def _resolve_paths(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    reports = Path(str(_as_dict(config.get("data")).get("reports", "")))
    if not str(reports):
        raise ReceiverFeatureGateError("data.reports is not configured.")
    return {
        "receiver_feature_report": Path(
            args.receiver_feature_report or reports / "receiver_derived_feature_report_v001.json"
        ),
        "receiver_feature_ablation": Path(
            args.receiver_feature_ablation or reports / "receiver_feature_ablation_v001.json"
        ),
        "mvp4b_remediation_gate_report": Path(
            args.mvp4b_remediation_gate_report
            or reports
            / "mvp4b_remediation_gate_report.json"
        ),
        "output_report_md": Path(
            args.output_report_md or reports / "receiver_feature_gate_report.md"
        ),
        "output_report_json": Path(
            args.output_report_json or reports / "receiver_feature_gate_report.json"
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
        raise ReceiverFeatureGateError(f"Required report does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ReceiverFeatureGateError(f"Report must be a JSON object: {path}")
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
        raise ReceiverFeatureGateError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ReceiverFeatureGateError(
            f"Refusing to {action} receiver feature gate path outside data.{key}: {path}"
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
