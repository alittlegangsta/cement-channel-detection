from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

EXPECTED_RECEIVER_COUNT = 13
EXPECTED_SIDE_COUNT = 8
EXPECTED_TIME_SAMPLE_COUNT = 1024
EXPECTED_CAST_AZIMUTH_COUNT = 180

LEGAL_PRESENCE_LABELS = {-1, 0, 1}
LEGAL_SEVERITY_LABELS = {-1, 0, 1, 2, 3}


@dataclass(frozen=True)
class ValidationResult:
    errors: list[str]
    warnings: list[str]
    is_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_manifest_basic(manifest: dict) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(manifest, dict):
        return ValidationResult(
            errors=["manifest must be a JSON object"],
            warnings=[],
            is_valid=False,
        )

    _validate_top_level_fields(manifest, errors)
    wells = manifest.get("wells")
    if not isinstance(wells, list):
        errors.append("missing or invalid top-level field: wells")
        wells = []
    elif not wells:
        errors.append("manifest must contain at least one well")

    manifest_expected_receivers = _manifest_expected_receiver_count(manifest)
    for index, well in enumerate(wells):
        _validate_well_basic(
            well,
            index=index,
            manifest_expected_receivers=manifest_expected_receivers,
            errors=errors,
            warnings=warnings,
        )

    return ValidationResult(errors=errors, warnings=warnings, is_valid=not errors)


def _validate_top_level_fields(manifest: dict[str, Any], errors: list[str]) -> None:
    for field_name in ("schema_version", "data_version"):
        if not manifest.get(field_name):
            errors.append(f"missing top-level field: {field_name}")

    if not (manifest.get("generated_at") or manifest.get("created_at")):
        errors.append("missing top-level field: generated_at or created_at")


def _validate_well_basic(
    well: Any,
    *,
    index: int,
    manifest_expected_receivers: int | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(well, dict):
        errors.append(f"wells[{index}] must be an object")
        return

    well_id = well.get("well_id")
    well_name = str(well_id) if well_id else f"wells[{index}]"
    if not well_id:
        errors.append(f"{well_name}: missing well_id")

    role_counts = _well_role_counts(well)
    if role_counts is None:
        errors.append(f"{well_name}: missing xsi_receiver file list")
        return

    if role_counts.get("cast", 0) < 1:
        warnings.append(f"{well_name}: missing CAST file")
    if role_counts.get("pose", 0) < 1:
        warnings.append(f"{well_name}: missing pose file")

    actual_receivers = role_counts.get("xsi_receiver", 0)
    expected_receivers = _well_expected_receiver_count(well, manifest_expected_receivers)
    if actual_receivers != expected_receivers:
        warnings.append(
            f"{well_name}: XSI receiver count mismatch; "
            f"expected {expected_receivers}, observed {actual_receivers}"
        )
    elif actual_receivers != EXPECTED_RECEIVER_COUNT:
        warnings.append(
            f"{well_name}: XSI receiver count is {actual_receivers}; "
            f"project expected count is {EXPECTED_RECEIVER_COUNT}"
        )


def _manifest_expected_receiver_count(manifest: dict[str, Any]) -> int | None:
    raw_layout = manifest.get("raw_layout")
    if isinstance(raw_layout, dict):
        return _optional_int(raw_layout.get("expected_xsi_receiver_files"))
    return None


def _well_expected_receiver_count(
    well: dict[str, Any],
    manifest_expected_receivers: int | None,
) -> int:
    return (
        _optional_int(well.get("expected_xsi_receiver_files"))
        or manifest_expected_receivers
        or EXPECTED_RECEIVER_COUNT
    )


def _well_role_counts(well: dict[str, Any]) -> dict[str, int] | None:
    files = well.get("files")
    if isinstance(files, list):
        counts: dict[str, int] = {}
        for file_record in files:
            if not isinstance(file_record, dict):
                continue
            file_role = file_record.get("file_role")
            if isinstance(file_role, str):
                counts[file_role] = counts.get(file_role, 0) + 1
        return counts

    counts = well.get("counts")
    if isinstance(counts, dict):
        return {str(key): int(value) for key, value in counts.items()}

    legacy_counts = {
        "cast": _list_count(well.get("cast_files")),
        "pose": _list_count(well.get("pose_files")),
        "xsi_receiver": _list_count(well.get("xsi_receiver_files")),
    }
    if any(value is not None for value in legacy_counts.values()):
        return {key: value or 0 for key, value in legacy_counts.items()}

    return None


def _list_count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, list):
        return len(value)
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
