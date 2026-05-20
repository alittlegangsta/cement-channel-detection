from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.alignment.depth_audit import DEFAULT_MAX_DEPTH_SAMPLES  # noqa: E402
from cement_channel.alignment.depth_reader import (  # noqa: E402
    read_depth_only_from_configs,
    write_depth_only_outputs,
)
from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402


class DepthOnlyReaderCliError(RuntimeError):
    """Raised when controlled depth-only reading cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read controlled depth-only and pose arrays.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--mapping", default="configs/raw_variable_mapping.yaml")
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--output-summary-json", default=None)
    parser.add_argument("--max-depth-samples", type=int, default=DEFAULT_MAX_DEPTH_SAMPLES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        output_npz = _resolve_interim_path(config, args.output_npz, "depth_only_v001.npz")
        output_summary = _resolve_interim_path(
            config,
            args.output_summary_json,
            "depth_only_summary_v001.json",
        )
        _ensure_interim_output(config, output_npz)
        _ensure_interim_output(config, output_summary)
        result, arrays = read_depth_only_from_configs(
            args.paths_config,
            args.mapping,
            max_depth_samples=args.max_depth_samples,
        )
        if not args.dry_run:
            write_depth_only_outputs(
                result,
                arrays,
                output_npz=output_npz,
                output_summary_json=output_summary,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        DepthOnlyReaderCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Depth-only read "
        f"arrays={len(arrays)}; "
        f"errors={len(result.errors)}; "
        f"warnings={len(result.warnings)}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote NPZ: {output_npz}")
        print(f"Wrote summary JSON: {output_summary}")
    return 1 if result.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise DepthOnlyReaderCliError("data.interim is not configured; pass an explicit output path.")


def _ensure_interim_output(config: dict[str, Any], output_path: Path) -> None:
    data = _as_dict(config.get("data"))
    interim = Path(str(data.get("interim", ""))).resolve()
    if not str(interim):
        raise DepthOnlyReaderCliError("data.interim is not configured.")
    try:
        output_path.resolve().relative_to(interim)
    except ValueError as exc:
        raise DepthOnlyReaderCliError(
            f"Refusing to write depth-only output outside data.interim: {output_path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
