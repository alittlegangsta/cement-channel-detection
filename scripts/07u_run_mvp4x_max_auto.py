from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402
from cement_channel.modeling.dependencies import ModelingDependencyError  # noqa: E402
from cement_channel.modeling.mvp4x_max_auto import (  # noqa: E402
    MaxAutoError,
    MaxAutoPaths,
    run_phase,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded MVP-4X max autonomous research phases."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--phase",
        choices=("preflight", "scoring", "intervals", "triage", "morphology", "decision", "all"),
        default="preflight",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths_config = load_paths_config(args.paths_config)
        paths = MaxAutoPaths.from_config(paths_config)
        if args.dry_run:
            print(
                "Dry run: would run MVP-4X max-auto phase "
                f"{args.phase} with reports root {paths.reports}."
            )
            return 0
        output = run_phase(paths, phase=args.phase, overwrite=args.overwrite)
    except (
        ManifestBuildError,
        ModelingDependencyError,
        MaxAutoError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "MVP-4X max-auto phase completed "
        f"phase={args.phase}; "
        f"research_only={output.get('research_only')}; "
        f"not_validated_for_deployment={output.get('not_validated_for_deployment')}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
