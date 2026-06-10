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
from cement_channel.modeling.mvp4x_ls_mw_auto import (  # noqa: E402
    DEFAULT_PILOT_RUN_DIRS,
    LsMwAutoError,
    run_ls_mw_auto_from_paths,
)
from cement_channel.modeling.mvp4x_max_auto import MaxAutoPaths  # noqa: E402
from cement_channel.modeling.mvp4x_sa_audit import SaAuditError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MVP-4X label-semantics and multiwell readiness review."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--pilot-run-dir",
        action="append",
        default=None,
        help="Fetched cement-remote run directory. May be supplied multiple times.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths_config = load_paths_config(args.paths_config)
        paths = MaxAutoPaths.from_config(paths_config)
        pilot_run_dirs = [Path(item) for item in (args.pilot_run_dir or DEFAULT_PILOT_RUN_DIRS)]
        if args.dry_run:
            print(
                "Dry run: would run MVP-4X-LS-MW auto review for "
                f"{len(pilot_run_dirs)} fetched pilot run(s); reports root={paths.reports}."
            )
            return 0
        output = run_ls_mw_auto_from_paths(
            paths=paths,
            project_root=PROJECT_ROOT,
            pilot_run_dirs=pilot_run_dirs,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        ModelingDependencyError,
        LsMwAutoError,
        SaAuditError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    decision = output["decision"]["decision"]
    print(
        "MVP-4X-LS-MW auto review completed "
        f"decision={decision}; "
        f"research_only={output.get('research_only')}; "
        f"not_validated_for_deployment={output.get('not_validated_for_deployment')}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
