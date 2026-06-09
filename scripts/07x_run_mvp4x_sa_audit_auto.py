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
from cement_channel.modeling.mvp4x_max_auto import MaxAutoPaths  # noqa: E402
from cement_channel.modeling.mvp4x_sa_audit import (  # noqa: E402
    SaAuditError,
    run_sa_audit_from_paths,
)

DEFAULT_B80_RUN = "outputs/remote-runs/20260609T030834Z_mvp4x-sa-pilot-auto-b80_4108bd1ded95"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MVP-4X-SA bounded proxy contract and matched screening audit."
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
    parser.add_argument("--snapshot-npz", default=None)
    parser.add_argument("--waveform-npz", default=None)
    parser.add_argument("--tfv2-npz", default=None)
    parser.add_argument("--final-scores-npz", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths_config = load_paths_config(args.paths_config)
        paths = MaxAutoPaths.from_config(paths_config)
        pilot_run_dirs = [Path(item) for item in (args.pilot_run_dir or [DEFAULT_B80_RUN])]
        if args.dry_run:
            print(
                "Dry run: would run MVP-4X-SA audit for "
                f"{len(pilot_run_dirs)} fetched pilot run(s); reports root={paths.reports}."
            )
            return 0
        output = run_sa_audit_from_paths(
            paths=paths,
            pilot_run_dirs=pilot_run_dirs,
            snapshot_npz=args.snapshot_npz,
            waveform_npz=args.waveform_npz,
            tfv2_npz=args.tfv2_npz,
            final_scores_npz=args.final_scores_npz,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        ModelingDependencyError,
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
    expansion_next = output["expansion"]["next_action"]
    print(
        "MVP-4X-SA audit completed "
        f"decision={decision}; "
        f"expansion_next_action={expansion_next}; "
        f"research_only={output.get('research_only')}; "
        f"not_validated_for_deployment={output.get('not_validated_for_deployment')}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
