from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def test_max_auto_cli_dry_run_resolves_paths(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for name in ("interim", "features", "reports", "labels", "manifests"):
        (data / name).mkdir(parents=True)
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        yaml.safe_dump(
            {
                "data": {
                    name: str(data / name)
                    for name in ("interim", "features", "reports", "labels", "manifests")
                }
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07u_run_mvp4x_max_auto.py",
            "--paths",
            str(paths),
            "--phase",
            "preflight",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Dry run" in result.stdout
