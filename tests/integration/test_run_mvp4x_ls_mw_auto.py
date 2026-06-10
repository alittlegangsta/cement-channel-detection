from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def test_run_mvp4x_ls_mw_auto_dry_run(tmp_path: Path) -> None:
    for name in ("interim", "features", "reports", "manifests"):
        (tmp_path / name).mkdir()
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "interim": str(tmp_path / "interim"),
                    "features": str(tmp_path / "features"),
                    "reports": str(tmp_path / "reports"),
                    "manifests": str(tmp_path / "manifests"),
                }
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07y_run_mvp4x_ls_mw_auto.py",
            "--paths",
            str(paths_config),
            "--pilot-run-dir",
            str(tmp_path / "unused-run"),
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Dry run: would run MVP-4X-LS-MW auto review" in result.stdout
