from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_mvp4x_sa_pilot_auto_dry_run(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    for name in ("raw", "interim", "reports"):
        (data_root / name).mkdir(parents=True)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "schema_version: schema_v001",
                "data:",
                f"  raw: {data_root / 'raw'}",
                f"  interim: {data_root / 'interim'}",
                f"  reports: {data_root / 'reports'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07w_run_mvp4x_sa_pilot_auto.py",
            "--paths",
            str(paths_config),
            "--interval-count",
            "80",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Dry run: would run MVP-4X-SA bounded pilot" in result.stdout
    assert "interval_count=80" in result.stdout
