from __future__ import annotations

import subprocess
import sys


def test_stratified_design_cli_dry_run(tmp_path) -> None:
    for name in ("interim", "features", "reports", "labels", "manifests"):
        (tmp_path / name).mkdir()
    paths_config = tmp_path / "paths.local.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  interim: {tmp_path / 'interim'}",
                f"  features: {tmp_path / 'features'}",
                f"  reports: {tmp_path / 'reports'}",
                f"  labels: {tmp_path / 'labels'}",
                f"  manifests: {tmp_path / 'manifests'}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07j_build_mvp4x_stratified_design.py",
            "--paths",
            str(paths_config),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Dry run: would build MVP-4X stratified design" in result.stdout
