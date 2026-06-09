from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from cement_channel.remote.runner import REMOTE_PYTHON

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CEMENT_REMOTE_BACKEND"] = "fake"
    env["CEMENT_REMOTE_FAKE_ROOT"] = str(tmp_path / "fake-remote")
    env["CEMENT_REMOTE_RUN_ID"] = "integration-run-001"
    return subprocess.run(
        [sys.executable, "scripts/cement_remote.py", *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_cement_remote_fake_cli_lifecycle(tmp_path: Path) -> None:
    doctor = _run_cli(tmp_path, "doctor")
    assert "fake backend" in doctor.stdout
    assert "systemd_run_user=available" in doctor.stdout

    output_manifest = tmp_path / "stc_apes_pilot_dependencies.json"
    build_manifest = _run_cli(
        tmp_path,
        "build-data-manifest",
        "--purpose",
        "stc-apes-pilot",
        "--output",
        str(output_manifest),
    )
    assert "wrote data dependency manifest" in build_manifest.stdout
    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert manifest["purpose"] == "stc-apes-pilot"
    assert manifest["sync_entries"] == []

    sync_dry_run = _run_cli(tmp_path, "sync-data", "--manifest", str(output_manifest), "--dry-run")
    assert "remote read-only" in sync_dry_run.stdout

    submit = _run_cli(
        tmp_path,
        "--local-fetch-root",
        str(tmp_path / "fetched"),
        "submit",
        "--name",
        "bounded-pilot",
        "--ref",
        COMMIT,
        "--",
        "python",
        "scripts/00_check_env.py",
    )
    assert "scheduler=systemd-run" in submit.stdout
    assert "run_id=integration-run-001" in submit.stdout

    status = _run_cli(tmp_path, "status", "--run-id", "integration-run-001")
    assert json.loads(status.stdout)["status"] == "succeeded"

    logs = _run_cli(tmp_path, "logs", "--run-id", "integration-run-001", "--tail", "20")
    assert "fake backend did not execute remote command" in logs.stdout

    run_manifest = _run_cli(tmp_path, "manifest", "--run-id", "integration-run-001")
    run_manifest_json = json.loads(run_manifest.stdout)
    assert run_manifest_json["git_commit"] == COMMIT
    assert run_manifest_json["scheduler"] == "systemd-run"
    assert run_manifest_json["remote_python"] == REMOTE_PYTHON
    assert run_manifest_json["python_no_user_site"] is True
    assert run_manifest_json["command"][0] == REMOTE_PYTHON

    listed = _run_cli(tmp_path, "list")
    assert "integration-run-001\tsucceeded\tbounded-pilot" in listed.stdout

    fetched_root = tmp_path / "fetched"
    fetch = _run_cli(
        tmp_path,
        "--local-fetch-root",
        str(fetched_root),
        "fetch",
        "--run-id",
        "integration-run-001",
        "--reports-only",
    )
    assert "manifest.json" in fetch.stdout
    assert (fetched_root / "integration-run-001/manifest.json").exists()
