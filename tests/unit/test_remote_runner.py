from __future__ import annotations

import json
from pathlib import Path

import pytest

from cement_channel.remote.runner import (
    FakeSshBackend,
    RemoteConfig,
    RemoteRunner,
    RemoteRunnerError,
    build_data_dependency_manifest,
)

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_build_stc_apes_pilot_dependency_manifest(tmp_path: Path) -> None:
    output = tmp_path / "stc_apes_pilot_dependencies.json"
    config = RemoteConfig(local_repo_root=Path.cwd(), local_data_root=tmp_path / "data")

    manifest = build_data_dependency_manifest(
        purpose="stc-apes-pilot",
        output_path=output,
        config=config,
    )

    assert output.exists()
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == manifest
    assert manifest["purpose"] == "stc-apes-pilot"
    assert manifest["sync_entries"] == []
    assert manifest["remote_dependency_checks"][0]["access"] == "read_only"
    assert manifest["no_full_well_stc"] is True
    assert manifest["no_full_well_apes"] is True
    assert manifest["no_deep_learning"] is True
    assert "deep learning" in manifest["forbidden_operations"]


def test_fake_backend_submit_writes_required_run_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CEMENT_REMOTE_RUN_ID", "unit-run-001")
    config = RemoteConfig(local_fetch_root=tmp_path / "fetched")
    runner = RemoteRunner(config, FakeSshBackend(tmp_path / "fake-remote"))

    result = runner.submit(
        name="bounded-pilot",
        ref=COMMIT,
        command=["python", "scripts/00_check_env.py"],
    )

    assert result.returncode == 0
    run_dir = tmp_path / "fake-remote/home/xiaoj/cement-channel-runs/unit-run-001"
    required = [
        "manifest.json",
        "status.json",
        "stdout.log",
        "stderr.log",
        "command.sh",
        "environment.txt",
        "git_commit.txt",
        "outputs.json",
        "DONE",
    ]
    for filename in required:
        assert (run_dir / filename).exists(), filename

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert manifest["scheduler"] == "systemd-run"
    assert manifest["git_commit"] == COMMIT


@pytest.mark.parametrize(
    "command",
    [
        ["sudo", "python", "scripts/00_check_env.py"],
        ["rm", "-rf", "/home/xiaoj/cement-channel-data"],
        ["rsync", "-av", "--delete", "a", "b"],
        ["python", "scripts/07_train_xsi_only.py"],
        ["python", "scripts/run_stc.py"],
    ],
)
def test_submit_rejects_unsafe_commands(
    command: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CEMENT_REMOTE_RUN_ID", "unsafe-run-001")
    runner = RemoteRunner(RemoteConfig(), FakeSshBackend(tmp_path / "fake-remote"))

    with pytest.raises(RemoteRunnerError):
        runner.submit(name="unsafe", ref=COMMIT, command=command)


def test_sync_data_rejects_large_or_raw_payloads(tmp_path: Path) -> None:
    local_mat = tmp_path / "raw.mat"
    local_mat.write_text("not real data", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sync_entries": [
                    {
                        "local_path": str(local_mat),
                        "remote_path": "/home/xiaoj/cement-channel-data/raw/raw.mat",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = RemoteRunner(RemoteConfig(), FakeSshBackend(tmp_path / "fake-remote"))

    with pytest.raises(RemoteRunnerError):
        runner.sync_data(manifest, dry_run=True)


def test_sync_data_copies_safe_payload_with_fake_backend(tmp_path: Path) -> None:
    local_json = tmp_path / "pilot_manifest.json"
    local_json.write_text('{"ok": true}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sync_entries": [
                    {
                        "local_path": str(local_json),
                        "remote_path": (
                            "/home/xiaoj/cement-channel-runs/_manifests/"
                            "pilot_manifest.json"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = RemoteRunner(RemoteConfig(), FakeSshBackend(tmp_path / "fake-remote"))

    result = runner.sync_data(manifest, dry_run=False)

    assert result.returncode == 0
    copied = (
        tmp_path
        / "fake-remote/home/xiaoj/cement-channel-runs/_manifests/pilot_manifest.json"
    )
    assert copied.read_text(encoding="utf-8") == '{"ok": true}\n'


def test_wait_cancel_and_fetch_policies_use_fake_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CEMENT_REMOTE_RUN_ID", "policy-run-001")
    config = RemoteConfig(local_fetch_root=tmp_path / "fetched")
    runner = RemoteRunner(config, FakeSshBackend(tmp_path / "fake-remote"))
    runner.submit(
        name="bounded-pilot",
        ref=COMMIT,
        command=["python", "scripts/00_check_env.py"],
    )
    run_dir = tmp_path / "fake-remote/home/xiaoj/cement-channel-runs/policy-run-001"
    (run_dir / "large_features.npz").write_text("large", encoding="utf-8")

    waited = runner.wait("policy-run-001", poll_seconds=1)
    assert json.loads(waited.stdout)["status"] == "completed"

    reports_only = runner.fetch("policy-run-001", include_large_artifacts=False)
    assert reports_only.returncode == 0
    assert (tmp_path / "fetched/policy-run-001/manifest.json").exists()
    assert not (tmp_path / "fetched/policy-run-001/large_features.npz").exists()

    include_large = runner.fetch("policy-run-001", include_large_artifacts=True)
    assert include_large.returncode == 0
    assert (tmp_path / "fetched/policy-run-001/large_features.npz").exists()

    canceled = runner.cancel("policy-run-001")
    assert canceled.returncode == 0
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "canceled"
    assert (run_dir / "FAILED").exists()
