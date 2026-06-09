from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from cement_channel.remote.runner import (
    REMOTE_BIN,
    REMOTE_ENV_ROOT,
    REMOTE_PYTHON,
    CommandResult,
    FakeSshBackend,
    RemoteConfig,
    RemoteRunner,
    RemoteRunnerError,
    _prepare_bundle_upload_script,
    _submit_script,
    _sync_code_from_bundle_script,
    build_data_dependency_manifest,
)

COMMIT = "0123456789abcdef0123456789abcdef01234567"


class RecordingSshBackend:
    def __init__(self) -> None:
        self.scripts: list[tuple[str, str]] = []
        self.uploads: list[tuple[Path, str, bool]] = []

    def run_script(self, script: str, *, operation: str) -> CommandResult:
        self.scripts.append((operation, script))
        if operation == "sync-code":
            return CommandResult(0, "checked_out_commit=recorded\n", "")
        return CommandResult(0, "ok\n", "")

    def rsync_upload(self, local_path: Path, remote_path: str, *, dry_run: bool) -> CommandResult:
        self.uploads.append((local_path, remote_path, dry_run))
        return CommandResult(0, "uploaded\n", "")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("test repo\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(
        repo,
        "-c",
        "user.name=Cement Remote Tests",
        "-c",
        "user.email=cement-remote-tests@example.invalid",
        "commit",
        "-m",
        "initial",
    )
    return repo, _git(repo, "rev-parse", "HEAD")


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
    assert manifest["remote_env_root"] == REMOTE_ENV_ROOT
    assert manifest["remote_bin"] == REMOTE_BIN
    assert manifest["remote_python"] == REMOTE_PYTHON
    assert manifest["python_no_user_site"] is True


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
    command_sh = (run_dir / "command.sh").read_text(encoding="utf-8")
    assert status["status"] == "succeeded"
    assert manifest["scheduler"] == "systemd-run"
    assert manifest["git_commit"] == COMMIT
    assert manifest["remote_env_root"] == REMOTE_ENV_ROOT
    assert manifest["remote_python"] == REMOTE_PYTHON
    assert manifest["python_no_user_site"] is True
    assert "export CEMENT_REMOTE_SCHEDULER=systemd-run" in command_sh
    assert f"export CEMENT_REMOTE_GIT_COMMIT={COMMIT}" in command_sh


@pytest.mark.parametrize("python_executable", ["python", "python3"])
def test_submit_rewrites_python_argv0_only(
    python_executable: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = f"python-rewrite-{python_executable}"
    monkeypatch.setenv("CEMENT_REMOTE_RUN_ID", run_id)
    runner = RemoteRunner(RemoteConfig(), FakeSshBackend(tmp_path / "fake-remote"))

    runner.submit(
        name="bounded-pilot",
        ref=COMMIT,
        command=[python_executable, "script.py", "--note", "literal python arg"],
    )

    run_dir = tmp_path / f"fake-remote/home/xiaoj/cement-channel-runs/{run_id}"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    command_sh = (run_dir / "command.sh").read_text(encoding="utf-8")
    assert manifest["command"][0] == REMOTE_PYTHON
    assert manifest["command"][1:] == ["script.py", "--note", "literal python arg"]
    assert f"  {REMOTE_PYTHON}" in command_sh
    assert "literal python arg" in command_sh
    assert f"{python_executable} script.py" not in command_sh
    assert "/usr/bin/python" not in command_sh
    assert "/usr/bin/python" not in json.dumps(manifest)


def test_submit_does_not_replace_python_in_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CEMENT_REMOTE_RUN_ID", "python-argument-safe")
    runner = RemoteRunner(RemoteConfig(), FakeSshBackend(tmp_path / "fake-remote"))

    runner.submit(
        name="bounded-pilot",
        ref=COMMIT,
        command=["make", "PYTHON=python", "note=python"],
    )

    run_dir = tmp_path / "fake-remote/home/xiaoj/cement-channel-runs/python-argument-safe"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"] == ["make", "PYTHON=python", "note=python"]


def test_sync_code_uploads_local_bundle_and_uses_safe_remote_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, ref = _make_git_repo(tmp_path)
    monkeypatch.setenv("CEMENT_REMOTE_LOCAL_BUNDLE_ROOT", str(tmp_path / "bundles"))
    backend = RecordingSshBackend()
    runner = RemoteRunner(RemoteConfig(local_repo_root=repo), backend)  # type: ignore[arg-type]

    result = runner.sync_code(ref)

    assert result.returncode == 0
    assert len(backend.uploads) == 1
    local_bundle, remote_bundle, dry_run = backend.uploads[0]
    assert local_bundle.exists()
    assert local_bundle.suffix == ".bundle"
    assert remote_bundle.startswith("/home/xiaoj/cement-channel-runs/_bundles/")
    assert dry_run is False
    operations = [operation for operation, _script in backend.scripts]
    assert operations == ["sync-code-prepare-bundle-upload", "sync-code"]
    checkout_script = backend.scripts[-1][1]
    assert "git diff --quiet" in checkout_script
    assert "git fetch" in checkout_script
    assert "refs/remotes/cement-bundle" in checkout_script
    assert "checkout --detach" in checkout_script
    assert ref in checkout_script
    assert "git reset" not in checkout_script


def test_sync_code_scripts_prepare_bundle_dir_and_fetch_bundle() -> None:
    config = RemoteConfig()
    prepare_script = _prepare_bundle_upload_script(config)
    checkout_script = _sync_code_from_bundle_script(
        config,
        COMMIT,
        "/home/xiaoj/cement-channel-runs/_bundles/cement-code.bundle",
    )

    assert "mkdir -p /home/xiaoj/cement-channel-runs/_bundles" in prepare_script
    assert "git fetch" in checkout_script
    assert "git cat-file -e" in checkout_script
    assert "checkout --detach" in checkout_script
    assert "remote repo has tracked local changes" in checkout_script
    assert "git reset" not in checkout_script


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
                            "/home/xiaoj/cement-channel-runs/_manifests/pilot_manifest.json"
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
    copied = tmp_path / "fake-remote/home/xiaoj/cement-channel-runs/_manifests/pilot_manifest.json"
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
    assert json.loads(waited.stdout)["status"] == "succeeded"

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


def test_wait_accepts_historical_completed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CEMENT_REMOTE_RUN_ID", "completed-history-run")
    runner = RemoteRunner(RemoteConfig(), FakeSshBackend(tmp_path / "fake-remote"))
    runner.submit(
        name="bounded-pilot",
        ref=COMMIT,
        command=["python", "scripts/00_check_env.py"],
    )
    status_path = (
        tmp_path / "fake-remote/home/xiaoj/cement-channel-runs/completed-history-run/status.json"
    )
    status_path.write_text(
        json.dumps({"run_id": "completed-history-run", "status": "completed"}) + "\n",
        encoding="utf-8",
    )

    waited = runner.wait("completed-history-run", poll_seconds=1)

    assert json.loads(waited.stdout)["status"] == "completed"


def test_command_script_records_remote_environment_fields() -> None:
    script = _submit_script(
        RemoteConfig(),
        run_id="env-run-001",
        name="bounded-pilot",
        ref=COMMIT,
        command=[REMOTE_PYTHON, "scripts/00_check_env.py"],
    )

    assert f"REMOTE_ENV_ROOT={REMOTE_ENV_ROOT}" in script
    assert f"REMOTE_BIN={REMOTE_BIN}" in script
    assert f"REMOTE_PYTHON={REMOTE_PYTHON}" in script
    assert 'export PATH="$REMOTE_BIN:${PATH:-}"' in script
    assert "export PYTHONNOUSERSITE=1" in script
    assert "command -v python" in script
    assert "python --version" in script
    assert "command -v pip" in script
    assert "python -m pip --version" in script
    assert "repo root: $REPO" in script
    assert "data root: $CEMENT_CHANNEL_DATA_ROOT" in script
    assert "run id: $RUN_ID" in script
    assert "scheduler: $CEMENT_REMOTE_SCHEDULER" in script
    assert "git commit: $CEMENT_REMOTE_GIT_COMMIT" in script


def test_scheduler_launches_inject_remote_environment() -> None:
    script = _submit_script(
        RemoteConfig(),
        run_id="scheduler-run-001",
        name="bounded-pilot",
        ref=COMMIT,
        command=[REMOTE_PYTHON, "scripts/00_check_env.py"],
    )

    assert "systemd-run --user" in script
    assert '--setenv=PATH="$REMOTE_BIN:${PATH:-}"' in script
    assert "--setenv=PYTHONNOUSERSITE=1" in script
    assert '--setenv=CEMENT_REMOTE_SCHEDULER="$scheduler"' in script
    assert "tmux new-session" in script
    assert 'TMUX_COMMAND="env PATH=\\"$REMOTE_BIN:${PATH:-}\\""' in script
    assert 'TMUX_COMMAND="$TMUX_COMMAND PYTHONNOUSERSITE=1"' in script
    assert 'CEMENT_REMOTE_RUN_ID=\\"$RUN_ID\\"' in script
    assert 'nohup "$RUN_DIR/command.sh"' in script
    assert 'env PATH="$REMOTE_BIN:${PATH:-}" \\' in script


def test_command_script_uses_bash_array_to_prevent_shell_injection() -> None:
    script = _submit_script(
        RemoteConfig(),
        run_id="injection-run-001",
        name="bounded-pilot",
        ref=COMMIT,
        command=[REMOTE_PYTHON, "script.py", "value; touch /tmp/pwn", "$(whoami)"],
    )

    assert "${REMOTE_COMMAND[@]}" in script
    assert "  'value; touch /tmp/pwn'" in script
    assert "  '$(whoami)'" in script
    assert "value; touch /tmp/pwn >>" not in script
    assert "$(whoami) >>" not in script
