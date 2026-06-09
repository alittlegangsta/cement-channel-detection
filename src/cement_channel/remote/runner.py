from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SSH_ALIAS = "cement-server"
DEFAULT_LOCAL_REPO_ROOT = Path("/home/xiaoj/cement-channel-detection")
DEFAULT_LOCAL_DATA_ROOT = Path("/home/xiaoj/cement-channel-data")
DEFAULT_REMOTE_REPO_ROOT = "/home/xiaoj/cement-channel-detection"
DEFAULT_REMOTE_DATA_ROOT = "/home/xiaoj/cement-channel-data"
DEFAULT_REMOTE_RUNS_ROOT = "/home/xiaoj/cement-channel-runs"
REMOTE_ENV_ROOT = "/home/xiaoj/conda-envs/cement_env_v3"
REMOTE_BIN = f"{REMOTE_ENV_ROOT}/bin"
REMOTE_PYTHON = f"{REMOTE_BIN}/python"
DEFAULT_REMOTE_PYTHON_ENV = REMOTE_ENV_ROOT
DEFAULT_FETCH_ROOT = Path("outputs/remote-runs")
DEFAULT_PILOT_MANIFEST = Path("experiments/manifests/stc_apes_pilot_dependencies.json")
DEFAULT_LOCAL_BUNDLE_ROOT = Path("/tmp/cement-remote-bundles")
REMOTE_BUNDLE_DIRNAME = "_bundles"
SAFE_FETCH_SUFFIXES = {".json", ".csv", ".md", ".png", ".log", ".txt", ".sh"}
LARGE_ARTIFACT_SUFFIXES = {
    ".h5",
    ".hdf5",
    ".joblib",
    ".mat",
    ".npy",
    ".npz",
    ".onnx",
    ".pt",
    ".pth",
    ".zarr",
}
SAFE_SYNC_SUFFIXES = {".json", ".yaml", ".yml", ".csv", ".md", ".txt", ".png"}
FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,140}$")
SUCCESS_STATUSES = {"succeeded", "completed"}
TERMINAL_STATUSES = SUCCESS_STATUSES | {"failed", "canceled"}


class RemoteRunnerError(RuntimeError):
    """Raised when a remote runner operation cannot be completed safely."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class RemoteConfig:
    ssh_alias: str = DEFAULT_SSH_ALIAS
    local_repo_root: Path = DEFAULT_LOCAL_REPO_ROOT
    local_data_root: Path = DEFAULT_LOCAL_DATA_ROOT
    remote_repo_root: str = DEFAULT_REMOTE_REPO_ROOT
    remote_data_root: str = DEFAULT_REMOTE_DATA_ROOT
    remote_runs_root: str = DEFAULT_REMOTE_RUNS_ROOT
    remote_env_root: str = REMOTE_ENV_ROOT
    local_fetch_root: Path = DEFAULT_FETCH_ROOT

    @property
    def remote_python_env(self) -> str:
        return self.remote_env_root

    @property
    def remote_bin(self) -> str:
        return f"{self.remote_env_root.rstrip('/')}/bin"

    @property
    def remote_python(self) -> str:
        return f"{self.remote_bin}/python"


class SshBackend:
    """Small SSH/rsync backend. Tests must use FakeSshBackend instead."""

    def __init__(self, ssh_alias: str) -> None:
        self.ssh_alias = ssh_alias

    def run_script(self, script: str, *, operation: str) -> CommandResult:
        completed = subprocess.run(
            ["ssh", self.ssh_alias, "bash", "-s"],
            input=script,
            capture_output=True,
            text=True,
            check=False,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    def rsync_upload(self, local_path: Path, remote_path: str, *, dry_run: bool) -> CommandResult:
        args = ["rsync", "-av"]
        if dry_run:
            args.append("--dry-run")
        args.extend([str(local_path), f"{self.ssh_alias}:{remote_path}"])
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    def rsync_fetch(
        self,
        remote_dir: str,
        local_dir: Path,
        *,
        include_large_artifacts: bool,
    ) -> CommandResult:
        local_dir.mkdir(parents=True, exist_ok=True)
        args = ["rsync", "-av"]
        if not include_large_artifacts:
            args.extend(
                [
                    "--include=*/",
                    "--include=*.json",
                    "--include=*.csv",
                    "--include=*.md",
                    "--include=*.png",
                    "--include=*.log",
                    "--include=*.txt",
                    "--include=*.sh",
                    "--include=DONE",
                    "--include=FAILED",
                    "--exclude=*",
                ]
            )
        args.extend([f"{self.ssh_alias}:{remote_dir.rstrip('/')}/", str(local_dir)])
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


class FakeSshBackend:
    """Fake SSH backend used by tests and local dry bootstrap checks."""

    def __init__(self, state_root: Path) -> None:
        self.state_root = state_root
        self.recorded_operations: list[str] = []

    def doctor(self, config: RemoteConfig) -> CommandResult:
        self._ensure_layout(config)
        self.recorded_operations.append("doctor")
        stdout = "\n".join(
            [
                "cement-remote doctor: fake backend",
                f"repo_root={self._map(config.remote_repo_root)}",
                f"data_root={self._map(config.remote_data_root)}",
                f"runs_root={self._map(config.remote_runs_root)}",
                "systemd_run_user=available",
                "tmux=available",
                "nohup=available",
            ]
        )
        return CommandResult(0, stdout + "\n", "")

    def init_layout(self, config: RemoteConfig) -> CommandResult:
        self._ensure_layout(config)
        self.recorded_operations.append("init-layout")
        return CommandResult(0, "cement-remote init-layout: fake layout ready\n", "")

    def sync_code(self, config: RemoteConfig, ref: str) -> CommandResult:
        self._ensure_layout(config)
        self.recorded_operations.append("sync-code")
        (self._map(config.remote_repo_root) / ".cement_remote_fake_head").write_text(
            ref + "\n",
            encoding="utf-8",
        )
        return CommandResult(0, f"checked_out_commit={ref}\n", "")

    def verify_env(self, config: RemoteConfig) -> CommandResult:
        self._ensure_layout(config)
        self.recorded_operations.append("verify-env")
        stdout = "\n".join(
            [
                "cement-remote verify-env: fake backend",
                "python=3.10.18",
                "scikit-learn=1.7.2",
                "check_env=passed",
                "pip_check=passed",
            ]
        )
        return CommandResult(0, stdout + "\n", "")

    def sync_data(self, manifest_path: Path, *, dry_run: bool) -> CommandResult:
        self.recorded_operations.append("sync-data-dry-run" if dry_run else "sync-data")
        manifest = _read_json(manifest_path)
        entries = list(_iter_sync_entries(manifest))
        if not entries:
            return CommandResult(
                0,
                "sync-data: no local-to-remote payloads; dependencies are remote read-only\n",
                "",
            )
        messages = []
        for entry in entries:
            local_path = Path(str(entry["local_path"]))
            remote_path = str(entry["remote_path"])
            _validate_sync_entry(local_path, remote_path)
            if not dry_run:
                mapped_remote = self._map(remote_path)
                mapped_remote.parent.mkdir(parents=True, exist_ok=True)
                if local_path.is_dir():
                    if mapped_remote.exists():
                        raise RemoteRunnerError(f"Fake remote destination exists: {mapped_remote}")
                    shutil.copytree(local_path, mapped_remote)
                else:
                    shutil.copy2(local_path, mapped_remote)
            messages.append(f"{local_path} -> {remote_path}")
        prefix = "sync-data dry-run" if dry_run else "sync-data"
        return CommandResult(0, prefix + ":\n" + "\n".join(messages) + "\n", "")

    def submit(
        self,
        config: RemoteConfig,
        *,
        run_id: str,
        name: str,
        ref: str,
        command: Sequence[str],
    ) -> CommandResult:
        self._ensure_layout(config)
        self.recorded_operations.append("submit")
        run_dir = self._map(f"{config.remote_runs_root}/{run_id}")
        run_dir.mkdir(parents=True, exist_ok=False)
        command_text = shlex.join(command)
        manifest = _build_run_manifest(
            config,
            run_id=run_id,
            name=name,
            ref=ref,
            command=command,
            scheduler="systemd-run",
        )
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(run_dir / "status.json", _status_payload("succeeded", run_id, 0))
        (run_dir / "stdout.log").write_text(
            f"fake backend did not execute remote command: {command_text}\n",
            encoding="utf-8",
        )
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
        (run_dir / "command.sh").write_text(
            _command_script(
                config,
                run_id,
                command,
                scheduler="systemd-run",
                git_commit=ref,
            ),
            encoding="utf-8",
        )
        (run_dir / "environment.txt").write_text(
            _fake_environment_text(config, run_id, ref, "systemd-run"),
            encoding="utf-8",
        )
        (run_dir / "git_commit.txt").write_text(ref + "\n", encoding="utf-8")
        _write_json(run_dir / "outputs.json", _collect_fake_outputs(run_dir))
        (run_dir / "DONE").write_text("", encoding="utf-8")
        stdout = f"submitted run_id={run_id} scheduler=systemd-run backend=fake\n"
        return CommandResult(0, stdout, "")

    def status(self, config: RemoteConfig, run_id: str) -> CommandResult:
        self.recorded_operations.append("status")
        status_path = self._map(f"{config.remote_runs_root}/{run_id}/status.json")
        return CommandResult(0, status_path.read_text(encoding="utf-8"), "")

    def logs(self, config: RemoteConfig, run_id: str, *, tail: int) -> CommandResult:
        self.recorded_operations.append("logs")
        run_dir = self._map(f"{config.remote_runs_root}/{run_id}")
        stdout_lines = _tail_lines(run_dir / "stdout.log", tail)
        stderr_lines = _tail_lines(run_dir / "stderr.log", tail)
        text = "== stdout.log ==\n" + stdout_lines + "\n== stderr.log ==\n" + stderr_lines
        return CommandResult(0, text, "")

    def manifest(self, config: RemoteConfig, run_id: str) -> CommandResult:
        self.recorded_operations.append("manifest")
        manifest_path = self._map(f"{config.remote_runs_root}/{run_id}/manifest.json")
        return CommandResult(0, manifest_path.read_text(encoding="utf-8"), "")

    def fetch(
        self,
        config: RemoteConfig,
        run_id: str,
        *,
        include_large_artifacts: bool,
    ) -> CommandResult:
        operation = "fetch-large" if include_large_artifacts else "fetch-reports"
        self.recorded_operations.append(operation)
        src = self._map(f"{config.remote_runs_root}/{run_id}")
        dst = config.local_fetch_root / run_id
        dst.mkdir(parents=True, exist_ok=True)
        copied = []
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            if not include_large_artifacts and not _is_safe_fetch_file(path):
                continue
            rel = path.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append(str(rel))
        return CommandResult(0, "fetched files:\n" + "\n".join(sorted(copied)) + "\n", "")

    def cancel(self, config: RemoteConfig, run_id: str) -> CommandResult:
        self.recorded_operations.append("cancel")
        run_dir = self._map(f"{config.remote_runs_root}/{run_id}")
        _write_json(run_dir / "status.json", _status_payload("canceled", run_id, 130))
        failed_path = run_dir / "FAILED"
        if not failed_path.exists():
            failed_path.write_text("", encoding="utf-8")
        return CommandResult(0, f"canceled run_id={run_id}\n", "")

    def list_runs(self, config: RemoteConfig) -> CommandResult:
        self.recorded_operations.append("list")
        runs_root = self._map(config.remote_runs_root)
        rows = []
        for manifest_path in sorted(runs_root.glob("*/manifest.json")):
            manifest = _read_json(manifest_path)
            run_id = str(manifest.get("run_id", manifest_path.parent.name))
            status_path = manifest_path.parent / "status.json"
            status = "unknown"
            if status_path.exists():
                status = str(_read_json(status_path).get("status", "unknown"))
            rows.append(f"{run_id}\t{status}\t{manifest.get('name', '')}")
        return CommandResult(0, "\n".join(rows) + ("\n" if rows else ""), "")

    def _ensure_layout(self, config: RemoteConfig) -> None:
        for remote_path in (
            config.remote_repo_root,
            config.remote_data_root,
            config.remote_runs_root,
            config.remote_python_env,
        ):
            self._map(remote_path).mkdir(parents=True, exist_ok=True)

    def _map(self, remote_path: str) -> Path:
        clean = remote_path.strip()
        if not clean.startswith("/"):
            raise RemoteRunnerError(f"Fake remote path must be absolute: {remote_path}")
        return self.state_root / clean.lstrip("/")


class RemoteRunner:
    def __init__(self, config: RemoteConfig, backend: SshBackend | FakeSshBackend) -> None:
        self.config = config
        self.backend = backend

    def doctor(self) -> CommandResult:
        if isinstance(self.backend, FakeSshBackend):
            return self.backend.doctor(self.config)
        return self.backend.run_script(_doctor_script(self.config), operation="doctor")

    def init_layout(self) -> CommandResult:
        if isinstance(self.backend, FakeSshBackend):
            return self.backend.init_layout(self.config)
        return self.backend.run_script(_init_layout_script(self.config), operation="init-layout")

    def sync_code(self, ref: str) -> CommandResult:
        _validate_commit(ref)
        if isinstance(self.backend, FakeSshBackend):
            return self.backend.sync_code(self.config, ref)
        _ensure_local_commit(self.config.local_repo_root, ref)
        local_bundle = _create_local_code_bundle(self.config.local_repo_root, ref)
        remote_bundle = (
            f"{self.config.remote_runs_root.rstrip('/')}/"
            f"{REMOTE_BUNDLE_DIRNAME}/{local_bundle.name}"
        )
        prepare = self.backend.run_script(
            _prepare_bundle_upload_script(self.config),
            operation="sync-code-prepare-bundle-upload",
        )
        if prepare.returncode != 0:
            return prepare
        upload = self.backend.rsync_upload(local_bundle, remote_bundle, dry_run=False)
        if upload.returncode != 0:
            return upload
        return self.backend.run_script(
            _sync_code_from_bundle_script(self.config, ref, remote_bundle),
            operation="sync-code",
        )

    def verify_env(self) -> CommandResult:
        if isinstance(self.backend, FakeSshBackend):
            return self.backend.verify_env(self.config)
        return self.backend.run_script(_verify_env_script(self.config), operation="verify-env")

    def sync_data(self, manifest_path: Path, *, dry_run: bool) -> CommandResult:
        manifest = _read_json(manifest_path)
        entries = list(_iter_sync_entries(manifest))
        if isinstance(self.backend, FakeSshBackend):
            return self.backend.sync_data(manifest_path, dry_run=dry_run)
        if not entries:
            return CommandResult(
                0,
                "sync-data: no local-to-remote payloads; dependencies are remote read-only\n",
                "",
            )

        stdout_parts = []
        stderr_parts = []
        returncode = 0
        for entry in entries:
            local_path = Path(str(entry["local_path"]))
            remote_path = str(entry["remote_path"])
            _validate_sync_entry(local_path, remote_path)
            result = self.backend.rsync_upload(local_path, remote_path, dry_run=dry_run)
            stdout_parts.append(result.stdout)
            stderr_parts.append(result.stderr)
            returncode = max(returncode, result.returncode)
        return CommandResult(returncode, "".join(stdout_parts), "".join(stderr_parts))

    def submit(self, *, name: str, ref: str, command: Sequence[str]) -> CommandResult:
        _validate_run_name(name)
        _validate_commit(ref)
        command_args = _strip_command_separator(command)
        _validate_remote_command(command_args)
        resolved_command = _resolve_remote_command(self.config, command_args)
        run_id = _make_run_id(name, ref)
        if isinstance(self.backend, FakeSshBackend):
            return self.backend.submit(
                self.config,
                run_id=run_id,
                name=name,
                ref=ref,
                command=resolved_command,
            )
        return self.backend.run_script(
            _submit_script(
                self.config,
                run_id=run_id,
                name=name,
                ref=ref,
                command=resolved_command,
            ),
            operation="submit",
        )

    def status(self, run_id: str) -> CommandResult:
        _validate_run_id(run_id)
        if isinstance(self.backend, FakeSshBackend):
            return self.backend.status(self.config, run_id)
        return self.backend.run_script(_status_script(self.config, run_id), operation="status")

    def logs(self, run_id: str, *, tail: int) -> CommandResult:
        _validate_run_id(run_id)
        if tail < 1:
            raise RemoteRunnerError("--tail must be >= 1")
        if isinstance(self.backend, FakeSshBackend):
            return self.backend.logs(self.config, run_id, tail=tail)
        return self.backend.run_script(_logs_script(self.config, run_id, tail), operation="logs")

    def wait(self, run_id: str, *, poll_seconds: int) -> CommandResult:
        _validate_run_id(run_id)
        if poll_seconds < 1:
            raise RemoteRunnerError("--poll-seconds must be >= 1")
        while True:
            result = self.status(run_id)
            if result.returncode != 0:
                return result
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                return result
            status = str(payload.get("status", "unknown"))
            if _is_terminal_status(status):
                return result
            time.sleep(poll_seconds)

    def manifest(self, run_id: str) -> CommandResult:
        _validate_run_id(run_id)
        if isinstance(self.backend, FakeSshBackend):
            return self.backend.manifest(self.config, run_id)
        return self.backend.run_script(_manifest_script(self.config, run_id), operation="manifest")

    def fetch(self, run_id: str, *, include_large_artifacts: bool) -> CommandResult:
        _validate_run_id(run_id)
        if isinstance(self.backend, FakeSshBackend):
            return self.backend.fetch(
                self.config,
                run_id,
                include_large_artifacts=include_large_artifacts,
            )
        remote_run_dir = f"{self.config.remote_runs_root.rstrip('/')}/{run_id}"
        return self.backend.rsync_fetch(
            remote_run_dir,
            self.config.local_fetch_root / run_id,
            include_large_artifacts=include_large_artifacts,
        )

    def cancel(self, run_id: str) -> CommandResult:
        _validate_run_id(run_id)
        if isinstance(self.backend, FakeSshBackend):
            return self.backend.cancel(self.config, run_id)
        return self.backend.run_script(_cancel_script(self.config, run_id), operation="cancel")

    def list_runs(self) -> CommandResult:
        if isinstance(self.backend, FakeSshBackend):
            return self.backend.list_runs(self.config)
        return self.backend.run_script(_list_script(self.config), operation="list")


def build_data_dependency_manifest(
    *,
    purpose: str,
    output_path: Path,
    config: RemoteConfig,
) -> dict[str, Any]:
    if purpose != "stc-apes-pilot":
        raise RemoteRunnerError(f"Unsupported data manifest purpose: {purpose}")

    manifest = {
        "manifest_version": "remote_data_dependency_v001",
        "purpose": "stc-apes-pilot",
        "stage": "MVP-4X-bounded-research",
        "research_only": True,
        "no_final_labels": True,
        "no_deep_learning": True,
        "no_full_well_stc": True,
        "no_full_well_apes": True,
        "created_by": "cement-remote build-data-manifest",
        "local_repo_root": str(config.local_repo_root),
        "local_data_root": str(config.local_data_root),
        "remote_repo_root": config.remote_repo_root,
        "remote_data_root": config.remote_data_root,
        "remote_runs_root": config.remote_runs_root,
        "remote_env_root": config.remote_env_root,
        "remote_bin": config.remote_bin,
        "remote_python": config.remote_python,
        "python_no_user_site": True,
        "remote_dependency_checks": [
            {
                "role": "cast_raw",
                "path": f"{config.remote_data_root}/raw/CAST.mat",
                "access": "read_only",
                "required": True,
            },
            {
                "role": "pose_raw",
                "path": f"{config.remote_data_root}/raw/D2_XSI_RelBearing_Inclination.mat",
                "access": "read_only",
                "required": True,
            },
            {
                "role": "xsi_receiver_raw_dir",
                "path": f"{config.remote_data_root}/raw/XSILMR",
                "access": "read_only",
                "required": True,
                "expected_files": 13,
                "file_pattern": "XSILMR*.mat",
            },
        ],
        "sync_entries": [],
        "pilot_bounds": {
            "max_wells": 1,
            "representative_interval_count_min": 80,
            "representative_interval_count_max": 160,
            "representative_interval_count_default": 120,
            "selected_interval_chunked_waveform_reads_only": True,
            "resource_micro_benchmark_required": True,
            "requires_explicit_limit_depth": True,
            "requires_dry_run_before_submit": True,
            "default_cuda_visible_devices": "1,2",
        },
        "allowed_outputs": {
            "default_fetch": ["json", "csv", "md", "png", "log", "txt"],
            "large_artifacts_stay_remote_by_default": ["npz", "joblib", "h5", "hdf5"],
        },
        "forbidden_operations": [
            "modify raw MAT",
            "delete remote data",
            "rsync --delete",
            "full-well STC",
            "full-well APES",
            "deep learning",
            "final labels",
            "production deployment",
        ],
        "operator_preconditions": [
            "remote raw MAT checksum validation completed before bootstrap",
            "remote Python environment passed scripts/00_check_env.py",
            "remote make test and make lint passed before bootstrap",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        runner = RemoteRunner(config, _backend_from_env(config))
        result = _dispatch(args, runner, config)
    except RemoteRunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    return result.returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cement-remote",
        description="Deterministic remote runner for cement-channel research jobs.",
    )
    parser.add_argument(
        "--ssh-alias",
        default=os.environ.get("CEMENT_REMOTE_SSH_ALIAS", DEFAULT_SSH_ALIAS),
    )
    parser.add_argument(
        "--local-repo-root",
        type=Path,
        default=Path(os.environ.get("CEMENT_REMOTE_LOCAL_REPO_ROOT", DEFAULT_LOCAL_REPO_ROOT)),
    )
    parser.add_argument(
        "--local-data-root",
        type=Path,
        default=Path(os.environ.get("CEMENT_REMOTE_LOCAL_DATA_ROOT", DEFAULT_LOCAL_DATA_ROOT)),
    )
    parser.add_argument(
        "--remote-repo-root",
        default=os.environ.get("CEMENT_REMOTE_REPO_ROOT", DEFAULT_REMOTE_REPO_ROOT),
    )
    parser.add_argument(
        "--remote-data-root",
        default=os.environ.get("CEMENT_REMOTE_DATA_ROOT", DEFAULT_REMOTE_DATA_ROOT),
    )
    parser.add_argument(
        "--remote-runs-root",
        default=os.environ.get("CEMENT_REMOTE_RUNS_ROOT", DEFAULT_REMOTE_RUNS_ROOT),
    )
    parser.add_argument(
        "--remote-python-env",
        default=os.environ.get("CEMENT_REMOTE_PYTHON_ENV", DEFAULT_REMOTE_PYTHON_ENV),
    )
    parser.add_argument(
        "--local-fetch-root",
        type=Path,
        default=Path(os.environ.get("CEMENT_REMOTE_FETCH_ROOT", DEFAULT_FETCH_ROOT)),
    )

    subparsers = parser.add_subparsers(dest="command_name", required=True)
    subparsers.add_parser("doctor")
    subparsers.add_parser("init-layout")

    sync_code = subparsers.add_parser("sync-code")
    sync_code.add_argument("--ref", required=True)

    build_manifest = subparsers.add_parser("build-data-manifest")
    build_manifest.add_argument("--purpose", choices=["stc-apes-pilot"], required=True)
    build_manifest.add_argument("--output", type=Path, default=DEFAULT_PILOT_MANIFEST)

    sync_data = subparsers.add_parser("sync-data")
    sync_data.add_argument("--manifest", type=Path, required=True)
    sync_data.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("verify-env")

    submit = subparsers.add_parser("submit")
    submit.add_argument("--name", required=True)
    submit.add_argument("--ref", required=True)
    submit.add_argument("remote_command", nargs=argparse.REMAINDER)

    status = subparsers.add_parser("status")
    status.add_argument("--run-id", required=True)

    logs = subparsers.add_parser("logs")
    logs.add_argument("--run-id", required=True)
    logs.add_argument("--tail", type=int, default=100)

    wait = subparsers.add_parser("wait")
    wait.add_argument("--run-id", required=True)
    wait.add_argument("--poll-seconds", type=int, default=30)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--run-id", required=True)

    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--run-id", required=True)
    fetch.add_argument("--reports-only", action="store_true")
    fetch.add_argument("--include-large-artifacts", action="store_true")

    cancel = subparsers.add_parser("cancel")
    cancel.add_argument("--run-id", required=True)

    subparsers.add_parser("list")
    return parser


def _config_from_args(args: argparse.Namespace) -> RemoteConfig:
    return RemoteConfig(
        ssh_alias=args.ssh_alias,
        local_repo_root=args.local_repo_root,
        local_data_root=args.local_data_root,
        remote_repo_root=args.remote_repo_root,
        remote_data_root=args.remote_data_root,
        remote_runs_root=args.remote_runs_root,
        remote_env_root=args.remote_python_env,
        local_fetch_root=args.local_fetch_root,
    )


def _backend_from_env(config: RemoteConfig) -> SshBackend | FakeSshBackend:
    backend_name = os.environ.get("CEMENT_REMOTE_BACKEND", "ssh")
    if backend_name == "fake":
        state_root = Path(os.environ.get("CEMENT_REMOTE_FAKE_ROOT", "/tmp/cement-remote-fake"))
        return FakeSshBackend(state_root)
    if backend_name != "ssh":
        raise RemoteRunnerError(f"Unsupported CEMENT_REMOTE_BACKEND: {backend_name}")
    return SshBackend(config.ssh_alias)


def _dispatch(
    args: argparse.Namespace,
    runner: RemoteRunner,
    config: RemoteConfig,
) -> CommandResult:
    command_name = args.command_name
    if command_name == "doctor":
        return runner.doctor()
    if command_name == "init-layout":
        return runner.init_layout()
    if command_name == "sync-code":
        return runner.sync_code(args.ref)
    if command_name == "build-data-manifest":
        manifest = build_data_dependency_manifest(
            purpose=args.purpose,
            output_path=args.output,
            config=config,
        )
        return CommandResult(
            0,
            f"wrote data dependency manifest: {args.output}\n"
            f"purpose={manifest['purpose']} sync_entries={len(manifest['sync_entries'])}\n",
            "",
        )
    if command_name == "sync-data":
        return runner.sync_data(args.manifest, dry_run=args.dry_run)
    if command_name == "verify-env":
        return runner.verify_env()
    if command_name == "submit":
        return runner.submit(name=args.name, ref=args.ref, command=args.remote_command)
    if command_name == "status":
        return runner.status(args.run_id)
    if command_name == "logs":
        return runner.logs(args.run_id, tail=args.tail)
    if command_name == "wait":
        return runner.wait(args.run_id, poll_seconds=args.poll_seconds)
    if command_name == "manifest":
        return runner.manifest(args.run_id)
    if command_name == "fetch":
        include_large = bool(args.include_large_artifacts)
        return runner.fetch(args.run_id, include_large_artifacts=include_large)
    if command_name == "cancel":
        return runner.cancel(args.run_id)
    if command_name == "list":
        return runner.list_runs()
    raise RemoteRunnerError(f"Unsupported command: {command_name}")


def _validate_commit(ref: str) -> None:
    if FULL_COMMIT_RE.fullmatch(ref) is None:
        raise RemoteRunnerError("--ref must be a full 40-character lowercase Git commit SHA")


def _validate_run_name(name: str) -> None:
    if RUN_NAME_RE.fullmatch(name) is None:
        raise RemoteRunnerError(
            "--name must start with an alphanumeric and contain only A-Z a-z 0-9 . _ -"
        )


def _validate_run_id(run_id: str) -> None:
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise RemoteRunnerError("--run-id contains unsafe characters")


def _strip_command_separator(command: Sequence[str]) -> list[str]:
    items = list(command)
    if items and items[0] == "--":
        items = items[1:]
    if not items:
        raise RemoteRunnerError("submit requires a command after --")
    return items


def _resolve_remote_command(config: RemoteConfig, command: Sequence[str]) -> list[str]:
    resolved = list(command)
    if resolved[0] in {"python", "python3"}:
        resolved[0] = config.remote_python
    return resolved


def _validate_remote_command(command: Sequence[str]) -> None:
    text = shlex.join(command)
    lowered = text.lower()
    banned_patterns = [
        (r"\bsshpass\b", "sshpass is forbidden"),
        (r"\bsudo\b", "sudo is forbidden"),
        (r"\brm\s+-[^\n]*r[^\n]*f", "rm -rf is forbidden"),
        (r"\brsync\b[^\n]*--delete\b", "rsync --delete is forbidden"),
        (r"\bgit\s+(push|merge|reset|rebase)\b", "push/merge/reset/rebase are forbidden"),
        (r"\btorchrun\b", "deep learning launchers are forbidden in this bootstrap"),
        (r"\bdeepspeed\b", "deep learning launchers are forbidden in this bootstrap"),
        (r"train_xsi_only|train_fusion|07_train_xsi_only", "deep learning is forbidden"),
        (r"final[_ -]?labels?", "final labels are forbidden"),
        (r"production[_ -]?deployment", "production deployment is forbidden"),
    ]
    for pattern, message in banned_patterns:
        if re.search(pattern, lowered):
            raise RemoteRunnerError(f"Unsafe submit command: {message}")

    mentions_stc_or_apes = re.search(r"(^|[^a-z])(stc|apes)([^a-z]|$)|stc_|apes_", lowered)
    bounded_markers = ("pilot", "bounded", "--limit-depth", "--limit_depth", "--dry-run")
    if mentions_stc_or_apes and not any(marker in lowered for marker in bounded_markers):
        raise RemoteRunnerError(
            "STC/APES submit commands must be explicitly bounded or pilot dry-runs"
        )


def _validate_sync_entry(local_path: Path, remote_path: str) -> None:
    if not local_path.exists():
        raise RemoteRunnerError(f"Local sync path does not exist: {local_path}")
    if not remote_path.startswith("/"):
        raise RemoteRunnerError(f"Remote sync path must be absolute: {remote_path}")
    suffix = local_path.suffix.lower()
    if suffix in LARGE_ARTIFACT_SUFFIXES or suffix not in SAFE_SYNC_SUFFIXES:
        raise RemoteRunnerError(f"Refusing to sync unsafe data file: {local_path}")
    if "--delete" in remote_path:
        raise RemoteRunnerError("Refusing unsafe remote sync path containing --delete")


def _iter_sync_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("sync_entries", [])
    if not isinstance(entries, list):
        raise RemoteRunnerError("manifest sync_entries must be a list")
    normalized = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise RemoteRunnerError("manifest sync_entries items must be mappings")
        if "local_path" not in entry or "remote_path" not in entry:
            raise RemoteRunnerError("sync_entries require local_path and remote_path")
        normalized.append(entry)
    return normalized


def _make_run_id(name: str, ref: str) -> str:
    override = os.environ.get("CEMENT_REMOTE_RUN_ID")
    if override:
        _validate_run_id(override)
        return override
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    return f"{timestamp}_{safe_name}_{ref[:12]}"


def _build_run_manifest(
    config: RemoteConfig,
    *,
    run_id: str,
    name: str,
    ref: str,
    command: Sequence[str],
    scheduler: str,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "manifest_version": "cement_remote_run_v001",
        "run_id": run_id,
        "name": name,
        "status": "submitted",
        "created_at": now,
        "ssh_alias": config.ssh_alias,
        "repo_root": config.remote_repo_root,
        "data_root": config.remote_data_root,
        "runs_root": config.remote_runs_root,
        "remote_env_root": config.remote_env_root,
        "remote_bin": config.remote_bin,
        "remote_python": config.remote_python,
        "python_no_user_site": True,
        "git_commit": ref,
        "command": list(command),
        "command_text": shlex.join(command),
        "scheduler": scheduler,
        "scheduler_preference": ["systemd-run", "tmux", "nohup"],
        "safety": {
            "no_secrets": True,
            "no_sudo": True,
            "no_sshpass": True,
            "no_rsync_delete": True,
            "raw_mat_read_only": True,
            "reports_only_fetch_by_default": True,
        },
    }


def _status_payload(status: str, run_id: str, exit_code: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if exit_code is not None:
        payload["exit_code"] = exit_code
    return payload


def _collect_fake_outputs(run_dir: Path) -> dict[str, Any]:
    report_files = []
    large_artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(run_dir))
        if _is_safe_fetch_file(path):
            report_files.append(rel)
        elif path.suffix.lower() in LARGE_ARTIFACT_SUFFIXES:
            large_artifacts.append(rel)
    return {
        "reports_only_files": report_files,
        "large_artifacts": large_artifacts,
    }


def _fake_environment_text(
    config: RemoteConfig,
    run_id: str,
    ref: str,
    scheduler: str,
) -> str:
    path_value = f"{config.remote_bin}:$PATH"
    return (
        "\n".join(
            [
                f"PATH={path_value}",
                f"command -v python: {config.remote_python}",
                "python --version: Python 3.10.18",
                f"command -v pip: {config.remote_bin}/pip",
                f"python -m pip --version: pip from {config.remote_env_root}",
                f"repo root: {config.remote_repo_root}",
                f"data root: {config.remote_data_root}",
                f"run id: {run_id}",
                f"scheduler: {scheduler}",
                f"git commit: {ref}",
                "PYTHONNOUSERSITE=1",
            ]
        )
        + "\n"
    )


def _is_safe_fetch_file(path: Path) -> bool:
    return path.name in {"DONE", "FAILED"} or path.suffix.lower() in SAFE_FETCH_SUFFIXES


def _tail_lines(path: Path, tail: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-tail:])


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RemoteRunnerError(f"JSON file must contain an object: {path}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_local_git(repo_root: Path, args: Sequence[str]) -> CommandResult:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _ensure_local_commit(repo_root: Path, ref: str) -> None:
    result = _run_local_git(
        repo_root,
        ["cat-file", "-e", f"{ref}^{{commit}}"],
    )
    if result.returncode != 0:
        raise RemoteRunnerError(
            f"Local repo does not contain commit {ref}: {result.stderr.strip()}"
        )


def _create_local_code_bundle(repo_root: Path, ref: str) -> Path:
    bundle_root = Path(os.environ.get("CEMENT_REMOTE_LOCAL_BUNDLE_ROOT", DEFAULT_LOCAL_BUNDLE_ROOT))
    bundle_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_path = bundle_root / f"cement-code-{ref}-{timestamp}.bundle"
    bundle_refs = _bundle_refs_for_commit(repo_root, ref)
    result = _run_local_git(
        repo_root,
        ["bundle", "create", str(bundle_path), *bundle_refs],
    )
    if result.returncode != 0:
        raise RemoteRunnerError(f"Unable to create local Git bundle: {result.stderr.strip()}")
    if not bundle_path.exists():
        raise RemoteRunnerError(f"Git bundle was not created: {bundle_path}")
    return bundle_path


def _bundle_refs_for_commit(repo_root: Path, ref: str) -> list[str]:
    current = _run_local_git(
        repo_root,
        ["symbolic-ref", "--quiet", "--short", "HEAD"],
    )
    branch_refs = _local_branches_containing_commit(repo_root, ref)
    current_ref = f"refs/heads/{current.stdout.strip()}" if current.returncode == 0 else ""
    if current_ref and current_ref in branch_refs:
        return [current_ref]
    if branch_refs:
        return [branch_refs[0]]
    raise RemoteRunnerError(
        f"Commit {ref} is not reachable from a local branch; create a branch before sync-code."
    )


def _local_branches_containing_commit(repo_root: Path, ref: str) -> list[str]:
    result = _run_local_git(
        repo_root,
        ["for-each-ref", "--contains", ref, "--format=%(refname)", "refs/heads"],
    )
    if result.returncode != 0:
        raise RemoteRunnerError(
            f"Unable to find local branches containing {ref}: {result.stderr.strip()}"
        )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def _is_terminal_status(status: str) -> bool:
    return status in TERMINAL_STATUSES


def _doctor_script(config: RemoteConfig) -> str:
    return f"""\
set -euo pipefail
echo "cement-remote doctor"
test -d {_q(config.remote_repo_root)}
test -d {_q(config.remote_data_root)}
command -v git >/dev/null
if command -v systemd-run >/dev/null 2>&1; then
  echo "systemd_run_user=available"
else
  echo "systemd_run_user=missing"
fi
if command -v tmux >/dev/null 2>&1; then
  echo "tmux=available"
else
  echo "tmux=missing"
fi
command -v nohup >/dev/null
test -x {_q(config.remote_python)}
{_q(config.remote_python)} --version
"""


def _init_layout_script(config: RemoteConfig) -> str:
    return f"""\
set -euo pipefail
mkdir -p {_q(config.remote_runs_root)}
mkdir -p {_q(config.remote_data_root)}/{{manifests,reports,logs,tmp}}
test -d {_q(config.remote_repo_root)}
test -d {_q(config.remote_data_root)}/raw
echo "initialized remote layout at {config.remote_runs_root}"
"""


def _prepare_bundle_upload_script(config: RemoteConfig) -> str:
    return f"""\
set -euo pipefail
mkdir -p {_q(config.remote_runs_root.rstrip("/") + "/" + REMOTE_BUNDLE_DIRNAME)}
cd {_q(config.remote_repo_root)}
test "$(git rev-parse --is-inside-work-tree)" = "true"
echo "bundle_upload_dir={config.remote_runs_root.rstrip("/")}/{REMOTE_BUNDLE_DIRNAME}"
"""


def _sync_code_from_bundle_script(config: RemoteConfig, ref: str, remote_bundle: str) -> str:
    return f"""\
set -euo pipefail
cd {_q(config.remote_repo_root)}
test "$(git rev-parse --is-inside-work-tree)" = "true"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: remote repo has tracked local changes; refusing safe checkout" >&2
  exit 2
fi
test -f {_q(remote_bundle)}
git fetch {_q(remote_bundle)} '+refs/heads/*:refs/remotes/cement-bundle/*'
git cat-file -e {_q(ref)}^{{commit}}
git -c advice.detachedHead=false checkout --detach {_q(ref)}
actual="$(git rev-parse HEAD)"
test "$actual" = {_q(ref)}
echo "checked_out_commit=$actual"
"""


def _verify_env_script(config: RemoteConfig) -> str:
    return f"""\
set -euo pipefail
cd {_q(config.remote_repo_root)}
test -x {_q(config.remote_python)}
{_q(config.remote_python)} scripts/00_check_env.py
{_q(config.remote_python)} -m pip check
{_q(config.remote_python)} -c 'import sklearn; print("scikit-learn", sklearn.__version__)'
"""


def _submit_script(
    config: RemoteConfig,
    *,
    run_id: str,
    name: str,
    ref: str,
    command: Sequence[str],
) -> str:
    manifest = json.dumps(
        _build_run_manifest(
            config,
            run_id=run_id,
            name=name,
            ref=ref,
            command=command,
            scheduler="__CEMENT_REMOTE_SCHEDULER__",
        ),
        indent=2,
        sort_keys=True,
    )
    command_script = _command_script(config, run_id, command)
    return f"""\
set -euo pipefail
RUN_ID={_q(run_id)}
RUN_DIR={_q(config.remote_runs_root.rstrip("/") + "/" + run_id)}
REPO={_q(config.remote_repo_root)}
REF={_q(ref)}
PYTHON={_q(config.remote_python)}
REMOTE_ENV_ROOT={_q(config.remote_env_root)}
REMOTE_BIN={_q(config.remote_bin)}
cd "$REPO"
git cat-file -e "$REF^{{commit}}"
current="$(git rev-parse HEAD)"
if [ "$current" != "$REF" ]; then
  git checkout --detach "$REF"
fi
mkdir -p "$RUN_DIR"
{_remote_choose_scheduler_function()}
scheduler="$(choose_scheduler)"
cat > "$RUN_DIR/manifest.json" <<'CEMENT_REMOTE_JSON'
{manifest}
CEMENT_REMOTE_JSON
sed -i "s/__CEMENT_REMOTE_SCHEDULER__/$scheduler/g" "$RUN_DIR/manifest.json"
cat > "$RUN_DIR/status.json" <<CEMENT_REMOTE_STATUS
{{"run_id":"$RUN_ID","status":"submitted","updated_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}}
CEMENT_REMOTE_STATUS
cat > "$RUN_DIR/command.sh" <<'CEMENT_REMOTE_COMMAND'
{command_script}
CEMENT_REMOTE_COMMAND
sed -i "s#__CEMENT_REMOTE_SCHEDULER__#$scheduler#g" "$RUN_DIR/command.sh"
sed -i "s#__CEMENT_REMOTE_GIT_COMMIT__#$REF#g" "$RUN_DIR/command.sh"
chmod 700 "$RUN_DIR/command.sh"
touch "$RUN_DIR/stdout.log" "$RUN_DIR/stderr.log"
printf "%s\\n" "$REF" > "$RUN_DIR/git_commit.txt"
{{
  hostname || true
  uname -a || true
  echo "remote_env_root=$REMOTE_ENV_ROOT"
  echo "remote_python=$PYTHON"
  "$PYTHON" --version || true
  nvidia-smi || true
}} > "$RUN_DIR/environment.txt" 2>&1
cat > "$RUN_DIR/outputs.json" <<'CEMENT_REMOTE_OUTPUTS'
{{"reports_only_files":[],"large_artifacts":[]}}
CEMENT_REMOTE_OUTPUTS
TMUX_COMMAND="env PATH=\\"$REMOTE_BIN:${{PATH:-}}\\""
TMUX_COMMAND="$TMUX_COMMAND PYTHONNOUSERSITE=1"
TMUX_COMMAND="$TMUX_COMMAND CEMENT_REMOTE_RUN_ID=\\"$RUN_ID\\""
TMUX_COMMAND="$TMUX_COMMAND CEMENT_REMOTE_SCHEDULER=\\"$scheduler\\""
TMUX_COMMAND="$TMUX_COMMAND CEMENT_REMOTE_GIT_COMMIT=\\"$REF\\""
TMUX_COMMAND="$TMUX_COMMAND \\"$RUN_DIR/command.sh\\""
case "$scheduler" in
  systemd-run)
    systemd-run --user --unit "cement-$RUN_ID" --collect \\
      --setenv=PATH="$REMOTE_BIN:${{PATH:-}}" \\
      --setenv=PYTHONNOUSERSITE=1 \\
      --setenv=CEMENT_REMOTE_RUN_ID="$RUN_ID" \\
      --setenv=CEMENT_REMOTE_SCHEDULER="$scheduler" \\
      --setenv=CEMENT_REMOTE_GIT_COMMIT="$REF" \\
      "$RUN_DIR/command.sh"
    ;;
  tmux)
    tmux new-session -d -s "cement_$RUN_ID" "$TMUX_COMMAND"
    ;;
  nohup)
    env PATH="$REMOTE_BIN:${{PATH:-}}" \\
      PYTHONNOUSERSITE=1 \\
      CEMENT_REMOTE_RUN_ID="$RUN_ID" \\
      CEMENT_REMOTE_SCHEDULER="$scheduler" \\
      CEMENT_REMOTE_GIT_COMMIT="$REF" \\
      nohup "$RUN_DIR/command.sh" >/dev/null 2>&1 &
    echo "$!" > "$RUN_DIR/pid.txt"
    ;;
esac
echo "submitted run_id=$RUN_ID scheduler=$scheduler"
"""


def _command_script(
    config: RemoteConfig,
    run_id: str,
    command: Sequence[str],
    *,
    scheduler: str = "__CEMENT_REMOTE_SCHEDULER__",
    git_commit: str = "__CEMENT_REMOTE_GIT_COMMIT__",
) -> str:
    run_dir = f"{config.remote_runs_root.rstrip('/')}/{run_id}"
    command_array = _bash_array_literal("REMOTE_COMMAND", command)
    return f"""#!/bin/bash
set -uo pipefail
RUN_ID={shlex.quote(run_id)}
RUN_DIR={shlex.quote(run_dir)}
REPO={shlex.quote(config.remote_repo_root)}
REMOTE_ENV_ROOT={shlex.quote(config.remote_env_root)}
REMOTE_BIN={shlex.quote(config.remote_bin)}
REMOTE_PYTHON={shlex.quote(config.remote_python)}
export CEMENT_REMOTE_RUN_ID="$RUN_ID"
export CEMENT_REMOTE_RUN_DIR="$RUN_DIR"
export CEMENT_CHANNEL_DATA_ROOT={shlex.quote(config.remote_data_root)}
export CEMENT_REMOTE_SCHEDULER={shlex.quote(scheduler)}
export CEMENT_REMOTE_GIT_COMMIT={shlex.quote(git_commit)}
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-1,2}}"
export PATH="$REMOTE_BIN:${{PATH:-}}"
export PYTHONNOUSERSITE=1
{command_array}
{{
  echo "PATH=$PATH"
  printf "command -v python: "
  command -v python || true
  printf "python --version: "
  python --version || true
  printf "command -v pip: "
  command -v pip || true
  printf "python -m pip --version: "
  python -m pip --version || true
  echo "repo root: $REPO"
  echo "data root: $CEMENT_CHANNEL_DATA_ROOT"
  echo "run id: $RUN_ID"
  echo "scheduler: $CEMENT_REMOTE_SCHEDULER"
  echo "git commit: $CEMENT_REMOTE_GIT_COMMIT"
}} > "$RUN_DIR/environment.txt" 2>&1
if [ ! -x "$REMOTE_PYTHON" ]; then
  echo "ERROR: remote Python is not executable: $REMOTE_PYTHON" >> "$RUN_DIR/stderr.log"
  touch "$RUN_DIR/FAILED"
cat > "$RUN_DIR/status.json" <<CEMENT_REMOTE_STATUS
{{
  "run_id":"$RUN_ID",
  "status":"failed",
  "exit_code":127,
  "updated_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}}
CEMENT_REMOTE_STATUS
  exit 127
fi
if ! cd "$REPO"; then
  echo "ERROR: repo root is not accessible: $REPO" >> "$RUN_DIR/stderr.log"
  touch "$RUN_DIR/FAILED"
cat > "$RUN_DIR/status.json" <<CEMENT_REMOTE_STATUS
{{
  "run_id":"$RUN_ID",
  "status":"failed",
  "exit_code":2,
  "updated_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}}
CEMENT_REMOTE_STATUS
  exit 2
fi
cat > "$RUN_DIR/status.json" <<CEMENT_REMOTE_STATUS
{{"run_id":"$RUN_ID","status":"running","updated_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}}
CEMENT_REMOTE_STATUS
"${{REMOTE_COMMAND[@]}}" >> "$RUN_DIR/stdout.log" 2>> "$RUN_DIR/stderr.log"
exit_code=$?
if [ "$exit_code" -eq 0 ]; then
  touch "$RUN_DIR/DONE"
  final_status=succeeded
else
  touch "$RUN_DIR/FAILED"
  final_status=failed
fi
find "$RUN_DIR" -maxdepth 3 -type f | sort > "$RUN_DIR/output_files.txt" 2>/dev/null || true
cat > "$RUN_DIR/outputs.json" <<CEMENT_REMOTE_OUTPUTS
{{"output_files_txt":"$RUN_DIR/output_files.txt"}}
CEMENT_REMOTE_OUTPUTS
cat > "$RUN_DIR/status.json" <<CEMENT_REMOTE_STATUS
{{
  "run_id":"$RUN_ID",
  "status":"$final_status",
  "exit_code":$exit_code,
  "updated_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}}
CEMENT_REMOTE_STATUS
exit "$exit_code"
"""


def _remote_choose_scheduler_function() -> str:
    return """\
choose_scheduler() {
  if command -v systemd-run >/dev/null 2>&1 && systemd-run --user --version >/dev/null 2>&1; then
    echo systemd-run
    return 0
  fi
  if command -v tmux >/dev/null 2>&1; then
    echo tmux
    return 0
  fi
  echo nohup
}
"""


def _bash_array_literal(name: str, values: Sequence[str]) -> str:
    lines = [f"{name}=("]
    lines.extend(f"  {shlex.quote(value)}" for value in values)
    lines.append(")")
    return "\n".join(lines)


def _status_script(config: RemoteConfig, run_id: str) -> str:
    run_dir = f"{config.remote_runs_root.rstrip('/')}/{run_id}"
    return f"""\
set -euo pipefail
cat {_q(run_dir + "/status.json")}
"""


def _logs_script(config: RemoteConfig, run_id: str, tail: int) -> str:
    run_dir = f"{config.remote_runs_root.rstrip('/')}/{run_id}"
    return f"""\
set -euo pipefail
echo "== stdout.log =="
tail -n {tail} {_q(run_dir + "/stdout.log")} || true
echo "== stderr.log =="
tail -n {tail} {_q(run_dir + "/stderr.log")} || true
"""


def _manifest_script(config: RemoteConfig, run_id: str) -> str:
    run_dir = f"{config.remote_runs_root.rstrip('/')}/{run_id}"
    return f"""\
set -euo pipefail
cat {_q(run_dir + "/manifest.json")}
"""


def _cancel_script(config: RemoteConfig, run_id: str) -> str:
    run_dir = f"{config.remote_runs_root.rstrip('/')}/{run_id}"
    return f"""\
set -euo pipefail
RUN_ID={_q(run_id)}
RUN_DIR={_q(run_dir)}
if systemctl --user --quiet is-active "cement-$RUN_ID.service"; then
  systemctl --user stop "cement-$RUN_ID.service" || true
fi
if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "cement_$RUN_ID" 2>/dev/null || true
fi
if [ -f "$RUN_DIR/pid.txt" ]; then
  kill "$(cat "$RUN_DIR/pid.txt")" 2>/dev/null || true
fi
touch "$RUN_DIR/FAILED"
cat > "$RUN_DIR/status.json" <<CEMENT_REMOTE_STATUS
{{
  "run_id":"$RUN_ID",
  "status":"canceled",
  "exit_code":130,
  "updated_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}}
CEMENT_REMOTE_STATUS
echo "canceled run_id=$RUN_ID"
"""


def _list_script(config: RemoteConfig) -> str:
    return f"""\
set -euo pipefail
find {_q(config.remote_runs_root)} \\
  -mindepth 2 -maxdepth 2 -name manifest.json -print \\
  | sort \\
  | while read -r manifest; do
  run_dir="$(dirname "$manifest")"
  run_id="$(basename "$run_dir")"
  status=unknown
  if [ -f "$run_dir/status.json" ]; then
    status="$(sed -n 's/.*"status"[ ]*:[ ]*"\\([^"]*\\)".*/\\1/p' \\
      "$run_dir/status.json" | head -n 1)"
  fi
  printf "%s\\t%s\\n" "$run_id" "${{status:-unknown}}"
done
"""


def _q(value: str) -> str:
    return shlex.quote(value)


if __name__ == "__main__":
    raise SystemExit(main())
