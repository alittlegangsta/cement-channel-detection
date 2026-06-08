# cement-remote-research

Use this skill when running or preparing remote research jobs for the
`cement-channel-detection` repository.

## Scope

This skill covers deterministic remote job orchestration through:

```bash
cement-remote
python scripts/cement_remote.py
```

Default endpoints:

```text
local repo: /home/xiaoj/cement-channel-detection
local data: /home/xiaoj/cement-channel-data
remote SSH alias: cement-server
remote repo root: /home/xiaoj/cement-channel-detection
remote data root: /home/xiaoj/cement-channel-data
remote runs root: /home/xiaoj/cement-channel-runs
remote Python env: /home/xiaoj/conda-envs/cement_env_v3
```

## Required Order

1. Run `cement-remote doctor`.
2. Run `cement-remote init-layout` if the runs root or report/log directories are missing.
3. Build the bounded pilot dependency manifest:

   ```bash
   cement-remote build-data-manifest --purpose stc-apes-pilot
   ```

4. Run `cement-remote sync-data --manifest <manifest> --dry-run`.
5. Only after review, run `cement-remote sync-data --manifest <manifest>` if it has safe
   local-to-remote payloads.
6. Run `cement-remote sync-code --ref <40-char-commit>` after the commit is available to the
   remote repository.
7. Run `cement-remote verify-env`.
8. Submit only bounded or dry-run research commands:

   ```bash
   cement-remote submit --name <name> --ref <40-char-commit> -- <command>
   ```

9. Monitor with `status`, `logs`, `wait`, `manifest`, and `list`.
10. Fetch with `fetch --reports-only` by default.

## Scheduler Policy

Remote submissions must use this scheduler priority:

```text
systemd-run --user -> tmux -> nohup
```

Do not choose `tmux` while `systemd-run --user` is available. Do not choose `nohup` while either
`systemd-run --user` or `tmux` is available.

## Safety Boundaries

Never request, print, or save passwords, tokens, SSH private keys, or `.env` values.

Forbidden commands and operations:

```text
sshpass
sudo
rm -rf
rsync --delete
git push
git merge
git reset
git rebase
modify raw MAT
delete remote data
full-well STC
full-well APES
deep learning
final labels
production deployment
```

Use full 40-character Git commit SHAs for `sync-code --ref` and `submit --ref`; never use a branch
name, tag, short SHA, or `HEAD` for remote execution.

## Fetch Policy

Default fetch mode is reports-only. It may pull:

```text
JSON
CSV
MD
PNG
LOG
TXT
command.sh
DONE
FAILED
```

Large artifacts such as NPZ, joblib, HDF5, STC/APES maps, and model weights stay on the server unless
the user explicitly requests `fetch --include-large-artifacts`.

## Testing Policy

Automated tests must set:

```bash
CEMENT_REMOTE_BACKEND=fake
```

Tests must not connect to `cement-server`, must not run real `ssh`, and must not run real `rsync`.

## Bootstrap Boundary

During REMOTE-RUNNER-BOOTSTRAP, stop after creating the CLI, skill, manifests, tests, docs, and
commit. Do not submit expensive remote jobs.
