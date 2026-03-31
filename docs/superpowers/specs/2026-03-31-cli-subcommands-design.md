# CLI Subcommands Design

**Date:** 2026-03-31
**Status:** Approved

## Overview

Migrate service lifecycle management from `scripts/service.sh` into native Python Click subcommands, so users can manage the server via `mlx-speech-server <subcommand>` instead of the shell script. `service.sh` is retained as a backup but no longer the primary interface.

**Installation flow:** Users install the CLI via `pipx install .`, then run `mlx-speech-server install` which creates a dedicated service venv and registers a launchd agent.

## Architecture

### Files Changed

| File | Change |
| :--- | :--- |
| `cli.py` | New — Click group entry point, all subcommand definitions |
| `app/service.py` | New — service management logic (venv, launchd, process control) |
| `pyproject.toml` | Modify — entry point `main:cli` → `cli:cli` |
| `main.py` | Unchanged — server process entry point for launchd |
| `scripts/service.sh` | Unchanged — retained as backup |

### Runtime Paths

```
User interface:
  pipx install .  →  mlx-speech-server <subcommand>  →  cli.py  →  app/service.py

launchd service process:
  launchctl  →  python main.py  →  app/server.py  (unchanged)
```

`cli.py` handles Click definitions and terminal output only — no platform logic.
`app/service.py` owns all macOS-specific operations (launchctl, plist, venv, subprocess).

### Project Root Detection

`app/service.py` derives the project root via `Path(__file__).parent.parent`, equivalent to `service.sh`'s `$(dirname "$0")/..`. No user-supplied path required.

### Paths

| Resource | Path |
| :--- | :--- |
| Service venv | `~/.local/venvs/mlx-speech-server/` |
| Logs | `~/.local/logs/mlx-speech-server/` |
| launchd plist | `~/Library/LaunchAgents/com.local.mlx-speech-server.plist` |
| Service label | `com.local.mlx-speech-server` |

## Subcommands

| Subcommand | Behavior |
| :--- | :--- |
| `install` | Create venv at `~/.local/venvs/mlx-speech-server/` → `pip install -e <project_root>` → write launchd plist with `.env` vars injected → print summary. Does **not** start the service. |
| `uninstall` | `launchctl bootout` → delete plist. Venv is kept. |
| `upgrade` | `git pull origin main` → if new commits: `pip install -e <project_root>` → ask "Restart now? [y/N]" → restart if yes. If already up to date: skip install. |
| `start` | If plist missing: auto-run `install` first. Then `launchctl bootstrap` + `launchctl kickstart`. |
| `stop` | `launchctl bootout`. |
| `restart` | `stop` → brief wait → `start`. |
| `status` | Show plist path, venv, PID, health check (`GET /health`), queue stats (`GET /v1/queue/stats`). Matches `service.sh status` output style. |
| `logs` | Print last 30 lines of stdout + stderr logs. Print `tail -f` hint at end. |

### `.env` Support

`install` reads `<project_root>/.env` and injects `WHISPER_*` variables into the plist `EnvironmentVariables` dict, identical to `service.sh` behavior.

### `upgrade` Detail

```
git pull origin main
  ├─ "Already up to date." → print "Already up to date, nothing to do."
  ├─ new commits           → pip install -e <project_root>
  │                           → "Restart now? [y/N]"
  │                               ├─ y → restart
  │                               └─ N → print "Run: mlx-speech-server restart"
  └─ git error             → print error, exit 1 (no install)
```

## Error Handling

| Scenario | Behavior |
| :--- | :--- |
| Non-macOS platform | `app/service.py` checks `sys.platform` at module level; raises friendly error before any launchctl call |
| `stop`/`restart`/`status`/`logs`/`uninstall` called before `install` | Print `Not installed. Run: mlx-speech-server install`, exit code 1 |
| launchctl subprocess failure | Capture non-zero return code, print raw error output, do not swallow |
| `git pull` failure (network, conflict) | Print error, exit 1, do not proceed to pip install |
| `install` when venv already exists | Skip venv creation, re-run `pip install` (idempotent) |
| Interactive prompt in non-TTY environment | Default to N, do not block |

## Testing

### `tests/test_service.py` — unit tests
- Mock `subprocess.run` to verify launchctl, pip, and git call arguments
- Mock `Path.exists` to exercise installed/not-installed branches
- Test `.env` parsing and plist env var injection
- Test `upgrade` branching (up to date vs. new commits vs. git failure)

### `tests/test_cli.py` — CLI integration tests
- Use Click's `CliRunner` to invoke each subcommand
- Assert output text and exit codes
- All `app/service.py` functions mocked — CLI layer tested independently of platform logic

### Out of scope
- End-to-end tests involving real launchctl calls (not suitable for CI)
- Existing tests in `tests/api/` and `tests/test_config.py` are unaffected
