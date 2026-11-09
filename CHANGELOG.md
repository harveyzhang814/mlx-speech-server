## v0.3.0 — 2026-11-09

### Features
- Add explicit model preparation without an audio upload, with nonblocking prepare requests and pollable download, loading, ready, and failure states
- Add a versioned preparation capabilities endpoint identifying the configured model and server instance
- Add `mlx ensure-running --json` to install or start a missing service without restarting an existing healthy process
- Add first-install model and port options, machine-readable installation results, and registration of an already provisioned runtime
- Accept the exact configured model path as an alias in audio transcription requests

### Bug Fixes
- Share repeated preparation requests and reuse the same evaluated model weights for subsequent transcription
- Keep native loading and inference serialized after request timeout or cancellation; coordinate preparation, idle unload, and shutdown
- Report download activity without counting existing resume offsets or overlapping retries, and preserve model readiness on queue admission errors
- Preserve structured service failure diagnostics with sensitive details redacted, and probe the configured host correctly

### Chores / Other
- Document the preparation protocol, service readiness contract, and isolated real-model verification procedure
- Add real-model E2E coverage for preparation, transcription reuse, and native loading beyond the request deadline
- Clear existing lint errors required by the release checks

---

## v0.2.2 — 2026-07-08

### Features
- Add `mlx --version` flag to show installed package version

---

## v0.2.1 — 2026-07-08

### Features
- Add `--port` and `--json` flags to `mlx status`

---

## v0.2.0 — 2026-07-08

### Features
- Add idle model unload after 30min inactivity to free unified memory

### Bug Fixes
- Fix launchd service reliability: silence spurious bootstrap/bootout errors, ensure clean state on start
- Fix plist entry point name (`mlx-speech-server-run` → `mlx-run`)
- Fix `mlx status` showing wrong port (8000 → 47300 default)
- Close watcher race window and skip redundant unload in handler

### Chores / Other
- Rename CLI entry point to `mlx`, change default port to 47300
- Add git workflow hooks (conventional commits, branch protection)
- Extract `DEFAULT_PORT` constant to eliminate duplicate magic numbers
- Add unit tests and e2e test for idle unload
