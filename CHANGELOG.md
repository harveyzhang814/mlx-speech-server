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
