"""CLI entry point for mlx-speech-server service management."""
import subprocess
import sys

import click

from app import service


@click.group()
def cli() -> None:
    """mlx-speech-server: OpenAI-compatible Whisper API on Apple Silicon."""


@cli.command()
def install() -> None:
    """Create service venv, install deps, register launchd agent."""
    try:
        service.install()
    except Exception as e:
        click.echo(f"Install failed: {e}")
        sys.exit(1)
    click.echo("Installation complete.")
    click.echo(f"  Project:  {service.PROJECT_ROOT}")
    click.echo(f"  Venv:     {service.VENV_DIR}")
    click.echo(f"  Logs:     {service.LOG_DIR}")
    click.echo(f"  Plist:    {service.PLIST_PATH}")
    click.echo("")
    click.echo("Run: mlx-speech-server start")


@cli.command()
def uninstall() -> None:
    """Stop and remove launchd agent. Venv is kept."""
    try:
        service.uninstall()
    except (RuntimeError, subprocess.CalledProcessError) as e:
        click.echo(str(e))
        sys.exit(1)
    click.echo("Service uninstalled.")
    click.echo(f"  Venv kept at: {service.VENV_DIR}")
    click.echo(f"  To remove it: rm -rf {service.VENV_DIR}")


@cli.command()
def upgrade() -> None:
    """Pull latest code and reinstall if updated."""
    try:
        result = service.upgrade()
    except subprocess.CalledProcessError as e:
        click.echo(f"Upgrade failed: {e}")
        sys.exit(1)
    if result["status"] == "up_to_date":
        click.echo("Already up to date, nothing to do.")
        return
    click.echo("Dependencies updated.")
    if click.confirm("Restart now?", default=False):
        try:
            service.restart()
            click.echo("Service restarted.")
        except Exception as e:
            click.echo(f"Restart failed: {e}")
            sys.exit(1)
    else:
        click.echo("Run: mlx-speech-server restart")


@cli.command()
def start() -> None:
    """Start service (auto-installs if needed)."""
    try:
        service.start()
    except Exception as e:
        click.echo(f"Start failed: {e}")
        sys.exit(1)
    click.echo("Service started.")
    _print_status()


@cli.command()
def stop() -> None:
    """Stop the running service."""
    try:
        service.stop()
    except (RuntimeError, subprocess.CalledProcessError) as e:
        click.echo(str(e))
        sys.exit(1)
    click.echo("Service stopped.")


@cli.command()
def restart() -> None:
    """Restart the service."""
    try:
        service.restart()
    except RuntimeError as e:
        click.echo(str(e))
        sys.exit(1)
    click.echo("Service restarted.")
    _print_status()


@cli.command()
def status() -> None:
    """Show process status and health."""
    _print_status()


@cli.command()
def logs() -> None:
    """Show recent stdout and stderr logs."""
    stdout, stderr = service.get_logs()
    click.echo("─── stdout (last 30 lines) ─────────────")
    click.echo(stdout)
    click.echo("")
    click.echo("─── stderr (last 30 lines) ─────────────")
    click.echo(stderr)
    click.echo("")
    click.echo(f"Live tail: tail -f {service.STDOUT_LOG} {service.STDERR_LOG}")


def _print_status() -> None:
    s = service.get_status()
    click.echo("════════════════════════════════════════")
    click.echo("  mlx-speech-server — Status")
    click.echo("════════════════════════════════════════")
    if not s["installed"]:
        click.echo("  ⚪ Not installed")
        click.echo("════════════════════════════════════════")
        return
    click.echo(f"  Plist:    {s['plist_path']}")
    click.echo(f"  Venv:     {s['venv_dir']}")
    click.echo(f"  Logs:     {s['log_dir']}")
    if not s.get("loaded"):
        click.echo("")
        click.echo("  ⚪ Installed but not loaded")
        click.echo("════════════════════════════════════════")
        return
    click.echo("")
    if s["running"]:
        click.echo(f"  Process:  🟢 Running (PID {s['pid']})")
        click.echo(f"  Port:     {s['port']}")
        if s["health"]:
            click.echo(f"  Health:   🟢 OK — {s['health']}")
        else:
            click.echo("  Health:   🟡 Not responding yet (model may still be loading)")
        if s["queue"]:
            click.echo(f"  Queue:    {s['queue']}")
        click.echo("")
        click.echo("  API Docs:")
        click.echo(f"    Swagger UI → http://localhost:{s['port']}/docs")
        click.echo(f"    ReDoc      → http://localhost:{s['port']}/redoc")
    else:
        click.echo("  Process:  🔴 Not running")
    click.echo("════════════════════════════════════════")
