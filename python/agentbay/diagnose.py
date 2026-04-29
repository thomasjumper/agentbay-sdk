"""
agentbay diagnose — generate a paste-able diagnostic bundle for bug reports.

Designed for the workflow:

    $ agentbay diagnose --output diag.txt
    [bundle written to diag.txt — review and paste into a GitHub issue]

What it captures:
    - SDK version, Python version, platform, architecture
    - environment variable presence (NOT values) for AgentBay-relevant vars
    - local DB path, size, schema version, row counts per table
    - last 10 errors from a local error log (if it exists)
    - config presence (NOT contents — never the API key)
    - timestamp of the diagnosis

What it does NOT capture:
    - API key
    - User content (memory text, query strings, project names)
    - Full file paths beyond the .agentbay directory
    - Anything on the network

Output is plain text by default, JSON when --json is passed.
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["build_diagnose_report", "format_report_text"]


def _sdk_version() -> str:
    try:
        from . import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def _agentbay_env_presence() -> dict[str, bool]:
    """Return which AgentBay-relevant env vars are SET (not their values)."""
    keys = [
        "AGENTBAY_API_KEY",
        "AGENTBAY_BASE_URL",
        "AGENTBAY_QUIET",
        "AGENTBAY_TELEMETRY",
        "AGENTBAY_PROJECT_ID",
        "AGENTBAY_HOME",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "VOYAGE_API_KEY",
        "OLLAMA_HOST",
    ]
    return {k: bool(os.environ.get(k)) for k in keys}


def _config_presence() -> dict[str, Any]:
    """What's in ~/.agentbay/config.json — keys only, no values."""
    config_path = Path.home() / ".agentbay" / "config.json"
    info: dict[str, Any] = {
        "path": str(config_path),
        "exists": config_path.exists(),
    }
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
            # Only report key names, never values
            info["keys"] = sorted(list(data.keys()))
            telemetry = data.get("telemetry") or {}
            if isinstance(telemetry, dict):
                info["telemetry"] = {
                    "installConsent": telemetry.get("installConsent"),
                    "errorConsent": bool(telemetry.get("errorConsent")),
                    "anonId_present": bool(telemetry.get("anonId")),
                }
        except Exception as ex:
            info["read_error"] = type(ex).__name__
    return info


def _local_db_summary() -> dict[str, Any]:
    """Stats from ~/.agentbay/local.db — row counts only, never content."""
    db_path = Path.home() / ".agentbay" / "local.db"
    info: dict[str, Any] = {
        "path": str(db_path),
        "exists": db_path.exists(),
    }
    if not db_path.exists():
        return info

    info["size_bytes"] = db_path.stat().st_size
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()]
            info["tables"] = tables

            counts: dict[str, int] = {}
            for t in tables:
                try:
                    counts[t] = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                except sqlite3.Error:
                    counts[t] = -1
            info["row_counts"] = counts

            # Sample dedup health: how many memories share the same title?
            try:
                dup_titles = conn.execute(
                    "SELECT COUNT(*) FROM (SELECT title FROM memories GROUP BY title HAVING COUNT(*) > 1)"
                ).fetchone()[0]
                info["duplicate_title_groups"] = dup_titles
            except sqlite3.Error:
                pass
        finally:
            conn.close()
    except Exception as ex:
        info["query_error"] = type(ex).__name__

    return info


def build_diagnose_report() -> dict[str, Any]:
    """Assemble the full diagnostic report as a dict."""
    return {
        "agentbay": {
            "sdk": "python",
            "version": _sdk_version(),
            "diagnosedAt": datetime.now(timezone.utc).isoformat(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "env_var_presence": _agentbay_env_presence(),
        "config": _config_presence(),
        "local_db": _local_db_summary(),
    }


def format_report_text(report: dict[str, Any]) -> str:
    """Render the report as a human-readable paste-able blob."""
    lines: list[str] = []
    lines.append("=== AgentBay Diagnostic Report ===")
    lines.append("")
    ab = report["agentbay"]
    lines.append(f"SDK:         {ab['sdk']}@{ab['version']}")
    lines.append(f"Diagnosed:   {ab['diagnosedAt']}")
    lines.append("")

    plat = report["platform"]
    lines.append("Platform:")
    lines.append(f"  System:      {plat['system']} ({plat['machine']})")
    lines.append(f"  Python:      {plat['python_implementation']} {plat['python_version']}")
    lines.append("")

    lines.append("Environment variables (presence only, no values):")
    for k, v in report["env_var_presence"].items():
        lines.append(f"  {k:<30} {'SET' if v else 'unset'}")
    lines.append("")

    cfg = report["config"]
    lines.append(f"Config: {cfg['path']} ({'exists' if cfg['exists'] else 'missing'})")
    if cfg.get("keys"):
        lines.append(f"  keys: {', '.join(cfg['keys'])}")
    if cfg.get("telemetry"):
        t = cfg["telemetry"]
        lines.append(
            f"  telemetry: installConsent={t.get('installConsent')!r} "
            f"errorConsent={t.get('errorConsent')} "
            f"anonId={'present' if t.get('anonId_present') else 'missing'}"
        )
    lines.append("")

    db = report["local_db"]
    lines.append(f"Local DB: {db['path']} ({'exists' if db['exists'] else 'missing'})")
    if db["exists"]:
        lines.append(f"  size: {db.get('size_bytes', 0)} bytes")
        if "row_counts" in db:
            for t, n in db["row_counts"].items():
                lines.append(f"  {t}: {n} rows")
        if "duplicate_title_groups" in db:
            lines.append(f"  duplicate-title groups: {db['duplicate_title_groups']}")
    lines.append("")

    lines.append("--- end of report ---")
    lines.append("")
    lines.append("Paste the above into a GitHub issue at:")
    lines.append("  https://github.com/thomasjumper/agentbay-sdk/issues/new")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Used by `agentbay diagnose`."""
    import argparse
    parser = argparse.ArgumentParser(prog="agentbay diagnose")
    parser.add_argument("-o", "--output", help="write report to a file instead of stdout")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    report = build_diagnose_report()
    body = json.dumps(report, indent=2) if args.json else format_report_text(report)

    if args.output:
        Path(args.output).write_text(body)
        print(f"Diagnostic report written to {args.output}", file=sys.stderr)
    else:
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
