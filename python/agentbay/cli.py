"""Command-line interface for AgentBay."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .auth import login_via_browser
from .client import __version__
from .config import DEFAULT_BASE_URL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentbay")
    subparsers = parser.add_subparsers(dest="command")

    login_parser = subparsers.add_parser("login", help="Connect the local SDK to your AgentBay cloud account.")
    login_parser.add_argument(
        "--base-url",
        default=os.environ.get("AGENTBAY_BASE_URL") or os.environ.get("AGENTBAY_URL") or DEFAULT_BASE_URL,
        help="Override the AgentBay app URL.",
    )

    # `agentbay diagnose [--output FILE] [--json]` — local-only paste-able bundle for bug reports.
    diag_parser = subparsers.add_parser(
        "diagnose",
        help="Generate a local diagnostic report you can paste into a GitHub issue (does not send anything).",
    )
    diag_parser.add_argument("-o", "--output", help="write report to a file instead of stdout")
    diag_parser.add_argument("--json", action="store_true", help="emit JSON instead of text")

    # `agentbay telemetry {enable,disable,status}` — opt-in error reporting controls.
    tele_parser = subparsers.add_parser(
        "telemetry",
        help="Manage opt-in anonymous error reporting (off by default).",
    )
    tele_sub = tele_parser.add_subparsers(dest="tele_command")
    tele_sub.add_parser("enable", help="Opt in to anonymous error reports.")
    tele_sub.add_parser("disable", help="Opt out of anonymous error reports.")
    tele_sub.add_parser("status", help="Show current telemetry consent state.")

    return parser


def _cmd_diagnose(args: argparse.Namespace) -> int:
    from .diagnose import build_diagnose_report, format_report_text

    report = build_diagnose_report()
    body = json.dumps(report, indent=2) if args.json else format_report_text(report)

    if args.output:
        from pathlib import Path
        Path(args.output).write_text(body)
        print(f"Diagnostic report written to {args.output}", file=sys.stderr)
    else:
        print(body)
    return 0


def _cmd_telemetry(args: argparse.Namespace) -> int:
    from .telemetry import (
        enable_error_reporting,
        disable_error_reporting,
        get_telemetry_status,
    )

    sub = args.tele_command
    if sub == "enable":
        enable_error_reporting()
        print("Anonymous error reporting: ENABLED.")
        print("  We will send: SDK version, OS, exception type/message,")
        print("  sanitized stack trace (file paths replaced with basenames).")
        print("  We will never send: memory content, project IDs, API keys.")
        print("  Run `agentbay telemetry disable` to opt out at any time.")
        return 0
    if sub == "disable":
        disable_error_reporting()
        print("Anonymous error reporting: DISABLED.")
        return 0
    if sub == "status":
        status = get_telemetry_status()
        print(json.dumps(status, indent=2))
        return 0
    print("Usage: agentbay telemetry {enable|disable|status}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "login":
        try:
            result = login_via_browser(base_url=args.base_url, version=__version__)
        except Exception as exc:
            print(f"agentbay login failed: {exc}", file=sys.stderr)
            return 1
        print(f"Saved API key to {result.config_path}")
        return 0

    if args.command == "diagnose":
        return _cmd_diagnose(args)

    if args.command == "telemetry":
        return _cmd_telemetry(args)

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
