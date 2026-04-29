"""Command-line interface for AgentBay."""

from __future__ import annotations

import argparse
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
    return parser


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

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
