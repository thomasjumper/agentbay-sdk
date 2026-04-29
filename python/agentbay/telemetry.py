"""
AgentBay local SDK telemetry — opt-in error reporting + diagnostics.

Two surfaces:

    enable_error_reporting()         enable opt-in crash reports
    disable_error_reporting()        disable them
    get_telemetry_status()           current consent + anon id
    report_exception(...)            sanitize + send a single exception
    error_reporting_decorator(...)   wrap a method to auto-report

What's collected:
    - SDK name + version + OS + runtime version
    - exception class name (errorType)
    - exception message (truncated to 1KB)
    - stack trace with file paths replaced by basenames
    - context string (e.g. "memory.recall")
    - anonymous install id (joinable across events from the same install)

What's NOT collected:
    - User content (memory text, query strings)
    - Full file paths
    - API keys
    - Project IDs / user IDs

Backend: POST https://www.aiagentsbay.com/api/v1/telemetry/error
"""

from __future__ import annotations

import functools
import os
import platform
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

import requests

from .config import DEFAULT_BASE_URL, load_config, save_config

__all__ = [
    "enable_error_reporting",
    "disable_error_reporting",
    "get_telemetry_status",
    "report_exception",
    "error_reporting_decorator",
    "is_error_reporting_enabled",
    "get_or_create_anon_id",
]

# 8KB cap matches backend
_MAX_STACK_BYTES = 8192
_MAX_MESSAGE_BYTES = 1024
_TIMEOUT_SEC = 3.0
_SDK_NAME = "python"


def _sdk_version() -> str:
    try:
        from . import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def _runtime_string() -> str:
    return f"python-{platform.python_version()}"


def _os_string() -> str:
    return f"{platform.system()}-{platform.machine()}".lower()


def get_or_create_anon_id() -> str:
    """Return a stable anonymous id for this install. Created on first call."""
    config = load_config()
    telemetry = dict(config.get("telemetry") or {})
    anon = telemetry.get("anonId")
    if not anon:
        anon = uuid.uuid4().hex
        telemetry["anonId"] = anon
        config["telemetry"] = telemetry
        save_config(config)
    return anon


def _sanitize_stack(tb_text: str) -> str:
    """Replace absolute file paths with basenames + truncate to bound."""
    out_lines: list[str] = []
    for line in tb_text.splitlines():
        # Pattern: '  File "/abs/path/to/file.py", line N, in func'
        if 'File "' in line and '/' in line:
            try:
                pre, rest = line.split('File "', 1)
                path, post = rest.split('"', 1)
                line = f'{pre}File "{Path(path).name}"{post}'
            except Exception:
                pass
        out_lines.append(line)
    sanitized = "\n".join(out_lines)
    if len(sanitized.encode("utf-8")) > _MAX_STACK_BYTES:
        # truncate by chars (close enough for ASCII-mostly traces)
        sanitized = sanitized[: _MAX_STACK_BYTES // 2] + "\n... [truncated]"
    return sanitized


def is_error_reporting_enabled() -> bool:
    """True iff the user has run `agentbay telemetry enable` (or equivalent)."""
    if os.environ.get("AGENTBAY_QUIET"):
        return False
    if os.environ.get("AGENTBAY_TELEMETRY") == "0":
        return False
    config = load_config()
    telemetry = config.get("telemetry") or {}
    return bool(telemetry.get("errorConsent"))


def enable_error_reporting() -> None:
    """User-invoked: opt in to anonymous crash reporting."""
    config = load_config()
    telemetry = dict(config.get("telemetry") or {})
    telemetry["errorConsent"] = True
    telemetry.setdefault("anonId", uuid.uuid4().hex)
    config["telemetry"] = telemetry
    save_config(config)


def disable_error_reporting() -> None:
    """User-invoked: opt out of anonymous crash reporting."""
    config = load_config()
    telemetry = dict(config.get("telemetry") or {})
    telemetry["errorConsent"] = False
    config["telemetry"] = telemetry
    save_config(config)


def get_telemetry_status() -> dict[str, Any]:
    """Snapshot of current telemetry state for `agentbay telemetry status`."""
    config = load_config()
    telemetry = config.get("telemetry") or {}
    return {
        "installConsent": telemetry.get("installConsent"),
        "errorConsent": bool(telemetry.get("errorConsent")),
        "anonId": telemetry.get("anonId", "(not yet generated)"),
        "quietEnvVar": bool(os.environ.get("AGENTBAY_QUIET")),
        "telemetryDisabledEnvVar": os.environ.get("AGENTBAY_TELEMETRY") == "0",
    }


def report_exception(
    exc: BaseException,
    *,
    context: str = "",
    base_url: str = DEFAULT_BASE_URL,
    session: requests.Session | None = None,
) -> bool:
    """Send one exception to the telemetry endpoint. Returns True iff sent.

    Fail-soft: never raises. Does nothing if reporting is disabled.
    """
    if not is_error_reporting_enabled():
        return False

    try:
        anon_id = get_or_create_anon_id()
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        payload = {
            "anonId": anon_id,
            "sdk": _SDK_NAME,
            "sdkVersion": _sdk_version(),
            "os": _os_string(),
            "runtime": _runtime_string(),
            "errorType": type(exc).__name__,
            "errorMessage": str(exc)[:_MAX_MESSAGE_BYTES],
            "errorStack": _sanitize_stack(tb_text),
            "context": context[:128] if context else None,
        }
        http = session or requests.Session()
        http.post(
            f"{base_url.rstrip('/')}/api/v1/telemetry/error",
            json=payload,
            timeout=_TIMEOUT_SEC,
        )
        return True
    except Exception:
        # Never let telemetry errors propagate
        return False


F = TypeVar("F", bound=Callable[..., Any])


def error_reporting_decorator(context: str) -> Callable[[F], F]:
    """Decorator that auto-reports any exception raised by the wrapped fn.

    Usage:
        @error_reporting_decorator(context="memory.store")
        def store(self, ...):
            ...

    The decorator does NOT swallow the exception — it re-raises after
    sending the report.
    """
    def deco(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:
                report_exception(exc, context=context)
                raise
        return wrapper  # type: ignore[return-value]
    return deco
