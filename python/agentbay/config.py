"""AgentBay local config helpers."""

from __future__ import annotations

import json
import os
import platform
import uuid
from pathlib import Path
from typing import Any, Callable

import requests

DEFAULT_BASE_URL = "https://www.aiagentsbay.com"
LOCAL_UPGRADE_PROMPT = "Running locally. Run `agentbay login` to sync to cloud and unlock Teams (free, no card)."
INSTALL_PING_PROMPT = "Send anonymous install ping? [y/N]"


def get_config_dir() -> Path:
    return Path.home() / ".agentbay"


def get_config_file() -> Path:
    return get_config_dir() / "config.json"


def load_config() -> dict[str, Any]:
    config_file = get_config_file()
    if not config_file.exists():
        return {}
    try:
        data = json.loads(config_file.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_config(config: dict[str, Any]) -> Path:
    config_dir = get_config_dir()
    config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(config_dir, 0o700)
    except OSError:
        pass

    config_file = get_config_file()
    temp_file = config_file.with_suffix(".tmp")
    temp_file.write_text(json.dumps(config, indent=2))
    try:
        os.chmod(temp_file, 0o600)
    except OSError:
        pass
    temp_file.replace(config_file)
    try:
        os.chmod(config_file, 0o600)
    except OSError:
        pass
    return config_file


def load_saved_key() -> str | None:
    config = load_config()
    value = config.get("apiKey") or config.get("api_key")
    if not isinstance(value, str):
        return None
    return value.strip() or None


def save_api_key(api_key: str, base_url: str = DEFAULT_BASE_URL) -> Path:
    config = load_config()
    config["apiKey"] = api_key
    config["api_key"] = api_key
    config["baseUrl"] = base_url.rstrip("/")
    return save_config(config)


def maybe_prompt_for_install_ping(
    version: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    input_fn: Callable[[str], str] = input,
    session: requests.Session | None = None,
) -> bool | None:
    """Ask for install telemetry consent once and persist the choice."""
    if os.environ.get("AGENTBAY_QUIET"):
        return None

    try:
        if not os.isatty(0):
            return None
    except OSError:
        return None

    config = load_config()
    telemetry = dict(config.get("telemetry") or {})
    if "installConsent" in telemetry:
        return bool(telemetry["installConsent"])

    try:
        answer = input_fn(INSTALL_PING_PROMPT).strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""

    consent = answer in {"y", "yes"}
    telemetry["installConsent"] = consent
    telemetry["anonId"] = telemetry.get("anonId") or uuid.uuid4().hex
    config["telemetry"] = telemetry
    save_config(config)

    if consent:
        http = session or requests.Session()
        try:
            http.post(
                f"{base_url.rstrip('/')}/api/v1/telemetry/install",
                json={
                    "version": version,
                    "os": platform.system(),
                    "anonId": telemetry["anonId"],
                },
                timeout=5,
            )
        except requests.RequestException:
            pass

    return consent
