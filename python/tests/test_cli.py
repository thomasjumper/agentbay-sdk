"""Tests for AgentBay CLI login and install telemetry."""

from __future__ import annotations

import json
import os
import platform
import stat
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from agentbay.auth import login_via_browser
from agentbay.config import get_config_file, maybe_prompt_for_install_ping


@pytest.fixture
def fake_agentbay_server():
    state: dict[str, Any] = {
        "session_token": "session-test-token",
        "api_key": "ab_live_test_cli_key",
        "telemetry_payloads": [],
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw or b"{}")

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path == "/api/v1/auth/agent-sessions":
                self._send_json(200, {"token": state["session_token"]})
                return

            if self.path == "/api/v1/telemetry/install":
                payload = self._read_json()
                state["telemetry_payloads"].append(payload)
                self._send_json(201, {"ok": True})
                return

            self._send_json(404, {"error": "not found"})

        def do_GET(self) -> None:
            self._send_json(404, {"error": "not found"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    state["base_url"] = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_login_via_browser_round_trips_key_to_local_config(tmp_path, monkeypatch, fake_agentbay_server):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AGENTBAY_QUIET", "1")

    def open_browser(url: str) -> bool:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        callback_url = (
            f"http://127.0.0.1:{params['port'][0]}/callback"
            f"?apiKey={fake_agentbay_server['api_key']}&state={params['state'][0]}"
        )
        with urllib.request.urlopen(callback_url, timeout=5) as response:
            assert response.status == 200
        return True

    result = login_via_browser(
        base_url=fake_agentbay_server["base_url"],
        version="1.5.0",
        open_browser=open_browser,
        timeout=5,
        prompt_for_telemetry=False,
    )

    config_path = Path(result.config_path)
    config = json.loads(config_path.read_text())

    assert config["apiKey"] == fake_agentbay_server["api_key"]
    assert config["baseUrl"] == fake_agentbay_server["base_url"]
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def test_install_ping_opt_in_posts_payload_and_persists_choice(tmp_path, monkeypatch, fake_agentbay_server):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AGENTBAY_QUIET", raising=False)
    monkeypatch.setattr(os, "isatty", lambda _fd: True)

    prompts: list[str] = []

    consent = maybe_prompt_for_install_ping(
        version="1.5.0",
        base_url=fake_agentbay_server["base_url"],
        input_fn=lambda prompt: prompts.append(prompt) or "y",
    )

    config = json.loads(get_config_file().read_text())
    payload = fake_agentbay_server["telemetry_payloads"][0]

    assert consent is True
    assert prompts == ["Send anonymous install ping? [y/N]"]
    assert config["telemetry"]["installConsent"] is True
    assert payload["version"] == "1.5.0"
    assert payload["os"] == platform.system()
    assert payload["anonId"] == config["telemetry"]["anonId"]


def test_install_ping_opt_out_persists_choice_without_posting(tmp_path, monkeypatch, fake_agentbay_server):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AGENTBAY_QUIET", raising=False)
    monkeypatch.setattr(os, "isatty", lambda _fd: True)

    consent = maybe_prompt_for_install_ping(
        version="1.5.0",
        base_url=fake_agentbay_server["base_url"],
        input_fn=lambda _prompt: "n",
    )

    config = json.loads(get_config_file().read_text())

    assert consent is False
    assert config["telemetry"]["installConsent"] is False
    assert fake_agentbay_server["telemetry_payloads"] == []
