"""Browser-based AgentBay login flow for the SDK and CLI."""

from __future__ import annotations

import secrets
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlencode, urlparse

import requests

from .config import DEFAULT_BASE_URL, maybe_prompt_for_install_ping, save_api_key


@dataclass
class LoginResult:
    api_key: str
    config_path: str
    browser_url: str


class LoopbackLoginError(RuntimeError):
    """Raised when the browser login flow cannot complete."""


class _LoopbackLoginServer:
    def __init__(self, base_url: str, state: str, session: requests.Session):
        self.base_url = base_url.rstrip("/")
        self.state = state
        self.session = session
        self._event = threading.Event()
        self._error: str | None = None
        self._api_key: str | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> int:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args) -> None:
                return

            def _send_html(self, status: int, title: str, body: str) -> None:
                payload = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    f"<title>{title}</title></head><body>"
                    f"<h1>{title}</h1><p>{body}</p></body></html>"
                ).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path not in {"/", "/callback"}:
                    self._send_html(404, "AgentBay login failed", "Unknown callback path.")
                    return

                params = parse_qs(parsed.query)
                if params.get("state", [None])[0] != parent.state:
                    parent._error = "The login callback state did not match."
                    parent._event.set()
                    self._send_html(400, "AgentBay login failed", parent._error)
                    return

                api_key = params.get("apiKey", [None])[0] or params.get("api_key", [None])[0]
                if api_key:
                    parent._api_key = api_key
                else:
                    token = params.get("token", [None])[0]
                    if not token:
                        parent._error = "Missing API key callback."
                        parent._event.set()
                        self._send_html(400, "AgentBay login failed", parent._error)
                        return

                    try:
                        parent._api_key = parent.claim_api_key(token)
                    except Exception as exc:  # pragma: no cover - defensive
                        parent._error = str(exc)
                        parent._event.set()
                        self._send_html(502, "AgentBay login failed", parent._error)
                        return

                parent._event.set()
                self._send_html(
                    200,
                    "AgentBay connected",
                    "Your API key reached the CLI. You can close this tab and return to your terminal.",
                )

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return int(self._server.server_address[1])

    def claim_api_key(self, token: str) -> str:
        response = self.session.get(
            f"{self.base_url}/api/v1/auth/agent-sessions/{token}",
            timeout=10,
        )
        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise LoopbackLoginError("AgentBay returned invalid JSON while claiming the API key.") from exc

        api_key = data.get("apiKey")
        if response.status_code != 200 or data.get("status") != "completed" or not isinstance(api_key, str):
            message = data.get("error") or data.get("message") or "AgentBay did not return an API key."
            raise LoopbackLoginError(message)
        return api_key

    def wait_for_api_key(self, timeout: float) -> str:
        try:
            if not self._event.wait(timeout):
                raise LoopbackLoginError("Timed out waiting for the AgentBay browser login to finish.")
            if self._error:
                raise LoopbackLoginError(self._error)
            if not self._api_key:
                raise LoopbackLoginError("AgentBay login finished without returning an API key.")
            return self._api_key
        finally:
            self.close()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None


def create_agent_session(base_url: str, *, session: requests.Session | None = None) -> str:
    http = session or requests.Session()
    response = http.post(f"{base_url.rstrip('/')}/api/v1/auth/agent-sessions", timeout=10)
    try:
        data = response.json()
    except ValueError as exc:  # pragma: no cover - defensive
        raise LoopbackLoginError("AgentBay returned invalid JSON while starting the login flow.") from exc

    token = data.get("token")
    if response.status_code != 200 or not isinstance(token, str):
        message = data.get("error") or "AgentBay could not start the login flow."
        raise LoopbackLoginError(message)
    return token


def login_via_browser(
    *,
    base_url: str = DEFAULT_BASE_URL,
    version: str,
    open_browser: Callable[[str], bool] = webbrowser.open,
    print_fn: Callable[[str], None] = print,
    session: requests.Session | None = None,
    timeout: float = 300,
    prompt_for_telemetry: bool = True,
) -> LoginResult:
    http = session or requests.Session()

    if prompt_for_telemetry:
        maybe_prompt_for_install_ping(version=version, base_url=base_url, session=http)

    token = create_agent_session(base_url, session=http)
    state = secrets.token_urlsafe(18)
    callback_server = _LoopbackLoginServer(base_url, state, http)
    port = callback_server.start()

    browser_url = f"{base_url.rstrip('/')}/cli-auth?{urlencode({'token': token, 'port': port, 'state': state})}"

    opened = False
    try:
        opened = bool(open_browser(browser_url))
    except Exception:
        opened = False

    if not opened:
        print_fn(f"Open this URL in your browser:\n{browser_url}")

    api_key = callback_server.wait_for_api_key(timeout)
    config_path = save_api_key(api_key, base_url=base_url)

    return LoginResult(
        api_key=api_key,
        config_path=str(config_path),
        browser_url=browser_url,
    )
