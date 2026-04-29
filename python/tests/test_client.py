"""Tests for AgentBay client — constructor, error handling, provider detection."""

import os
import pytest
from unittest.mock import patch, MagicMock
from agentbay.client import (
    AgentBay, AgentBayError, AuthenticationError, NotFoundError, RateLimitError,
    _detect_type, __version__,
)


class TestConstructor:
    def test_local_mode_no_args(self):
        """AgentBay() with no args → local mode."""
        with patch.dict(os.environ, {}, clear=True):
            brain = AgentBay()
            assert brain._is_local is True

    def test_local_mode_prints_upgrade_prompt(self, capsys, monkeypatch, tmp_path):
        monkeypatch.delenv("AGENTBAY_QUIET", raising=False)
        with patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True):
            brain = AgentBay()
        assert brain._is_local is True
        captured = capsys.readouterr()
        assert captured.out == (
            "Running locally. Run `agentbay login` to sync to cloud and unlock Teams (free, no card).\n"
        )

    def test_cloud_mode_with_key(self):
        """AgentBay(api_key=...) → cloud mode."""
        brain = AgentBay(api_key="ab_live_test123")
        assert brain._is_local is False
        assert brain.api_key == "ab_live_test123"

    def test_env_var_detection(self):
        """Picks up AGENTBAY_API_KEY from environment."""
        with patch.dict(os.environ, {"AGENTBAY_API_KEY": "ab_live_from_env"}):
            brain = AgentBay()
            assert brain._is_local is False
            assert brain.api_key == "ab_live_from_env"

    def test_explicit_key_overrides_env(self):
        """Explicit api_key beats env var."""
        with patch.dict(os.environ, {"AGENTBAY_API_KEY": "ab_live_env"}):
            brain = AgentBay(api_key="ab_live_explicit")
            assert brain.api_key == "ab_live_explicit"

    def test_project_id_stored(self):
        brain = AgentBay(api_key="ab_live_test", project_id="proj-123")
        assert brain.project_id == "proj-123"

    def test_base_url_default(self):
        brain = AgentBay(api_key="ab_live_test")
        assert "aiagentsbay.com" in brain.base_url

    def test_custom_base_url(self):
        brain = AgentBay(api_key="ab_live_test", base_url="http://localhost:3000")
        assert brain.base_url == "http://localhost:3000"

    def test_timeout_default(self):
        brain = AgentBay(api_key="ab_live_test")
        assert brain.timeout == 30


class TestVersion:
    def test_version_exists(self):
        assert __version__
        assert isinstance(__version__, str)

    def test_version_format(self):
        parts = __version__.split(".")
        assert len(parts) >= 2  # at least major.minor


class TestDetectType:
    def test_pitfall_detection(self):
        assert _detect_type("The bug was caused by a null pointer") == "PITFALL"

    def test_decision_detection(self):
        assert _detect_type("We decided to use PostgreSQL") == "DECISION"

    def test_procedure_detection(self):
        assert _detect_type("Step 1: install the package. Step 2: configure.") == "PROCEDURE"

    def test_pattern_default(self):
        assert _detect_type("Use JWT tokens for authentication") == "PATTERN"


class TestErrors:
    def test_agentbay_error(self):
        err = AgentBayError("test", status_code=500)
        assert str(err) == "test"
        assert err.status_code == 500

    def test_auth_error_has_help_url(self):
        err = AuthenticationError("bad key")
        assert err.help_url is not None
        assert "api-keys" in err.help_url

    def test_rate_limit_error_has_help_url(self):
        err = RateLimitError("too fast")
        assert err.help_url is not None
        assert "pricing" in err.help_url

    def test_not_found_error_has_help_url(self):
        err = NotFoundError("missing")
        assert err.help_url is not None

    def test_errors_are_exceptions(self):
        assert issubclass(AgentBayError, Exception)
        assert issubclass(AuthenticationError, AgentBayError)
        assert issubclass(NotFoundError, AgentBayError)
        assert issubclass(RateLimitError, AgentBayError)


class TestHandleResponse:
    def test_401_raises_auth_error(self):
        brain = AgentBay(api_key="ab_live_test")
        resp = MagicMock()
        resp.status_code = 401
        resp.url = "https://test.com"
        with pytest.raises(AuthenticationError):
            brain._handle_response(resp)

    def test_404_raises_not_found(self):
        brain = AgentBay(api_key="ab_live_test")
        resp = MagicMock()
        resp.status_code = 404
        resp.url = "https://test.com"
        with pytest.raises(NotFoundError):
            brain._handle_response(resp)

    def test_429_raises_rate_limit(self):
        brain = AgentBay(api_key="ab_live_test")
        resp = MagicMock()
        resp.status_code = 429
        resp.url = "https://test.com"
        with pytest.raises(RateLimitError):
            brain._handle_response(resp)

    def test_500_raises_server_error(self):
        brain = AgentBay(api_key="ab_live_test")
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "Internal Server Error"
        with pytest.raises(AgentBayError):
            brain._handle_response(resp)

    def test_200_returns_json(self):
        brain = AgentBay(api_key="ab_live_test")
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b'{"ok": true}'
        resp.json.return_value = {"ok": True}
        result = brain._handle_response(resp)
        assert result == {"ok": True}

    def test_204_returns_empty(self):
        brain = AgentBay(api_key="ab_live_test")
        resp = MagicMock()
        resp.status_code = 204
        resp.content = b""
        result = brain._handle_response(resp)
        assert result == {}


class TestLocalModeOperations:
    @patch.dict(os.environ, {}, clear=True)
    def test_store_in_local_mode(self):
        brain = AgentBay()
        assert brain._is_local
        result = brain.store("Test pattern", title="Test")
        assert result["id"]

    @patch.dict(os.environ, {}, clear=True)
    def test_recall_in_local_mode(self):
        brain = AgentBay()
        brain.store("JWT authentication pattern", title="Auth")
        results = brain.recall("auth")
        assert len(results) >= 1

    @patch.dict(os.environ, {}, clear=True)
    def test_add_in_local_mode(self):
        brain = AgentBay()
        brain.add("The user prefers dark mode")
        results = brain.search("dark mode")
        assert len(results) >= 1

    @patch.dict(os.environ, {}, clear=True)
    def test_health_in_local_mode(self):
        brain = AgentBay()
        h = brain.health()
        assert "total_entries" in h

    @patch.dict(os.environ, {}, clear=True)
    def test_forget_in_local_mode(self):
        brain = AgentBay()
        result = brain.store("To delete", title="Temp")
        brain.forget(result["id"])
