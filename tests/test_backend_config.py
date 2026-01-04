"""
Tests for agent backend configuration.

Tests the multi-backend support system that allows switching between
Claude Code and Codex CLI as the agent backend.
"""

import os
import pytest
import sys
from pathlib import Path
from unittest.mock import patch

# Add backend to path
backend_path = Path(__file__).parent.parent / "apps" / "backend"
sys.path.insert(0, str(backend_path))


class TestAgentBackend:
    """Tests for AgentBackend enum."""

    def test_backend_values(self):
        """AgentBackend should have claude and codex values."""
        from core.backend_config import AgentBackend

        assert AgentBackend.CLAUDE.value == "claude"
        assert AgentBackend.CODEX.value == "codex"

    def test_backend_is_string_enum(self):
        """AgentBackend values should be usable as strings."""
        from core.backend_config import AgentBackend

        assert str(AgentBackend.CLAUDE) == "AgentBackend.CLAUDE"
        assert AgentBackend.CLAUDE == "claude"
        assert AgentBackend.CODEX == "codex"


class TestGetAgentBackend:
    """Tests for get_agent_backend() function."""

    def test_default_is_claude(self):
        """Default backend should be Claude."""
        from core.backend_config import AgentBackend, get_agent_backend

        # Ensure AGENT_BACKEND is not set
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AGENT_BACKEND", None)
            backend = get_agent_backend()
            assert backend == AgentBackend.CLAUDE

    def test_claude_backend_from_env(self):
        """Should return Claude when AGENT_BACKEND=claude."""
        from core.backend_config import AgentBackend, get_agent_backend

        with patch.dict(os.environ, {"AGENT_BACKEND": "claude"}):
            backend = get_agent_backend()
            assert backend == AgentBackend.CLAUDE

    def test_codex_backend_from_env(self):
        """Should return Codex when AGENT_BACKEND=codex."""
        from core.backend_config import AgentBackend, get_agent_backend

        with patch.dict(os.environ, {"AGENT_BACKEND": "codex"}):
            backend = get_agent_backend()
            assert backend == AgentBackend.CODEX

    def test_case_insensitive(self):
        """Backend selection should be case-insensitive."""
        from core.backend_config import AgentBackend, get_agent_backend

        with patch.dict(os.environ, {"AGENT_BACKEND": "CLAUDE"}):
            assert get_agent_backend() == AgentBackend.CLAUDE

        with patch.dict(os.environ, {"AGENT_BACKEND": "Codex"}):
            assert get_agent_backend() == AgentBackend.CODEX

    def test_invalid_backend_raises(self):
        """Invalid backend should raise ValueError."""
        from core.backend_config import get_agent_backend

        with patch.dict(os.environ, {"AGENT_BACKEND": "invalid_backend"}):
            with pytest.raises(ValueError) as excinfo:
                get_agent_backend()
            assert "Invalid AGENT_BACKEND" in str(excinfo.value)
            assert "invalid_backend" in str(excinfo.value)


class TestGetDefaultModel:
    """Tests for get_default_model() function."""

    def test_claude_default_model(self):
        """Claude backend should have sonnet model as default."""
        from core.backend_config import AgentBackend, get_default_model

        model = get_default_model(AgentBackend.CLAUDE)
        assert "claude" in model.lower() or "sonnet" in model.lower()

    def test_codex_default_model(self):
        """Codex backend should have o4-mini as default."""
        from core.backend_config import AgentBackend, get_default_model

        model = get_default_model(AgentBackend.CODEX)
        assert model == "o4-mini"

    def test_model_override_from_env(self):
        """Model should be overridable via environment variable."""
        from core.backend_config import AgentBackend, get_default_model

        with patch.dict(os.environ, {"CODEX_MODEL": "o3"}):
            model = get_default_model(AgentBackend.CODEX)
            assert model == "o3"

        with patch.dict(os.environ, {"AUTO_BUILD_MODEL": "claude-opus-4-20250514"}):
            model = get_default_model(AgentBackend.CLAUDE)
            assert model == "claude-opus-4-20250514"


class TestBackendHelpers:
    """Tests for backend helper functions."""

    def test_is_claude_backend(self):
        """is_claude_backend() should return True for Claude."""
        from core.backend_config import is_claude_backend

        with patch.dict(os.environ, {"AGENT_BACKEND": "claude"}):
            assert is_claude_backend() is True

        with patch.dict(os.environ, {"AGENT_BACKEND": "codex"}):
            assert is_claude_backend() is False

    def test_is_codex_backend(self):
        """is_codex_backend() should return True for Codex."""
        from core.backend_config import is_codex_backend

        with patch.dict(os.environ, {"AGENT_BACKEND": "codex"}):
            assert is_codex_backend() is True

        with patch.dict(os.environ, {"AGENT_BACKEND": "claude"}):
            assert is_codex_backend() is False

    def test_display_name(self):
        """get_backend_display_name() should return human-readable names."""
        from core.backend_config import AgentBackend, get_backend_display_name

        claude_name = get_backend_display_name(AgentBackend.CLAUDE)
        assert "Claude" in claude_name

        codex_name = get_backend_display_name(AgentBackend.CODEX)
        assert "Codex" in codex_name


class TestCodexClient:
    """Tests for CodexCLIClient class."""

    def test_options_dataclass(self):
        """CodexClientOptions should have expected fields."""
        from core.codex_client import CodexClientOptions

        options = CodexClientOptions(
            model="o4-mini",
            cwd="/tmp/test",
        )
        assert options.model == "o4-mini"
        assert options.cwd == "/tmp/test"
        assert options.approval_mode == "full-auto"

    def test_is_codex_available(self):
        """is_codex_available() should check for codex CLI."""
        from core.codex_client import is_codex_available

        # Result depends on whether codex is installed
        result = is_codex_available()
        assert isinstance(result, bool)


class TestAuthBackendSupport:
    """Tests for backend-specific authentication."""

    def test_get_openai_api_key(self):
        """get_openai_api_key() should return OPENAI_API_KEY."""
        from core.auth import get_openai_api_key

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
            key = get_openai_api_key()
            assert key == "sk-test-key"

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            key = get_openai_api_key()
            assert key is None

    def test_require_openai_api_key_raises(self):
        """require_openai_api_key() should raise if not set."""
        from core.auth import require_openai_api_key

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError) as excinfo:
                require_openai_api_key()
            assert "OPENAI_API_KEY" in str(excinfo.value)

    def test_get_backend_auth_status(self):
        """get_backend_auth_status() should return status for all backends."""
        from core.auth import get_backend_auth_status

        status = get_backend_auth_status()
        assert "claude" in status
        assert "codex" in status
        assert "available" in status["claude"]
        assert "available" in status["codex"]
