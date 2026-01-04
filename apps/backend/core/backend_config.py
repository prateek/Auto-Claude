"""
Agent Backend Configuration
===========================

Configures which AI agent backend to use for code generation:
- claude: Claude Code via Claude Agent SDK (default)
- codex: OpenAI Codex CLI

The backend is selected via the AGENT_BACKEND environment variable.
Each backend has its own authentication and configuration requirements.

Usage:
    from core.backend_config import get_agent_backend, AgentBackend

    backend = get_agent_backend()
    if backend == AgentBackend.CODEX:
        # Use Codex CLI
        ...
"""

import os
from enum import Enum

# =============================================================================
# Backend Types
# =============================================================================


class AgentBackend(str, Enum):
    """Supported agent backends."""

    CLAUDE = "claude"  # Claude Code via Claude Agent SDK (default)
    CODEX = "codex"  # OpenAI Codex CLI


# =============================================================================
# Backend Configuration
# =============================================================================

# Default backend if not specified
DEFAULT_BACKEND = AgentBackend.CLAUDE

# Environment variable names
BACKEND_ENV_VAR = "AGENT_BACKEND"

# Backend-specific model defaults
BACKEND_MODEL_DEFAULTS = {
    AgentBackend.CLAUDE: "claude-sonnet-4-5-20250929",
    AgentBackend.CODEX: "o4-mini",
}

# Backend-specific model environment variables
BACKEND_MODEL_ENV_VARS = {
    AgentBackend.CLAUDE: "AUTO_BUILD_MODEL",
    AgentBackend.CODEX: "CODEX_MODEL",
}


def get_agent_backend() -> AgentBackend:
    """
    Get the configured agent backend.

    Returns:
        AgentBackend enum value

    Environment:
        AGENT_BACKEND: "claude" or "codex" (default: "claude")
    """
    backend_str = os.environ.get(BACKEND_ENV_VAR, DEFAULT_BACKEND.value).lower().strip()

    try:
        return AgentBackend(backend_str)
    except ValueError:
        valid_backends = [b.value for b in AgentBackend]
        raise ValueError(
            f"Invalid AGENT_BACKEND: '{backend_str}'. "
            f"Valid options: {', '.join(valid_backends)}"
        )


def get_default_model(backend: AgentBackend | None = None) -> str:
    """
    Get the default model for the specified backend.

    Args:
        backend: Agent backend (uses current backend if not specified)

    Returns:
        Default model name for the backend
    """
    if backend is None:
        backend = get_agent_backend()

    # Check for model override via environment variable
    env_var = BACKEND_MODEL_ENV_VARS.get(backend)
    if env_var:
        model = os.environ.get(env_var)
        if model:
            return model

    return BACKEND_MODEL_DEFAULTS.get(backend, BACKEND_MODEL_DEFAULTS[AgentBackend.CLAUDE])


def is_claude_backend() -> bool:
    """Check if using Claude Code backend."""
    return get_agent_backend() == AgentBackend.CLAUDE


def is_codex_backend() -> bool:
    """Check if using Codex CLI backend."""
    return get_agent_backend() == AgentBackend.CODEX


def get_backend_display_name(backend: AgentBackend | None = None) -> str:
    """
    Get a human-readable name for the backend.

    Args:
        backend: Agent backend (uses current backend if not specified)

    Returns:
        Display name string
    """
    if backend is None:
        backend = get_agent_backend()

    names = {
        AgentBackend.CLAUDE: "Claude Code (Claude Agent SDK)",
        AgentBackend.CODEX: "OpenAI Codex CLI",
    }
    return names.get(backend, backend.value)


def validate_backend_requirements() -> list[str]:
    """
    Validate that all requirements for the current backend are met.

    Returns:
        List of error messages (empty if all requirements met)
    """
    backend = get_agent_backend()
    errors = []

    if backend == AgentBackend.CLAUDE:
        # Check for Claude authentication
        from core.auth import get_auth_token

        if not get_auth_token():
            errors.append(
                "Claude backend requires authentication. "
                "Run 'claude setup-token' or set CLAUDE_CODE_OAUTH_TOKEN."
            )

    elif backend == AgentBackend.CODEX:
        # Check for Codex CLI availability
        from core.codex_client import is_codex_available

        if not is_codex_available():
            errors.append(
                "Codex CLI not found. Install with: npm install -g @openai/codex"
            )

        # Check for OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            errors.append(
                "Codex backend requires OPENAI_API_KEY environment variable."
            )

    return errors


def print_backend_info() -> None:
    """Print information about the current backend configuration."""
    backend = get_agent_backend()
    model = get_default_model(backend)

    print(f"Agent Backend: {get_backend_display_name(backend)}")
    print(f"Default Model: {model}")

    # Validate requirements
    errors = validate_backend_requirements()
    if errors:
        print("⚠️  Configuration Issues:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✓ Configuration valid")
