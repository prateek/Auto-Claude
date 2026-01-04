"""
Agent Client Factory
====================

Creates the appropriate "agent session client" based on selected backend.

Claude backend returns the existing `ClaudeSDKClient` from `core.client`.
Codex backend returns a `CodexCLIClient` wrapper that shells out to `codex`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.agent_backend import resolve_agent_backend
from core.client import create_client
from core.codex_cli_client import CodexCLIClient


def create_agent_session_client(
    project_dir: Path,
    spec_dir: Path,
    model: str,
    agent_type: str = "coder",
    max_thinking_tokens: int | None = None,
    output_format: dict | None = None,
    agents: dict | None = None,
    backend: str | None = None,
) -> Any:
    """
    Create an agent session client for the selected backend.

    Note: return type is intentionally `Any` because different backends expose
    different implementations; session runners are responsible for handling
    them.
    """

    selected = resolve_agent_backend(backend)
    if selected == "codex":
        # Codex doesn't support MCP/tool allowlists through our SDK client.
        # It operates directly on the working directory via its own CLI.
        return CodexCLIClient(project_dir=project_dir, spec_dir=spec_dir, model=model)

    # Default: Claude SDK
    return create_client(
        project_dir=project_dir,
        spec_dir=spec_dir,
        model=model,
        agent_type=agent_type,
        max_thinking_tokens=max_thinking_tokens,
        output_format=output_format,
        agents=agents,
    )

