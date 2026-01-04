"""
Core Framework Module
=====================

Core components for the Auto Claude autonomous coding framework.

Multi-Backend Support
---------------------
Auto Claude supports multiple AI agent backends:
- Claude Code (default): Uses Claude Agent SDK
- Codex CLI: Uses OpenAI's Codex CLI

Use `create_agent_client()` as the primary entry point for creating agent clients.
"""

# Note: We use lazy imports here because the full agent module has many dependencies
# that may not be needed for basic operations like workspace management.

__all__ = [
    "run_autonomous_agent",
    "run_followup_planner",
    "WorkspaceManager",
    "WorktreeManager",
    "ProgressTracker",
    # Client factories
    "create_agent_client",
    "create_client",
    "create_codex_client",
    # Backend configuration
    "AgentBackend",
    "get_agent_backend",
]


def __getattr__(name):
    """Lazy imports to avoid circular dependencies and heavy imports."""
    if name in ("run_autonomous_agent", "run_followup_planner"):
        from .agent import run_autonomous_agent, run_followup_planner

        return locals()[name]
    elif name == "WorkspaceManager":
        from .workspace import WorkspaceManager

        return WorkspaceManager
    elif name == "WorktreeManager":
        from .worktree import WorktreeManager

        return WorktreeManager
    elif name == "ProgressTracker":
        from .progress import ProgressTracker

        return ProgressTracker
    elif name in (
        "create_claude_client",
        "ClaudeClient",
        "create_agent_client",
        "create_client",
        "create_codex_client",
    ):
        from . import client as _client

        return getattr(_client, name)
    elif name in ("AgentBackend", "get_agent_backend"):
        from .backend_config import AgentBackend, get_agent_backend

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
