"""
Agent Backend Selection
========================

Auto Claude can run its autonomous agent sessions using different backends:

- "claude" (default): Claude Code via claude-agent-sdk
- "codex": OpenAI Codex CLI (external command)

This module centralizes how we resolve the selected backend from CLI/env.
"""

from __future__ import annotations

import os
from typing import Literal

AgentBackend = Literal["claude", "codex"]


def resolve_agent_backend(value: str | None = None) -> AgentBackend:
    """
    Resolve the agent backend from an explicit value or environment.

    Precedence:
    1) `value` argument (typically from CLI)
    2) `AUTO_CLAUDE_AGENT_BACKEND` env var
    3) default: "claude"
    """

    raw = (
        (value or "").strip()
        or os.environ.get("AUTO_CLAUDE_AGENT_BACKEND", "").strip()
        or "claude"
    ).lower()

    if raw in ("claude", "claude-code", "claude_code", "sdk"):
        return "claude"
    if raw in ("codex", "openai-codex", "openai_codex", "codex-cli", "codex_cli"):
        return "codex"

    # Be conservative: unknown values fall back to Claude to preserve current behavior.
    return "claude"

