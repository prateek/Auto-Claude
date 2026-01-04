"""
Codex CLI Agent Client
======================

Implements a minimal async client wrapper around the external `codex` CLI.

This is intentionally lightweight and **optional**:
- It is only used when AUTO_CLAUDE_AGENT_BACKEND=codex (or CLI override)
- If `codex` is not installed, we fail with a clear error message.

Configuration (env vars):
- CODEX_CLI_COMMAND: executable name/path (default: "codex")
- CODEX_CLI_ARGS: arguments as JSON array or shell-like string.
  Placeholders supported: {model}, {prompt_file}, {cwd}
  Default: ["exec", "--model", "{model}", "--prompt-file", "{prompt_file}"]
- CODEX_MODEL: default model name to use when the provided model looks like a Claude id.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class CodexNotInstalledError(RuntimeError):
    pass


def _looks_like_claude_model(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("claude-") or "claude" in m


def _parse_args(value: str | None) -> list[str]:
    if not value:
        return []
    v = value.strip()
    if not v:
        return []
    if v.startswith("["):
        try:
            parsed = json.loads(v)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    return shlex.split(v)


def _default_args() -> list[str]:
    # The default is best-effort; users can override fully via CODEX_CLI_ARGS.
    return ["exec", "--model", "{model}", "--prompt-file", "{prompt_file}"]


@dataclass(slots=True)
class CodexCLIClient:
    project_dir: Path
    spec_dir: Path
    model: str

    def __post_init__(self) -> None:
        self._command = os.environ.get("CODEX_CLI_COMMAND", "codex").strip() or "codex"
        self._args_template = _parse_args(os.environ.get("CODEX_CLI_ARGS")) or _default_args()

        if shutil.which(self._command) is None:
            raise CodexNotInstalledError(
                "Codex CLI backend selected but `codex` was not found on PATH.\n"
                "Install it and/or set CODEX_CLI_COMMAND.\n"
                "Example:\n"
                "  npm i -g @openai/codex  (or follow Codex CLI install docs)\n"
            )

    async def __aenter__(self) -> "CodexCLIClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def _resolve_model(self) -> str:
        # If callers pass a Claude model id (common today), avoid blindly feeding it to Codex.
        if _looks_like_claude_model(self.model):
            return os.environ.get("CODEX_MODEL", "").strip() or "gpt-5"
        return self.model

    async def run(self, prompt: str, on_line: Callable[[str], None] | None = None) -> str:
        """
        Run Codex CLI once with the given prompt, streaming stdout/stderr to callers.

        Returns the full combined stdout text.
        """

        prompt_file = self.spec_dir / ".codex_prompt.md"
        prompt_file.write_text(prompt, encoding="utf-8")

        model = self._resolve_model()
        cwd = str(self.project_dir.resolve())

        args = [
            a.format(model=model, prompt_file=str(prompt_file.resolve()), cwd=cwd)
            for a in self._args_template
        ]

        proc = await asyncio.create_subprocess_exec(
            self._command,
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=os.environ.copy(),
        )

        assert proc.stdout is not None
        out_chunks: list[str] = []
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode(errors="replace")
            out_chunks.append(text)
            if on_line is not None:
                on_line(text)
        code = await proc.wait()

        output = "".join(out_chunks)
        if code != 0:
            raise RuntimeError(f"Codex CLI exited with code {code}\n\n{output}")
        return output

