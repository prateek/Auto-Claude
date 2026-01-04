"""
Codex CLI Client Wrapper
========================

Provides a client wrapper for the OpenAI Codex CLI that implements a compatible
interface with the Claude SDK client. This allows Auto Claude to use Codex CLI
as an alternative agent backend.

The Codex CLI is OpenAI's terminal-based AI coding assistant similar to Claude Code.
It runs as a subprocess and provides agentic coding capabilities.

Installation:
    npm install -g @openai/codex

Usage:
    from core.codex_client import CodexCLIClient, CodexClientOptions

    client = CodexCLIClient(options=CodexClientOptions(
        model="o4-mini",
        cwd="/path/to/project",
        env={"OPENAI_API_KEY": "sk-..."}
    ))

    async with client:
        await client.query("Implement feature X")
        async for msg in client.receive_response():
            # Process messages...
"""

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """Represents a text content block (compatible with Claude SDK TextBlock)."""

    text: str


@dataclass
class ToolUseBlock:
    """Represents a tool use content block (compatible with Claude SDK ToolUseBlock)."""

    name: str
    input: dict[str, Any] | None = None
    id: str = ""


@dataclass
class ToolResultBlock:
    """Represents a tool result content block (compatible with Claude SDK ToolResultBlock)."""

    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class AssistantMessage:
    """Represents an assistant message (compatible with Claude SDK AssistantMessage)."""

    content: list[TextBlock | ToolUseBlock]


@dataclass
class UserMessage:
    """Represents a user message with tool results (compatible with Claude SDK UserMessage)."""

    content: list[ToolResultBlock]


@dataclass
class CodexClientOptions:
    """Configuration options for Codex CLI client."""

    # Model to use (e.g., "o4-mini", "o3", "gpt-4.1")
    model: str = "o4-mini"

    # System prompt for the agent
    system_prompt: str | None = None

    # Working directory for file operations
    cwd: str | None = None

    # Environment variables to pass to the subprocess
    env: dict[str, str] = field(default_factory=dict)

    # Maximum conversation turns (Codex uses --max-turns)
    max_turns: int = 100

    # Approval mode: "suggest" (default), "auto-edit", or "full-auto"
    approval_mode: str = "full-auto"

    # Allowed tools list (used for permissions)
    allowed_tools: list[str] = field(default_factory=list)

    # Additional CLI arguments
    extra_args: list[str] = field(default_factory=list)


class CodexCLIClient:
    """
    Client wrapper for OpenAI Codex CLI.

    Implements a compatible interface with ClaudeSDKClient for seamless
    integration with the Auto Claude agent system.

    The client runs the Codex CLI as a subprocess and parses its output
    to generate compatible message objects.
    """

    def __init__(self, options: CodexClientOptions):
        """
        Initialize the Codex CLI client.

        Args:
            options: Configuration options for the client
        """
        self.options = options
        self._process: asyncio.subprocess.Process | None = None
        self._response_queue: asyncio.Queue[AssistantMessage | UserMessage | None] = (
            asyncio.Queue()
        )
        self._current_query: str | None = None
        self._connected = False
        self._output_task: asyncio.Task | None = None

    async def __aenter__(self) -> "CodexCLIClient":
        """Async context manager entry - connects to Codex CLI."""
        await self._connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - disconnects from Codex CLI."""
        await self._disconnect()

    async def _connect(self) -> None:
        """Initialize connection to Codex CLI subprocess."""
        if self._connected:
            return

        # Verify codex is installed
        codex_path = shutil.which("codex")
        if not codex_path:
            raise RuntimeError(
                "Codex CLI not found. Install with: npm install -g @openai/codex\n"
                "See: https://github.com/openai/codex"
            )

        logger.debug(f"Codex CLI found at: {codex_path}")
        self._connected = True

    async def _disconnect(self) -> None:
        """Clean up Codex CLI subprocess."""
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception as e:
                logger.warning(f"Error terminating Codex process: {e}")
            finally:
                self._process = None

        if self._output_task and not self._output_task.done():
            self._output_task.cancel()
            try:
                await self._output_task
            except asyncio.CancelledError:
                pass

        self._connected = False

    async def query(self, message: str) -> None:
        """
        Send a query to the Codex CLI.

        Args:
            message: The prompt/query to send

        Raises:
            RuntimeError: If client is not connected
        """
        if not self._connected:
            raise RuntimeError("Client not connected. Use 'async with client:' pattern.")

        self._current_query = message

        # Build command arguments
        cmd = ["codex"]

        # Add model
        cmd.extend(["--model", self.options.model])

        # Add approval mode (full-auto for non-interactive)
        cmd.extend(["--approval-mode", self.options.approval_mode])

        # Add working directory if specified
        if self.options.cwd:
            # Codex uses --cd to set working directory
            cmd.extend(["--cd", self.options.cwd])

        # Add system prompt if specified
        # Codex supports custom instructions via --instructions flag or instructions file
        if self.options.system_prompt:
            # Write system prompt to a temp file for Codex
            instructions_file = Path(self.options.cwd or ".") / ".codex-instructions"
            try:
                instructions_file.write_text(self.options.system_prompt, encoding="utf-8")
                # Codex will auto-detect this file in the working directory
            except Exception as e:
                logger.warning(f"Could not write instructions file: {e}")

        # Add extra arguments
        cmd.extend(self.options.extra_args)

        # Add the actual prompt at the end
        cmd.append(message)

        # Prepare environment
        env = os.environ.copy()
        env.update(self.options.env)

        # Ensure OPENAI_API_KEY is set
        if "OPENAI_API_KEY" not in env:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Required for Codex CLI.\n"
                "Set OPENAI_API_KEY in your .env file."
            )

        logger.debug(f"Starting Codex CLI: {' '.join(cmd)}")

        # Start the subprocess
        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.options.cwd,
                env=env,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Codex CLI not found. Install with: npm install -g @openai/codex"
            )

        # Start background task to process output
        self._output_task = asyncio.create_task(self._process_output())

    async def _process_output(self) -> None:
        """
        Process Codex CLI output and convert to compatible message format.

        Codex CLI outputs structured information that we parse and convert
        to Claude SDK-compatible message objects.
        """
        if not self._process or not self._process.stdout:
            return

        current_text = ""
        current_tool: str | None = None
        current_tool_input: dict | None = None

        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break

                decoded = line.decode("utf-8", errors="replace")
                logger.debug(f"Codex output: {decoded.strip()}")

                # Parse Codex output format
                # Codex outputs text and tool usage in a structured way
                # We convert these to compatible message objects

                # Check for tool invocation patterns
                # Codex uses patterns like: [tool_name] or Running: command
                if decoded.startswith("[") and "]" in decoded:
                    # Tool invocation
                    tool_end = decoded.index("]")
                    tool_name = decoded[1:tool_end].strip()
                    tool_detail = decoded[tool_end + 1 :].strip()

                    # Flush any pending text
                    if current_text.strip():
                        await self._response_queue.put(
                            AssistantMessage(content=[TextBlock(text=current_text)])
                        )
                        current_text = ""

                    # Map Codex tool names to Claude-compatible names
                    tool_mapping = {
                        "read": "Read",
                        "write": "Write",
                        "edit": "Edit",
                        "bash": "Bash",
                        "shell": "Bash",
                        "run": "Bash",
                        "grep": "Grep",
                        "glob": "Glob",
                        "search": "Grep",
                    }
                    mapped_name = tool_mapping.get(
                        tool_name.lower(), tool_name.capitalize()
                    )

                    # Parse tool input
                    tool_input = {"raw": tool_detail} if tool_detail else {}
                    if "path" in tool_detail.lower():
                        # Try to extract file path
                        parts = tool_detail.split()
                        if parts:
                            tool_input["file_path"] = parts[-1]
                    elif tool_name.lower() in ("bash", "shell", "run"):
                        tool_input["command"] = tool_detail

                    await self._response_queue.put(
                        AssistantMessage(
                            content=[ToolUseBlock(name=mapped_name, input=tool_input)]
                        )
                    )
                    current_tool = mapped_name
                    current_tool_input = tool_input

                elif decoded.startswith("Running:") or decoded.startswith("$ "):
                    # Shell command execution
                    cmd = decoded.replace("Running:", "").replace("$ ", "").strip()

                    if current_text.strip():
                        await self._response_queue.put(
                            AssistantMessage(content=[TextBlock(text=current_text)])
                        )
                        current_text = ""

                    await self._response_queue.put(
                        AssistantMessage(
                            content=[ToolUseBlock(name="Bash", input={"command": cmd})]
                        )
                    )
                    current_tool = "Bash"

                elif decoded.startswith("✓") or decoded.startswith("Done"):
                    # Tool completion
                    if current_tool:
                        await self._response_queue.put(
                            UserMessage(
                                content=[
                                    ToolResultBlock(
                                        tool_use_id=current_tool,
                                        content=decoded.strip(),
                                        is_error=False,
                                    )
                                ]
                            )
                        )
                        current_tool = None

                elif decoded.startswith("✗") or decoded.startswith("Error"):
                    # Tool error
                    if current_tool:
                        await self._response_queue.put(
                            UserMessage(
                                content=[
                                    ToolResultBlock(
                                        tool_use_id=current_tool,
                                        content=decoded.strip(),
                                        is_error=True,
                                    )
                                ]
                            )
                        )
                        current_tool = None

                else:
                    # Regular text output
                    current_text += decoded

            # Flush remaining text
            if current_text.strip():
                await self._response_queue.put(
                    AssistantMessage(content=[TextBlock(text=current_text)])
                )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error processing Codex output: {e}")
        finally:
            # Signal end of response
            await self._response_queue.put(None)

    async def receive_response(self) -> AsyncIterator[AssistantMessage | UserMessage]:
        """
        Receive response messages from Codex CLI.

        Yields:
            AssistantMessage or UserMessage objects compatible with Claude SDK
        """
        while True:
            msg = await self._response_queue.get()
            if msg is None:
                # End of response stream
                break
            yield msg

        # Wait for process to complete
        if self._process:
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10.0)

                # Check for errors
                if self._process.returncode != 0 and self._process.stderr:
                    stderr = await self._process.stderr.read()
                    if stderr:
                        error_msg = stderr.decode("utf-8", errors="replace")
                        logger.warning(f"Codex CLI exited with error: {error_msg}")

            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for Codex process to complete")


def is_codex_available() -> bool:
    """
    Check if Codex CLI is installed and available.

    Returns:
        True if Codex CLI is found in PATH
    """
    return shutil.which("codex") is not None


def get_codex_version() -> str | None:
    """
    Get the installed Codex CLI version.

    Returns:
        Version string or None if not available
    """
    import subprocess

    try:
        result = subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None
