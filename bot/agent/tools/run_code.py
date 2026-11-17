"""Run a short script in a throw-away Incus container.

Each call launches an ephemeral container from ``INCUS_IMAGE`` with the
``INCUS_SANDBOX_PROFILE`` profile. The profile must have no network device;
``verify_sandbox`` checks that before the tool is offered. The container is
killed when the call finishes, times out, or prints too much.
"""

import asyncio
import contextlib
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from bot.agent.tools.base import (
    Tier,
    Tool,
    ToolContext,
    ToolError,
    ToolResult,
    object_schema,
    require_str,
)

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 4000
MAX_CODE_CHARS = 20_000
MAX_CAPTURE_BYTES = 256_000  # per stream; the process is killed beyond this
CONTAINER_PREFIX = "bot-sandbox-"

LANGUAGES: dict[str, list[str]] = {
    "python": ["python3", "-"],
    "sh": ["sh", "-s"],
}


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


type CommandRunner = Callable[[list[str], bytes | None, float], Awaitable[CommandResult]]


async def run_command(
    argv: list[str],
    stdin: bytes | None,
    timeout: float,
    *,
    max_capture: int = MAX_CAPTURE_BYTES,
) -> CommandResult:
    """Run a local command with a timeout and an output cap.

    The process is killed when the timeout passes (``TimeoutError`` is raised)
    or when either stream exceeds ``max_capture`` bytes (output is cut there).
    """
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    def kill() -> None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()

    async def pump(stream: asyncio.StreamReader) -> bytes:
        buffer = bytearray()
        while chunk := await stream.read(65536):
            if len(buffer) < max_capture:
                buffer.extend(chunk[: max_capture - len(buffer)])
            else:
                kill()  # keep draining so the process can exit
        return bytes(buffer)

    async def feed() -> None:
        if stdin is None or process.stdin is None:
            return
        try:
            process.stdin.write(stdin)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the process exited before reading everything
        finally:
            process.stdin.close()

    assert process.stdout is not None and process.stderr is not None
    try:
        out, err, _ = await asyncio.wait_for(
            asyncio.gather(pump(process.stdout), pump(process.stderr), feed()), timeout
        )
        await process.wait()
    except TimeoutError:
        kill()
        await process.wait()
        raise
    return CommandResult(
        process.returncode or 0,
        out.decode("utf-8", "replace"),
        err.decode("utf-8", "replace"),
    )


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    image: str
    profile: str
    timeout: float = 30.0
    memory: str = "256MiB"
    cpu: str = "1"
    processes: str = "64"
    # How many sandboxes may run at once; each one takes RAM on the host.
    concurrency: int = 2


_NIC_DEVICE = re.compile(r"^\s*type:\s*nic\s*$", re.MULTILINE)


async def verify_sandbox(config: SandboxConfig, runner: CommandRunner = run_command) -> str | None:
    """Return None when the sandbox profile is usable, else the reason it is not."""
    try:
        shown = await runner(["incus", "profile", "show", config.profile], None, 30.0)
    except FileNotFoundError:
        return "incus is not installed"
    except TimeoutError:
        return "incus did not answer"
    if shown.returncode != 0:
        return f"profile '{config.profile}' not found: {shown.stderr.strip()[:200]}"
    if _NIC_DEVICE.search(shown.stdout):
        return f"profile '{config.profile}' has a network device; the sandbox must be offline"
    return None


async def remove_stale_sandboxes(runner: CommandRunner = run_command) -> int:
    """Delete sandboxes left behind by an earlier run. Returns how many."""
    listed = await runner(
        ["incus", "list", f"^{CONTAINER_PREFIX}", "--format", "csv", "--columns", "n"], None, 30.0
    )
    names = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    for name in names:
        await runner(["incus", "delete", "--force", name], None, 30.0)
        logger.warning("removed stale sandbox", extra={"container": name})
    return len(names)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


def run_code_tool(config: SandboxConfig, runner: CommandRunner = run_command) -> Tool:
    slots = asyncio.Semaphore(config.concurrency)

    async def run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        code = require_str(args, "code", max_len=MAX_CODE_CHARS)
        language = args.get("language", "python")
        if not isinstance(language, str) or language not in LANGUAGES:
            raise ToolError(f"language must be one of {', '.join(LANGUAGES)}")
        name = f"{CONTAINER_PREFIX}{uuid.uuid4().hex[:12]}"
        launch = [
            "incus", "launch", config.image, name,
            "--ephemeral",
            "--profile", config.profile,
            "--config", f"limits.memory={config.memory}",
            "--config", f"limits.cpu={config.cpu}",
            "--config", f"limits.processes={config.processes}",
            "--config", "security.privileged=false",
        ]  # fmt: skip
        exec_cmd = [
            "incus", "exec", name,
            "--user", "65534", "--group", "65534", "--cwd", "/tmp", "--env", "HOME=/tmp",
            "--", *LANGUAGES[language],
        ]  # fmt: skip
        async with slots:
            try:
                try:
                    launched = await runner(launch, None, 60.0)
                except TimeoutError:
                    raise ToolError("the sandbox took too long to start") from None
                if launched.returncode != 0:
                    logger.error("sandbox launch failed", extra={"stderr": launched.stderr[:500]})
                    raise ToolError("the sandbox could not be started")
                try:
                    result = await runner(exec_cmd, code.encode("utf-8"), config.timeout)
                except TimeoutError:
                    return ToolResult(f"Timed out after {config.timeout:.0f} seconds.")
            finally:
                # Ephemeral containers vanish when stopped; force it so a stuck
                # process cannot keep the sandbox alive.
                try:
                    await runner(["incus", "delete", "--force", name], None, 30.0)
                except Exception:
                    logger.exception("sandbox cleanup failed", extra={"container": name})

        parts = []
        if result.stdout.strip():
            parts.append("stdout:\n" + _clip(result.stdout.rstrip(), MAX_OUTPUT_CHARS))
        if result.stderr.strip():
            parts.append("stderr:\n" + _clip(result.stderr.rstrip(), MAX_OUTPUT_CHARS // 2))
        parts.append(f"exit code: {result.returncode}")
        return ToolResult("\n\n".join(parts))

    return Tool(
        name="run_code",
        description=(
            "Run a Python 3 or sh script in an isolated sandbox with no network and a "
            f"{config.timeout:.0f}s limit. Print what you want to see. Nothing persists."
        ),
        tier=Tier.SANDBOX,
        parameters=object_schema(
            {
                "code": {"type": "string", "description": "The script to run"},
                "language": {
                    "type": "string",
                    "enum": list(LANGUAGES),
                    "description": "Default python",
                },
            },
            required=["code"],
        ),
        run=run,
    )
