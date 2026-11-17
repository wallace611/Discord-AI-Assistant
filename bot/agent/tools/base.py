"""Tool interface for the agent loop.

Every tool declares a permission tier:

1. ``READ_ONLY``   runs automatically.
2. ``SANDBOX``     runs automatically inside an isolated container.
3. ``SIDE_EFFECT`` changes something visible; the user must confirm first.

Tool output is always treated as data, never as instructions.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol


class Tier(IntEnum):
    READ_ONLY = 1
    SANDBOX = 2
    SIDE_EFFECT = 3


class ToolError(Exception):
    """A tool could not do its job; the message is shown to the model."""


class DiscordActions(Protocol):
    """Side effects on the conversation the request came from."""

    async def create_thread(self, name: str) -> str:
        """Open a thread on the user's message. Returns its jump URL."""
        ...

    async def pin_message(self) -> None:
        """Pin the user's message."""
        ...

    async def create_poll(self, question: str, options: list[str], hours: int) -> str:
        """Post a poll in the channel. Returns its jump URL."""
        ...


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Where a tool call comes from. Tools must stay inside this scope."""

    guild_id: int | None
    channel_id: int
    user_id: int
    message_id: int
    discord: DiscordActions | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: str
    sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolCallRequest:
    """What the user is asked to approve for a side-effect tool."""

    tool: str
    description: str
    arguments: dict[str, Any]


type Confirmer = Callable[[ToolCallRequest], Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class Tool:
    """A callable the model may invoke.

    ``parameters`` is a JSON schema object describing the arguments.
    """

    name: str
    description: str
    tier: Tier
    run: Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "additionalProperties": False}
    )

    def spec(self) -> dict[str, Any]:
        """OpenAI function-calling description of this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def object_schema(
    properties: dict[str, dict[str, Any]], required: list[str] | None = None
) -> dict[str, Any]:
    """JSON schema for an object with the given properties."""
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False,
    }


def require_str(args: dict[str, Any], name: str, *, max_len: int = 2000) -> str:
    value = args.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"'{name}' must be a non-empty string")
    return value.strip()[:max_len]


def optional_int(args: dict[str, Any], name: str, default: int, *, lo: int, hi: int) -> int:
    value = args.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"'{name}' must be an integer")
    return max(lo, min(hi, value))
