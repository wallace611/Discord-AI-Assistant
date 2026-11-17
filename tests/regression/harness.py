"""Case files, stub tools and pass/fail checks shared by the regression tests."""

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from bot.agent import AGENT_PROMPT, AgentLoop, Done, TextDelta, ToolRegistry, ToolStarted
from bot.agent.loop import MAX_STEPS
from bot.agent.setup import build_registry
from bot.agent.tools.base import Tier, Tool, ToolCallRequest, ToolContext, ToolResult
from bot.agent.tools.run_code import SandboxConfig, run_code_tool
from bot.gateway.handler import SYSTEM_PROMPT
from bot.memory import InMemoryStore, MemoryService
from bot.providers.base import ChatProvider, Message, Usage
from bot.router import RouteDecision
from fakes import FakeProvider, make_settings

CASE_DIR = Path(__file__).parent
LIVE = os.environ.get("REGRESSION") == "1"
LIVE_SKIP_REASON = "live regression run; set REGRESSION=1 to call the real models"

# The intent the router reports when it could not classify at all.
FALLBACK_INTENT = "unknown"
USER_NAME = "tester"

# What each stubbed tool answers unless a case overrides it.
DEFAULT_TOOL_OUTPUTS: dict[str, str] = {
    "web_search": (
        "[1] Example result\nhttps://example.com/result\nA short snippet about the topic."
    ),
    "fetch_url": (
        "URL: https://example.com/article\nTitle: Example article\n\n"
        "This article explains the topic in three short paragraphs. The main point is "
        "that small, frequent releases are easier to review than large ones."
    ),
    "search_server_history": (
        "[1] 2026-11-10 14:30 UTC, https://discord.com/channels/1/2/3\n"
        "alice: let's use nightly pg_dump backups to the NAS\n"
        "bob: agreed, I'll set up the cron job tonight"
    ),
    "recall_memory": "- Works as a backend engineer\n- Mostly writes Rust",
    "remember": "Saved.",
    "create_thread": "Thread created: https://discord.com/channels/1/2/99",
    "pin_message": "Pinned the user's message.",
    "create_poll": "Poll posted: https://discord.com/channels/1/2/100",
    "run_code": "stdout:\n42\n\nexit code: 0",
}


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _read_cases(name: str, *, allowed: set[str], required: set[str]) -> list[dict[str, Any]]:
    rows = json.loads((CASE_DIR / name).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{name}: expected a JSON array of cases")
    for row in rows:
        unknown, missing = set(row) - allowed, required - set(row)
        if unknown or missing:
            raise ValueError(
                f"{name}: case {row.get('id')!r}: "
                f"unknown keys {sorted(unknown)}, missing keys {sorted(missing)}"
            )
    return rows


# --- router -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouterCase:
    id: str
    text: str
    tiers: tuple[str, ...]  # any of these is correct
    image_count: int = 0
    needs_vision: bool | None = None  # None: not checked
    intents: tuple[str, ...] = ()  # empty: not checked


def load_router_cases() -> list[RouterCase]:
    rows = _read_cases(
        "router_cases.json",
        allowed={"id", "text", "tier", "image_count", "needs_vision", "intent", "note"},
        required={"id", "text", "tier"},
    )
    return [
        RouterCase(
            id=row["id"],
            text=row["text"],
            tiers=_as_tuple(row["tier"]),
            image_count=row.get("image_count", 0),
            needs_vision=row.get("needs_vision"),
            intents=_as_tuple(row.get("intent")),
        )
        for row in rows
    ]


def check_route(case: RouterCase, decision: RouteDecision) -> list[str]:
    """Reasons ``decision`` is wrong for ``case``; empty when it passes."""
    if decision.intent == FALLBACK_INTENT:
        # The fallback tier is cloud_agent, which would otherwise pass many cases.
        return ["router fell back to its default: model unreachable or output unparseable"]
    problems: list[str] = []
    if decision.tier.value not in case.tiers:
        problems.append(f"tier {decision.tier.value!r}, expected one of {list(case.tiers)}")
    if case.needs_vision is not None and decision.needs_vision is not case.needs_vision:
        problems.append(f"needs_vision {decision.needs_vision}, expected {case.needs_vision}")
    if case.intents and decision.intent not in case.intents:
        problems.append(f"intent {decision.intent!r}, expected one of {list(case.intents)}")
    return problems


# --- agent ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentCase:
    id: str
    prompt: str
    first_tool: tuple[str, ...] = ()  # the first tool requested must be one of these
    no_tools: bool = False  # the model must answer without requesting any tool
    require_tools: tuple[str, ...] = ()  # each must be requested at some point
    forbid_tools: tuple[str, ...] = ()  # none may be requested, approved or not
    approve: bool = True  # the user's answer to every confirmation prompt
    max_calls: dict[str, int] = field(default_factory=dict)  # per-tool request cap
    answer_contains_any: tuple[str, ...] = ()  # case-insensitive substrings
    tool_outputs: dict[str, str] = field(default_factory=dict)  # canned output overrides

    def tool_names(self) -> set[str]:
        """Every tool this case mentions."""
        return {
            *self.first_tool,
            *self.require_tools,
            *self.forbid_tools,
            *self.max_calls,
            *self.tool_outputs,
        }


def load_agent_cases() -> list[AgentCase]:
    rows = _read_cases(
        "agent_cases.json",
        allowed={
            "id",
            "prompt",
            "first_tool",
            "no_tools",
            "require_tools",
            "forbid_tools",
            "approve",
            "max_calls",
            "answer_contains_any",
            "tool_outputs",
            "note",
        },
        required={"id", "prompt"},
    )
    return [
        AgentCase(
            id=row["id"],
            prompt=row["prompt"],
            first_tool=_as_tuple(row.get("first_tool")),
            no_tools=row.get("no_tools", False),
            require_tools=_as_tuple(row.get("require_tools")),
            forbid_tools=_as_tuple(row.get("forbid_tools")),
            approve=row.get("approve", True),
            max_calls=dict(row.get("max_calls", {})),
            answer_contains_any=_as_tuple(row.get("answer_contains_any")),
            tool_outputs=dict(row.get("tool_outputs", {})),
        )
        for row in rows
    ]


@dataclass(slots=True)
class AgentTrace:
    """What the model did during one case."""

    requested: list[str] = field(default_factory=list)  # tools the model asked for, in order
    arguments: list[dict[str, Any]] = field(default_factory=list)  # parallel to ``requested``
    confirmations: list[str] = field(default_factory=list)  # tools the user was asked about
    executed: list[str] = field(default_factory=list)  # stubs that actually ran
    answer: str = ""
    steps: int = 0
    usage: Usage = field(default_factory=Usage)


def reference_registry() -> ToolRegistry:
    """The production built-in tools (run_code included), wired to nothing real."""
    quiet = FakeProvider()
    memory = MemoryService(InMemoryStore(), chat=quiet, embedder=quiet)
    registry = build_registry(make_settings(), memory)
    # run_code is registered separately in production, after the sandbox is verified.
    registry.add(run_code_tool(SandboxConfig(image="regression", profile="regression")))
    return registry


def stub_registry(case: AgentCase, trace: AgentTrace) -> ToolRegistry:
    """Production tool names, descriptions, schemas and tiers with canned behaviour."""

    def stub(tool: Tool) -> Tool:
        output = case.tool_outputs.get(tool.name, DEFAULT_TOOL_OUTPUTS[tool.name])

        async def run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            trace.executed.append(tool.name)
            return ToolResult(output)

        return replace(tool, run=run)

    return ToolRegistry(stub(tool) for tool in reference_registry())


def initial_messages(case: AgentCase) -> list[Message]:
    """The request as the handler builds it when no memory context is available."""
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{AGENT_PROMPT}"},
        {"role": "user", "content": f"{USER_NAME}: {case.prompt}"},
    ]


async def run_agent_case(
    provider: ChatProvider, case: AgentCase, *, model: str | None = None
) -> AgentTrace:
    """Run ``case`` through the real agent loop with stubbed tools."""
    trace = AgentTrace()

    async def confirm(request: ToolCallRequest) -> bool:
        trace.confirmations.append(request.tool)
        return case.approve

    loop = AgentLoop(provider, stub_registry(case, trace), model=model)
    ctx = ToolContext(guild_id=1, channel_id=2, user_id=3, message_id=4)
    answer: list[str] = []
    async for event in loop.run(initial_messages(case), ctx, confirm=confirm):
        match event:
            case TextDelta(text):
                answer.append(text)
            case ToolStarted(name, arguments):
                trace.requested.append(name)
                trace.arguments.append(arguments)
            case Done(steps, usage, _):
                trace.steps = steps
                trace.usage = usage
    trace.answer = "".join(answer)
    return trace


def check_agent_trace(case: AgentCase, trace: AgentTrace) -> list[str]:
    """Reasons ``trace`` is wrong for ``case``; empty when it passes."""
    tiers = {tool.name: tool.tier for tool in reference_registry()}
    problems: list[str] = []

    unknown = sorted(set(trace.requested) - set(tiers))
    if unknown:
        problems.append(f"requested tools that do not exist: {unknown}")
    if case.no_tools and trace.requested:
        problems.append(f"expected no tool calls, got {trace.requested}")
    if case.first_tool and (not trace.requested or trace.requested[0] not in case.first_tool):
        first = trace.requested[0] if trace.requested else None
        problems.append(f"first tool {first!r}, expected one of {list(case.first_tool)}")
    for name in case.require_tools:
        if name not in trace.requested:
            problems.append(f"never requested {name!r}")
    for name in case.forbid_tools:
        if name in trace.requested:
            problems.append(f"requested forbidden tool {name!r}")
    for name, cap in case.max_calls.items():
        count = trace.requested.count(name)
        if count > cap:
            problems.append(f"requested {name!r} {count} times, at most {cap} allowed")

    # Invariants of the loop itself, checked for every case.
    side_effects = [name for name in trace.requested if tiers.get(name) is Tier.SIDE_EFFECT]
    if sorted(side_effects) != sorted(trace.confirmations):
        problems.append(
            f"side-effect requests {side_effects} do not match "
            f"confirmation prompts {trace.confirmations}"
        )
    if not case.approve:
        ran = sorted({name for name in trace.executed if tiers.get(name) is Tier.SIDE_EFFECT})
        if ran:
            problems.append(f"side-effect tools ran after the user declined: {ran}")
    if trace.steps > MAX_STEPS:
        problems.append(f"used {trace.steps} tool steps, the limit is {MAX_STEPS}")

    if not trace.answer.strip():
        problems.append("the final answer is empty")
    if case.answer_contains_any:
        lowered = trace.answer.lower()
        if not any(needle.lower() in lowered for needle in case.answer_contains_any):
            problems.append(f"answer mentions none of {list(case.answer_contains_any)}")
    return problems
