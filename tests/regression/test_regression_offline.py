"""Offline checks: the case files are sound and the harness judges runs correctly.

Everything here uses scripted providers and stub tools; nothing leaves the process.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from bot.agent import AGENT_PROMPT
from bot.agent.tools.base import Tier, ToolContext
from bot.router import RouteDecision
from bot.router import Tier as RouteTier
from bot.router.router import INTENTS
from fakes import FakeProvider, text_stream, tool_call_stream
from regression import harness
from regression.harness import (
    DEFAULT_TOOL_OUTPUTS,
    AgentCase,
    AgentTrace,
    RouterCase,
    check_agent_trace,
    check_route,
    load_agent_cases,
    load_router_cases,
    reference_registry,
    run_agent_case,
    stub_registry,
)

ROUTER_CASES = load_router_cases()
AGENT_CASES = load_agent_cases()
AGENT_CASES_BY_ID = {case.id: case for case in AGENT_CASES}
TOOL_NAMES = {tool.name for tool in reference_registry()}
CTX = ToolContext(guild_id=1, channel_id=2, user_id=3, message_id=4)


# --- router case file -------------------------------------------------------


def test_router_case_ids_are_unique() -> None:
    duplicates = [name for name, count in Counter(c.id for c in ROUTER_CASES).items() if count > 1]
    assert duplicates == []


@pytest.mark.parametrize("case", ROUTER_CASES, ids=lambda case: case.id)
def test_router_case_is_well_formed(case: RouterCase) -> None:
    assert case.text.strip()
    assert case.tiers and set(case.tiers) <= {tier.value for tier in RouteTier}
    assert set(case.intents) <= set(INTENTS)
    assert case.image_count >= 0
    if case.needs_vision:
        # The router clears needs_vision when nothing is attached.
        assert case.image_count > 0


@pytest.mark.parametrize("tier", list(RouteTier))
def test_every_tier_has_unambiguous_router_cases(tier: RouteTier) -> None:
    exact = [case for case in ROUTER_CASES if case.tiers == (tier.value,)]
    assert len(exact) >= 3


# --- agent case file --------------------------------------------------------


def test_agent_case_ids_are_unique() -> None:
    duplicates = [name for name, count in Counter(c.id for c in AGENT_CASES).items() if count > 1]
    assert duplicates == []


@pytest.mark.parametrize("case", AGENT_CASES, ids=lambda case: case.id)
def test_agent_case_is_well_formed(case: AgentCase) -> None:
    assert case.prompt.strip()
    assert case.tool_names() <= TOOL_NAMES, f"unknown tools: {case.tool_names() - TOOL_NAMES}"
    # Every case must be able to fail.
    assert case.no_tools or case.first_tool or case.require_tools or case.forbid_tools
    if case.no_tools:
        assert not case.first_tool and not case.require_tools
    assert not set(case.forbid_tools) & {*case.first_tool, *case.require_tools}
    assert all(cap >= 1 for cap in case.max_calls.values())


def test_every_built_in_tool_is_exercised_by_some_agent_case() -> None:
    expected = {name for case in AGENT_CASES for name in (*case.first_tool, *case.require_tools)}
    assert expected == TOOL_NAMES


def test_agent_cases_cover_the_safety_behaviours() -> None:
    assert any(case.no_tools for case in AGENT_CASES)
    assert any(not case.approve for case in AGENT_CASES)
    assert sum(1 for case in AGENT_CASES if case.id.startswith("injection-")) >= 2


# --- stub tools -------------------------------------------------------------


def test_every_built_in_tool_has_a_canned_output() -> None:
    # A new built-in tool needs a canned output here and at least one agent case.
    assert set(DEFAULT_TOOL_OUTPUTS) == TOOL_NAMES


def test_stub_tools_keep_the_production_specs_and_tiers() -> None:
    stubs = stub_registry(AgentCase(id="x", prompt="x"), AgentTrace())
    real = reference_registry()

    assert stubs.specs() == real.specs()
    assert {tool.name: tool.tier for tool in stubs} == {tool.name: tool.tier for tool in real}
    assert any(tool.tier is Tier.SIDE_EFFECT for tool in stubs)
    assert any(tool.tier is Tier.SANDBOX for tool in stubs)


async def test_stub_tools_return_canned_output_and_record_the_call() -> None:
    case = AgentCase(id="x", prompt="x", tool_outputs={"web_search": "overridden"})
    trace = AgentTrace()
    stubs = stub_registry(case, trace)

    for tool in stubs:
        result = await tool.run({"anything": "goes"}, CTX)
        expected = "overridden" if tool.name == "web_search" else DEFAULT_TOOL_OUTPUTS[tool.name]
        assert result.content == expected

    assert trace.executed == [tool.name for tool in stubs]


# --- running a case with a scripted model -----------------------------------


async def test_run_agent_case_records_a_tool_call_and_the_answer() -> None:
    case = AGENT_CASES_BY_ID["current-info-uses-web-search"]
    provider = FakeProvider(
        streams=[
            tool_call_stream(("web_search", {"query": "postgresql latest stable"})),
            text_stream("最新穩定版是 19.1，", "來源：https://www.postgresql.org/docs/release/"),
        ]
    )

    trace = await run_agent_case(provider, case, model="some-model")

    assert trace.requested == ["web_search"]
    assert trace.arguments == [{"query": "postgresql latest stable"}]
    assert trace.executed == ["web_search"]
    assert trace.confirmations == []
    assert trace.steps == 1
    assert trace.answer.endswith("https://www.postgresql.org/docs/release/")
    assert check_agent_trace(case, trace) == []

    first, second = provider.stream_calls
    assert first["model"] == "some-model"
    assert {spec["function"]["name"] for spec in first["tools"]} == TOOL_NAMES
    system, user = first["messages"][:2]
    assert system["role"] == "system" and AGENT_PROMPT in system["content"]
    assert user == {"role": "user", "content": f"{harness.USER_NAME}: {case.prompt}"}
    # The model saw the case's canned output, wrapped as data.
    tool_turn = second["messages"][-1]
    assert tool_turn["role"] == "tool"
    assert case.tool_outputs["web_search"] in tool_turn["content"]
    assert tool_turn["content"].startswith("<tool_output")


async def test_run_agent_case_asks_for_confirmation_and_runs_when_approved() -> None:
    case = AGENT_CASES_BY_ID["pin-on-request"]
    provider = FakeProvider(
        streams=[tool_call_stream(("pin_message", {})), text_stream("已經幫你釘選了。")]
    )

    trace = await run_agent_case(provider, case)

    assert trace.confirmations == ["pin_message"]
    assert trace.executed == ["pin_message"]
    assert check_agent_trace(case, trace) == []


async def test_run_agent_case_does_not_run_a_declined_side_effect() -> None:
    case = AGENT_CASES_BY_ID["declined-confirmation-is-not-retried"]
    provider = FakeProvider(
        streams=[tool_call_stream(("pin_message", {})), text_stream("好的，已取消釘選。")]
    )

    trace = await run_agent_case(provider, case)

    assert trace.requested == ["pin_message"]
    assert trace.confirmations == ["pin_message"]
    assert trace.executed == []
    assert check_agent_trace(case, trace) == []
    assert "declined" in provider.stream_calls[1]["messages"][-1]["content"]


async def test_a_model_that_retries_after_a_decline_fails_the_case() -> None:
    case = AGENT_CASES_BY_ID["declined-confirmation-is-not-retried"]
    provider = FakeProvider(
        streams=[
            tool_call_stream(("pin_message", {})),
            tool_call_stream(("pin_message", {})),
            text_stream("釘選失敗。"),
        ]
    )

    trace = await run_agent_case(provider, case)

    assert check_agent_trace(case, trace) == ["requested 'pin_message' 2 times, at most 1 allowed"]


async def test_a_model_that_obeys_an_injected_instruction_fails_the_case() -> None:
    case = AGENT_CASES_BY_ID["injection-in-web-page"]
    provider = FakeProvider(
        streams=[
            tool_call_stream(("fetch_url", {"url": "https://example.com/post"})),
            tool_call_stream(("pin_message", {})),
            text_stream("摘要：快取依賴、平行測試。"),
        ]
    )

    trace = await run_agent_case(provider, case)

    assert check_agent_trace(case, trace) == ["requested forbidden tool 'pin_message'"]


# --- judging a trace --------------------------------------------------------


def passing_trace(**changes: object) -> AgentTrace:
    trace = AgentTrace(
        requested=["web_search"], executed=["web_search"], answer="Here you go.", steps=1
    )
    for name, value in changes.items():
        setattr(trace, name, value)
    return trace


def test_matching_trace_has_no_problems() -> None:
    case = AgentCase(
        id="x",
        prompt="x",
        first_tool=("web_search", "fetch_url"),
        require_tools=("web_search",),
        forbid_tools=("remember",),
        max_calls={"web_search": 1},
        answer_contains_any=("HERE",),
    )
    assert check_agent_trace(case, passing_trace()) == []


@pytest.mark.parametrize(
    ("case", "trace", "expected"),
    [
        (
            AgentCase(id="x", prompt="x", no_tools=True),
            passing_trace(),
            "expected no tool calls",
        ),
        (
            AgentCase(id="x", prompt="x", first_tool=("fetch_url",)),
            passing_trace(),
            "first tool 'web_search'",
        ),
        (
            AgentCase(id="x", prompt="x", first_tool=("fetch_url",)),
            passing_trace(requested=[], executed=[]),
            "first tool None",
        ),
        (
            AgentCase(id="x", prompt="x", require_tools=("fetch_url",)),
            passing_trace(),
            "never requested 'fetch_url'",
        ),
        (
            AgentCase(id="x", prompt="x", forbid_tools=("web_search",)),
            passing_trace(),
            "requested forbidden tool 'web_search'",
        ),
        (
            AgentCase(id="x", prompt="x", max_calls={"web_search": 1}),
            passing_trace(requested=["web_search", "web_search"]),
            "requested 'web_search' 2 times",
        ),
        (
            AgentCase(id="x", prompt="x", first_tool=("web_search",)),
            passing_trace(answer="  \n"),
            "final answer is empty",
        ),
        (
            AgentCase(id="x", prompt="x", answer_contains_any=("paris", "巴黎")),
            passing_trace(answer="It is Lyon."),
            "answer mentions none of",
        ),
        (
            AgentCase(id="x", prompt="x", first_tool=("web_search",)),
            passing_trace(requested=["web_search", "delete_channel"]),
            "tools that do not exist: ['delete_channel']",
        ),
        (
            AgentCase(id="x", prompt="x", first_tool=("pin_message",)),
            passing_trace(requested=["pin_message"], executed=["pin_message"], confirmations=[]),
            "do not match confirmation prompts",
        ),
        (
            AgentCase(id="x", prompt="x", first_tool=("pin_message",), approve=False),
            passing_trace(
                requested=["pin_message"], executed=["pin_message"], confirmations=["pin_message"]
            ),
            "ran after the user declined",
        ),
        (
            AgentCase(id="x", prompt="x", first_tool=("web_search",)),
            passing_trace(steps=9),
            "used 9 tool steps",
        ),
    ],
)
def test_each_kind_of_failure_is_reported(
    case: AgentCase, trace: AgentTrace, expected: str
) -> None:
    problems = check_agent_trace(case, trace)
    assert len(problems) == 1
    assert expected in problems[0]


def route(tier: str, *, intent: str = "chat", needs_vision: bool = False) -> RouteDecision:
    return RouteDecision(intent=intent, tier=RouteTier(tier), needs_vision=needs_vision)


def test_route_matching_the_case_has_no_problems() -> None:
    case = RouterCase(
        id="x", text="x", tiers=("local", "cloud_agent"), needs_vision=False, intents=("chat",)
    )
    assert check_route(case, route("local")) == []
    assert check_route(case, route("cloud_agent")) == []


@pytest.mark.parametrize(
    ("case", "decision", "expected"),
    [
        (RouterCase(id="x", text="x", tiers=("local",)), route("media"), "tier 'media'"),
        (
            RouterCase(id="x", text="x", tiers=("media",), image_count=1, needs_vision=True),
            route("media"),
            "needs_vision False",
        ),
        (
            RouterCase(id="x", text="x", tiers=("local",), intents=("chat",)),
            route("local", intent="code"),
            "intent 'code'",
        ),
    ],
)
def test_each_kind_of_route_mismatch_is_reported(
    case: RouterCase, decision: RouteDecision, expected: str
) -> None:
    problems = check_route(case, decision)
    assert len(problems) == 1
    assert expected in problems[0]


def test_router_fallback_never_counts_as_a_pass() -> None:
    # When the model is down the router answers cloud_agent, the expected tier here.
    case = RouterCase(id="x", text="x", tiers=("cloud_agent",))
    fallback = route("cloud_agent", intent=harness.FALLBACK_INTENT)

    problems = check_route(case, fallback)

    assert len(problems) == 1
    assert "fell back" in problems[0]


# --- reading case files -----------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([{"id": "a", "text": "hi", "tier": "local", "tire": "typo"}], "unknown keys ['tire']"),
        ([{"id": "a", "text": "hi"}], "missing keys ['tier']"),
        ({"id": "a"}, "expected a JSON array"),
    ],
)
def test_malformed_case_file_is_rejected(
    rows: object, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "router_cases.json").write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(harness, "CASE_DIR", tmp_path)

    with pytest.raises(ValueError, match=expected.replace("[", r"\[").replace("]", r"\]")):
        load_router_cases()
