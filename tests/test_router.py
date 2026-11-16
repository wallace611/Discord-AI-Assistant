import logging
from datetime import date

import pytest

from bot.providers.base import ProviderError
from bot.router import DailyTokenBudget, RouteParseError, Router, Tier, parse_route
from bot.router.router import ROUTE_RESPONSE_FORMAT
from fakes import FakeProvider


def test_parse_route_plain_json() -> None:
    decision = parse_route('{"intent": "chat", "tier": "local", "needs_vision": false}')
    assert decision.intent == "chat"
    assert decision.tier is Tier.LOCAL
    assert decision.needs_vision is False
    assert decision.downgraded is False


def test_parse_route_tolerates_think_block_and_code_fence() -> None:
    raw = (
        '<think>the user wants {"something"} complicated</think>\n'
        '```json\n{"intent": "code", "tier": "cloud_agent", "needs_vision": false}\n```'
    )
    assert parse_route(raw).tier is Tier.CLOUD_AGENT


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "I think this is a chat message.",
        '{"intent": "chat", "tier": "local"',
        '{"intent": "chat", "tier": "galaxy", "needs_vision": false}',
        '{"intent": "chat", "tier": "local"}',
        '{"intent": "", "tier": "local", "needs_vision": false}',
        '{"tier": "local", "needs_vision": false}',
    ],
)
def test_parse_route_rejects_invalid_output(raw: str) -> None:
    with pytest.raises(RouteParseError):
        parse_route(raw)


def test_parse_route_ignores_flags_the_model_must_not_set() -> None:
    raw = '{"intent": "chat", "tier": "local", "needs_vision": false, "downgraded": true}'
    assert parse_route(raw).downgraded is False


async def test_route_calls_local_model_with_schema() -> None:
    provider = FakeProvider(reply='{"intent": "chat", "tier": "local", "needs_vision": false}')
    router = Router(provider, model="router-model")

    decision = await router.route("hello there")

    assert decision.tier is Tier.LOCAL
    call = provider.complete_calls[0]
    assert call["model"] == "router-model"
    assert call["temperature"] == 0
    assert call["response_format"] == ROUTE_RESPONSE_FORMAT
    assert call["messages"][0]["role"] == "system"
    assert "hello there" in call["messages"][1]["content"]


async def test_route_truncates_long_input() -> None:
    provider = FakeProvider(reply='{"intent": "chat", "tier": "local", "needs_vision": false}')
    router = Router(provider, max_input_chars=10)

    await router.route("a" * 50)

    assert "a" * 10 in provider.complete_calls[0]["messages"][1]["content"]
    assert "a" * 11 not in provider.complete_calls[0]["messages"][1]["content"]


async def test_unparseable_output_defaults_to_cloud_agent_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    router = Router(FakeProvider(reply="sure, happy to help!"))

    with caplog.at_level(logging.WARNING, logger="bot.router.router"):
        decision = await router.route("explain quantum computing", image_count=1)

    assert decision.tier is Tier.CLOUD_AGENT
    assert decision.needs_vision is True
    assert any("unparseable" in record.getMessage() for record in caplog.records)


async def test_provider_failure_defaults_to_cloud_agent() -> None:
    router = Router(FakeProvider(error=ProviderError("ollama down")))
    decision = await router.route("hi")
    assert decision.tier is Tier.CLOUD_AGENT
    assert decision.downgraded is False


async def test_needs_vision_is_cleared_without_images() -> None:
    provider = FakeProvider(reply='{"intent": "chat", "tier": "local", "needs_vision": true}')
    assert (await Router(provider).route("hi")).needs_vision is False
    assert (await Router(provider).route("hi", image_count=2)).needs_vision is True


@pytest.mark.parametrize("tier", ["cloud_agent", "media"])
async def test_exhausted_budget_forces_local(tier: str) -> None:
    provider = FakeProvider(reply=f'{{"intent": "code", "tier": "{tier}", "needs_vision": false}}')

    decision = await Router(provider).route("write a parser", cloud_allowed=False)

    assert decision.tier is Tier.LOCAL
    assert decision.downgraded is True


async def test_exhausted_budget_does_not_flag_already_local_routes() -> None:
    provider = FakeProvider(reply='{"intent": "chat", "tier": "local", "needs_vision": false}')
    decision = await Router(provider).route("hi", cloud_allowed=False)
    assert decision.tier is Tier.LOCAL
    assert decision.downgraded is False


async def test_exhausted_budget_also_overrides_parse_fallback() -> None:
    decision = await Router(FakeProvider(reply="garbage")).route("hi", cloud_allowed=False)
    assert decision.tier is Tier.LOCAL
    assert decision.downgraded is True


def test_budget_unlimited_when_limit_is_none() -> None:
    budget = DailyTokenBudget(None)
    budget.record(1, 10**9)
    assert budget.is_exhausted(1) is False


def test_budget_tracks_users_independently() -> None:
    budget = DailyTokenBudget(100)
    budget.record(1, 60)
    assert budget.is_exhausted(1) is False
    budget.record(1, 40)
    assert budget.is_exhausted(1) is True
    assert budget.is_exhausted(2) is False
    assert budget.used(1) == 100


def test_budget_zero_limit_disables_cloud() -> None:
    assert DailyTokenBudget(0).is_exhausted(1) is True


def test_budget_resets_on_new_day() -> None:
    today = date(2026, 1, 1)
    budget = DailyTokenBudget(100, today=lambda: today)
    budget.record(1, 150)
    assert budget.is_exhausted(1) is True

    today = date(2026, 1, 2)
    assert budget.is_exhausted(1) is False
    assert budget.used(1) == 0
