"""Live regression run against the real models. Skipped unless ``REGRESSION=1``.

    REGRESSION=1 pytest tests/regression/test_regression_live.py

Reads ``.env``: the router cases go to the local Ollama model, the agent cases to
the configured cloud model. Tools are stubs, so nothing is searched, fetched,
executed or posted, but the cloud model is called and tokens are spent.
"""

from collections.abc import AsyncIterator

import pytest

from bot.config import Settings
from bot.providers import ChatProvider, build_cloud_provider, build_local_provider
from bot.router import Router
from regression.harness import (
    LIVE,
    LIVE_SKIP_REASON,
    AgentCase,
    RouterCase,
    check_agent_trace,
    check_route,
    load_agent_cases,
    load_router_cases,
    run_agent_case,
)

pytestmark = pytest.mark.skipif(not LIVE, reason=LIVE_SKIP_REASON)


@pytest.fixture(scope="module")
def settings() -> Settings:
    # Module scope, so this is built before the per-test fixture that hides the
    # developer's environment variables.
    return Settings()  # type: ignore[call-arg]  # values come from .env


@pytest.fixture
async def local(settings: Settings) -> AsyncIterator[ChatProvider]:
    provider = build_local_provider(settings)
    yield provider
    await provider.aclose()


@pytest.fixture
async def cloud(settings: Settings) -> AsyncIterator[ChatProvider]:
    provider = build_cloud_provider(settings)
    yield provider
    await provider.aclose()


@pytest.mark.parametrize("case", load_router_cases(), ids=lambda case: case.id)
async def test_router_case(case: RouterCase, local: ChatProvider, settings: Settings) -> None:
    router = Router(local, model=settings.local_router_model)

    decision = await router.route(case.text, image_count=case.image_count)

    problems = check_route(case, decision)
    assert not problems, f"{'; '.join(problems)} (decision: {decision!r})"


@pytest.mark.parametrize("case", load_agent_cases(), ids=lambda case: case.id)
async def test_agent_case(case: AgentCase, cloud: ChatProvider) -> None:
    trace = await run_agent_case(cloud, case)

    problems = check_agent_trace(case, trace)
    assert not problems, (
        f"{'; '.join(problems)} (requested: {trace.requested}, answer: {trace.answer[:300]!r})"
    )
