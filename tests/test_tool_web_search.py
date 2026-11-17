"""web_search: the SearXNG request, result formatting and backend failures."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from bot.agent.tools.base import Tier, Tool, ToolContext, ToolError
from bot.agent.tools.web_search import MAX_RESULTS, format_results, web_search_tool

CTX = ToolContext(guild_id=1, channel_id=2, user_id=3, message_id=4)

type Handler = Callable[[httpx.Request], httpx.Response]


def make_tool(
    handler: Handler, base_url: str = "http://searxng.test:8080"
) -> tuple[Tool, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(recording))
    return web_search_tool(base_url, client=client), requests


def results_page(results: Any) -> Handler:
    return lambda request: httpx.Response(200, json={"query": "q", "results": results})


def hit(index: int) -> dict[str, str]:
    return {
        "url": f"https://site{index}.example/page",
        "title": f"Title {index}",
        "content": f"Snippet {index}",
    }


# --- the request ------------------------------------------------------------


async def test_query_is_sent_to_the_searxng_json_api() -> None:
    tool, requests = make_tool(results_page([hit(1)]))

    await tool.run({"query": "  台北 天氣 today  "}, CTX)

    [request] = requests
    assert request.method == "GET"
    assert request.url.host == "searxng.test"
    assert request.url.path == "/search"
    assert request.url.params["q"] == "台北 天氣 today"
    assert request.url.params["format"] == "json"
    assert request.headers["accept"] == "application/json"


@pytest.mark.parametrize("base_url", ["http://searxng.test:8080/", "http://searxng.test:8080"])
async def test_trailing_slash_in_base_url_does_not_double_up(base_url: str) -> None:
    tool, requests = make_tool(results_page([]), base_url)

    await tool.run({"query": "anything"}, CTX)

    assert str(requests[0].url).startswith("http://searxng.test:8080/search?")


# --- formatting -------------------------------------------------------------


async def test_results_are_numbered_with_title_url_and_snippet() -> None:
    tool, _ = make_tool(
        results_page(
            [
                {
                    "url": "https://docs.example/a",
                    "title": "  First result ",
                    "content": "Line one\n   line   two",
                },
                {"url": "https://docs.example/b", "title": "Second", "content": ""},
            ]
        )
    )

    result = await tool.run({"query": "docs"}, CTX)

    assert result.content == (
        "[1] First result\nhttps://docs.example/a\nLine one line two\n\n"
        "[2] Second\nhttps://docs.example/b"
    )
    assert result.sources == ("https://docs.example/a", "https://docs.example/b")


async def test_default_limit_is_five_results() -> None:
    tool, _ = make_tool(results_page([hit(index) for index in range(1, 21)]))

    result = await tool.run({"query": "many"}, CTX)

    assert len(result.sources) == 5
    assert result.sources[0] == "https://site1.example/page"
    assert "[5] Title 5" in result.content
    assert "Title 6" not in result.content


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(1, 1), (3, 3), (MAX_RESULTS, MAX_RESULTS), (50, MAX_RESULTS), (0, 1)],
)
async def test_max_results_is_clamped(requested: int, expected: int) -> None:
    tool, _ = make_tool(results_page([hit(index) for index in range(1, 21)]))

    result = await tool.run({"query": "many", "max_results": requested}, CTX)

    assert len(result.sources) == expected


def test_results_without_url_are_skipped() -> None:
    result = format_results([{"title": "no link", "content": "x"}, hit(2)], limit=5)

    assert result.sources == ("https://site2.example/page",)
    assert "no link" not in result.content


def test_title_falls_back_to_the_url() -> None:
    result = format_results([{"url": "https://bare.example/"}], limit=5)

    assert result.content == "[1] https://bare.example/\nhttps://bare.example/"


@pytest.mark.parametrize(
    "payload",
    [
        {"results": []},
        {"results": None},
        {"results": "not a list"},
        {"unexpected": True},
        ["a", "list"],
        {"results": [{"title": "only a title"}]},
    ],
)
async def test_nothing_usable_reports_no_results(payload: Any) -> None:
    tool, _ = make_tool(lambda request: httpx.Response(200, json=payload))

    result = await tool.run({"query": "obscure"}, CTX)

    assert result.content == "No results."
    assert result.sources == ()


# --- backend failures -------------------------------------------------------


async def test_unreachable_backend_is_a_tool_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    tool, _ = make_tool(handler)

    with pytest.raises(ToolError, match="search backend unavailable"):
        await tool.run({"query": "anything"}, CTX)


async def test_backend_timeout_is_a_tool_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    tool, _ = make_tool(handler)

    with pytest.raises(ToolError, match="search backend unavailable"):
        await tool.run({"query": "anything"}, CTX)


@pytest.mark.parametrize("status", [403, 429, 500, 502])
async def test_backend_error_status_is_a_tool_error(status: int) -> None:
    # 403 is what SearXNG answers when the JSON format is not enabled.
    tool, _ = make_tool(lambda request: httpx.Response(status, text="error"))

    with pytest.raises(ToolError, match="search backend unavailable"):
        await tool.run({"query": "anything"}, CTX)


async def test_non_json_body_is_a_tool_error() -> None:
    tool, _ = make_tool(lambda request: httpx.Response(200, text="<html>not json</html>"))

    with pytest.raises(ToolError, match="invalid JSON"):
        await tool.run({"query": "anything"}, CTX)


# --- arguments and metadata -------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": ["a"]},
        {"query": "ok", "max_results": "3"},
        {"query": "ok", "max_results": True},
    ],
)
async def test_invalid_arguments_are_tool_errors(args: dict[str, object]) -> None:
    tool, requests = make_tool(results_page([hit(1)]))

    with pytest.raises(ToolError):
        await tool.run(args, CTX)

    assert requests == []


async def test_very_long_query_is_truncated() -> None:
    tool, requests = make_tool(results_page([]))

    await tool.run({"query": "x" * 5000}, CTX)

    assert len(requests[0].url.params["q"]) == 400


def test_tool_is_read_only_and_describes_its_arguments() -> None:
    tool, _ = make_tool(results_page([]))

    assert tool.name == "web_search"
    assert tool.tier is Tier.READ_ONLY
    function = tool.spec()["function"]
    assert function["parameters"]["required"] == ["query"]
    assert set(function["parameters"]["properties"]) == {"query", "max_results"}
    # The spec must be JSON-serialisable to be sent to the model.
    assert json.loads(json.dumps(tool.spec()))["function"]["name"] == "web_search"
