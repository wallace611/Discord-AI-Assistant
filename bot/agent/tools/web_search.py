"""Web search through a self-hosted SearXNG instance.

SearXNG must have the JSON format enabled (``search.formats: [html, json]``).
"""

from typing import Any

import httpx

from bot.agent.tools.base import (
    Tier,
    Tool,
    ToolContext,
    ToolError,
    ToolResult,
    object_schema,
    optional_int,
    require_str,
)

MAX_RESULTS = 8


def format_results(results: list[dict[str, Any]], limit: int) -> ToolResult:
    lines: list[str] = []
    sources: list[str] = []
    for index, item in enumerate(results[:limit], start=1):
        url = str(item.get("url") or "")
        title = str(item.get("title") or url or "(untitled)").strip()
        snippet = " ".join(str(item.get("content") or "").split())
        if not url:
            continue
        sources.append(url)
        lines.append(f"[{index}] {title}\n{url}\n{snippet}".rstrip())
    if not lines:
        return ToolResult("No results.")
    return ToolResult("\n\n".join(lines), sources=tuple(sources))


def web_search_tool(searxng_url: str, client: httpx.AsyncClient | None = None) -> Tool:
    http = client or httpx.AsyncClient(timeout=15.0)

    async def run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = require_str(args, "query", max_len=400)
        limit = optional_int(args, "max_results", 5, lo=1, hi=MAX_RESULTS)
        try:
            response = await http.get(
                f"{searxng_url.rstrip('/')}/search",
                params={"q": query, "format": "json", "safesearch": 1},
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise ToolError(f"search backend unavailable: {exc}") from exc
        except ValueError as exc:
            raise ToolError("search backend returned invalid JSON") from exc
        results = data.get("results") if isinstance(data, dict) else None
        return format_results(results if isinstance(results, list) else [], limit)

    return Tool(
        name="web_search",
        description=(
            "Search the web. Returns titles, URLs and snippets. Use fetch_url to read a result."
        ),
        tier=Tier.READ_ONLY,
        parameters=object_schema(
            {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "1-8, default 5"},
            },
            required=["query"],
        ),
        run=run,
    )
