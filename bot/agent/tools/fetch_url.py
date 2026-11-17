"""Fetch a public web page and return its readable text.

Only public hosts are allowed. The host is resolved once, every address is
checked, and the connection is pinned to a checked address (the URL is rewritten
to the IP with the original host kept for the Host header and TLS), so a DNS
answer cannot change between the check and the connection. Redirects are
followed by hand so every hop goes through the same check.
"""

import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

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

MAX_BYTES = 2_000_000
MAX_CHARS = 8000
MAX_REDIRECTS = 5
TOTAL_TIMEOUT = 30.0

type Resolver = Callable[[str], Awaitable[list[str]]]

_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "head"})
_BLOCK_TAGS = frozenset(
    {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "pre"}
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_text(html: str) -> tuple[str, str]:
    """Return ``(title, text)`` for an HTML document."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return " ".join(parser.title.split()), parser.text()


async def resolve(host: str) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


def is_public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    # ``is_global`` also rejects shared address space (100.64/10, used by CGNAT
    # and Tailscale); the explicit checks stay for ranges it reports as global.
    return ip.is_global and not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


async def pin_to_public_address(url: str, resolver: Resolver) -> tuple[str, str]:
    """Validate ``url`` and return ``(pinned_url, host)``.

    The pinned URL has the host replaced by a checked IP address, so the
    connection goes exactly where the check looked.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise ToolError(f"invalid URL: {exc}") from exc
    if parts.scheme not in ("http", "https") or not host:
        raise ToolError("only http(s) URLs with a host are allowed")
    if parts.username or parts.password:
        raise ToolError("URLs with credentials are not allowed")
    try:
        addresses = await resolver(host)
    except (socket.gaierror, OSError) as exc:
        raise ToolError(f"cannot resolve host {host}") from exc
    try:
        public = bool(addresses) and all(is_public_address(a) for a in addresses)
    except ValueError as exc:
        raise ToolError(f"resolver returned an invalid address for {host}") from exc
    if not public:
        raise ToolError(f"host {host} is not a public address")
    address = addresses[0]
    literal = f"[{address}]" if ":" in address else address
    netloc = f"{literal}:{port}" if port else literal
    pinned = urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))
    return pinned, host


async def check_public(url: str, resolver: Resolver) -> None:
    """Raise ``ToolError`` unless ``url`` points at a public host."""
    await pin_to_public_address(url, resolver)


def fetch_url_tool(client: httpx.AsyncClient | None = None, resolver: Resolver = resolve) -> Tool:
    # Redirects are followed manually so every hop is checked. No proxies from the
    # environment: they would bypass the address check.
    http = client or httpx.AsyncClient(timeout=15.0, follow_redirects=False, trust_env=False)

    async def fetch(url: str) -> tuple[httpx.Response, str]:
        for _ in range(MAX_REDIRECTS + 1):
            pinned, host = await pin_to_public_address(url, resolver)
            headers = {"Host": host, "User-Agent": "Mozilla/5.0 (compatible; discord-bot)"}
            try:
                async with http.stream(
                    "GET", pinned, headers=headers, extensions={"sni_hostname": host}
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise ToolError("redirect without a location")
                        url = urljoin(url, location)
                        continue
                    response.raise_for_status()
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_BYTES:
                            break
                    response._content = bytes(body)
                    return response, url
            except httpx.HTTPStatusError as exc:
                raise ToolError(f"HTTP {exc.response.status_code} for {url}") from exc
            except httpx.HTTPError as exc:
                raise ToolError(f"request failed: {exc}") from exc
        raise ToolError("too many redirects")

    async def run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        url = require_str(args, "url", max_len=2000)
        offset = optional_int(args, "offset", 0, lo=0, hi=10_000_000)
        try:
            async with asyncio.timeout(TOTAL_TIMEOUT):
                response, final_url = await fetch(url)
        except TimeoutError as exc:
            raise ToolError(f"took longer than {TOTAL_TIMEOUT:.0f} seconds") from exc
        content_type = response.headers.get("content-type", "")
        if "html" in content_type:
            title, text = html_to_text(response.text)
        elif content_type.startswith("text/") or "json" in content_type or "xml" in content_type:
            title, text = "", response.text
        else:
            raise ToolError(f"unsupported content type: {content_type or 'unknown'}")
        window = text[offset : offset + MAX_CHARS]
        remaining = max(0, len(text) - offset - MAX_CHARS)
        header = f"URL: {final_url}\nTitle: {title}\n" if title else f"URL: {final_url}\n"
        footer = (
            f"\n[{remaining} more characters; call again with offset={offset + MAX_CHARS}]"
            if remaining
            else ""
        )
        return ToolResult(f"{header}\n{window}{footer}", sources=(final_url,))

    return Tool(
        name="fetch_url",
        description="Read a public web page as plain text. Long pages are paged with offset.",
        tier=Tier.READ_ONLY,
        parameters=object_schema(
            {
                "url": {"type": "string", "description": "http(s) URL to read"},
                "offset": {"type": "integer", "description": "Character offset for long pages"},
            },
            required=["url"],
        ),
        run=run,
    )
