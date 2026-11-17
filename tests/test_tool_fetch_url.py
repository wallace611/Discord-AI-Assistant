"""fetch_url: public-address checks, redirects, text extraction and paging."""

import asyncio
import ipaddress
import socket
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from bot.agent.tools import fetch_url as fetch_url_module
from bot.agent.tools.base import Tier, Tool, ToolContext, ToolError
from bot.agent.tools.fetch_url import (
    MAX_BYTES,
    MAX_CHARS,
    MAX_REDIRECTS,
    check_public,
    fetch_url_tool,
    html_to_text,
    is_public_address,
)

CTX = ToolContext(guild_id=1, channel_id=2, user_id=3, message_id=4)
PUBLIC_IP = "93.184.216.34"

type Handler = Callable[[httpx.Request], httpx.Response]


class FakeDNS:
    """Resolver double: literal IPs resolve to themselves, unknown hosts are public."""

    def __init__(self, hosts: dict[str, list[str]] | None = None) -> None:
        self.hosts = hosts or {}
        self.lookups: list[str] = []

    async def __call__(self, host: str) -> list[str]:
        self.lookups.append(host)
        if host in self.hosts:
            return self.hosts[host]
        try:
            ipaddress.ip_address(host)
        except ValueError:
            return [PUBLIC_IP]
        return [host]


def make_tool(
    handler: Handler, hosts: dict[str, list[str]] | None = None
) -> tuple[Tool, list[httpx.Request], FakeDNS]:
    """The tool wired to a mock transport; also returns the requests it sent."""
    requests: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    dns = FakeDNS(hosts)
    client = httpx.AsyncClient(transport=httpx.MockTransport(recording))
    return fetch_url_tool(client=client, resolver=dns), requests, dns


def text_page(body: str, content_type: str = "text/plain; charset=utf-8") -> Handler:
    return lambda request: httpx.Response(
        200, headers={"content-type": content_type}, content=body.encode("utf-8")
    )


# --- address classification -------------------------------------------------


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.5.4",
        "192.168.1.1",
        "169.254.169.254",  # cloud metadata endpoint
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
        "::ffff:10.0.0.1",
        "100.64.0.1",  # shared address space: CGNAT, Tailscale
        "100.100.100.100",
    ],
)
def test_internal_addresses_are_not_public(address: str) -> None:
    assert is_public_address(address) is False


@pytest.mark.parametrize("address", ["8.8.8.8", PUBLIC_IP, "2606:4700::1111"])
def test_internet_addresses_are_public(address: str) -> None:
    assert is_public_address(address) is True


# --- refusing to connect ----------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/api/tags",
        "http://localhost.example/",
        "http://[::1]:8080/",
        "http://169.254.169.254/latest/meta-data/",
    ],
)
async def test_private_address_is_refused_before_any_request(url: str) -> None:
    tool, requests, _ = make_tool(text_page("secret"), {"localhost.example": ["192.168.1.10"]})

    with pytest.raises(ToolError, match="not a public address"):
        await tool.run({"url": url}, CTX)

    assert requests == []


async def test_host_with_any_private_address_is_refused() -> None:
    tool, requests, _ = make_tool(text_page("secret"), {"mixed.example": [PUBLIC_IP, "10.0.0.5"]})

    with pytest.raises(ToolError, match="not a public address"):
        await tool.run({"url": "https://mixed.example/"}, CTX)

    assert requests == []


async def test_host_without_addresses_is_refused() -> None:
    tool, requests, _ = make_tool(text_page("x"), {"empty.example": []})

    with pytest.raises(ToolError, match="not a public address"):
        await tool.run({"url": "https://empty.example/"}, CTX)

    assert requests == []


async def test_unresolvable_host_is_a_tool_error() -> None:
    async def failing(host: str) -> list[str]:
        raise socket.gaierror("no such host")

    with pytest.raises(ToolError, match="cannot resolve"):
        await check_public("https://nowhere.example/", failing)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/file.txt",
        "gopher://example.com/",
        "javascript:alert(1)",
        "example.com/no-scheme",
        "http:///no-host",
    ],
)
async def test_non_http_urls_are_refused(url: str) -> None:
    tool, requests, dns = make_tool(text_page("x"))

    with pytest.raises(ToolError, match="only http"):
        await tool.run({"url": url}, CTX)

    assert requests == []
    assert dns.lookups == []


async def test_urls_with_credentials_are_refused() -> None:
    tool, requests, _ = make_tool(text_page("x"))

    with pytest.raises(ToolError, match="credentials"):
        await tool.run({"url": "https://user:pass@example.com/"}, CTX)

    assert requests == []


# --- redirects --------------------------------------------------------------


async def test_redirect_to_private_address_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["host"] == "public.example":
            return httpx.Response(302, headers={"location": "http://internal.example/admin"})
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"secret")

    tool, requests, dns = make_tool(handler, {"internal.example": ["192.168.1.10"]})

    with pytest.raises(ToolError, match="internal.example is not a public address"):
        await tool.run({"url": "https://public.example/start"}, CTX)

    # Only the first hop was requested; the private target was never contacted.
    assert [request.headers["host"] for request in requests] == ["public.example"]
    assert dns.lookups == ["public.example", "internal.example"]


async def test_redirect_to_literal_loopback_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(307, headers={"location": "http://127.0.0.1:11434/api/tags"})

    tool, requests, _ = make_tool(handler)

    with pytest.raises(ToolError, match="not a public address"):
        await tool.run({"url": "https://public.example/"}, CTX)

    assert len(requests) == 1


async def test_public_redirects_are_followed_and_every_hop_is_checked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(301, headers={"location": "https://b.example/next"})
        if request.url.path == "/next":
            return httpx.Response(302, headers={"location": "/final"})  # relative
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"arrived")

    tool, requests, dns = make_tool(handler)

    result = await tool.run({"url": "https://a.example/start"}, CTX)

    assert "arrived" in result.content
    assert [(request.headers["host"], request.url.path) for request in requests] == [
        ("a.example", "/start"),
        ("b.example", "/next"),
        ("b.example", "/final"),
    ]
    assert dns.lookups == ["a.example", "b.example", "b.example"]
    # The source is the page that was actually read, not the URL that was asked for.
    assert result.sources == ("https://b.example/final",)
    assert "URL: https://b.example/final" in result.content


async def test_redirect_loop_stops_after_the_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://loop.example/again"})

    tool, requests, _ = make_tool(handler)

    with pytest.raises(ToolError, match="too many redirects"):
        await tool.run({"url": "https://loop.example/"}, CTX)

    assert len(requests) == MAX_REDIRECTS + 1


async def test_redirect_without_location_is_a_tool_error() -> None:
    tool, _, _ = make_tool(lambda request: httpx.Response(302))

    with pytest.raises(ToolError, match="redirect without a location"):
        await tool.run({"url": "https://example.com/"}, CTX)


# --- connecting to the address that was checked ------------------------------


async def test_connection_goes_to_the_checked_address_under_the_original_name() -> None:
    tool, requests, _ = make_tool(text_page("hello"), {"site.example": ["8.8.4.4"]})

    result = await tool.run({"url": "https://site.example:8443/a/b?x=1&y=2#frag"}, CTX)

    [request] = requests
    assert request.url.host == "8.8.4.4"
    assert request.url.port == 8443
    assert request.url.path == "/a/b"
    assert request.url.query == b"x=1&y=2"
    assert request.headers["host"].split(":")[0] == "site.example"
    # TLS must present and verify the original name, not the IP address.
    assert request.extensions["sni_hostname"] == "site.example"
    # The user-facing URL is the one that was asked for, never the pinned one.
    assert result.sources == ("https://site.example:8443/a/b?x=1&y=2#frag",)
    assert "8.8.4.4" not in result.content


async def test_ipv6_address_is_bracketed_in_the_pinned_url() -> None:
    tool, requests, _ = make_tool(text_page("hello"), {"v6.example": ["2606:4700::1111"]})

    await tool.run({"url": "http://v6.example/"}, CTX)

    assert requests[0].url.host == "2606:4700::1111"
    assert requests[0].headers["host"] == "v6.example"


async def test_dns_rebinding_cannot_move_the_connection() -> None:
    # The name answers with a public address once and with loopback afterwards.
    answers = [[PUBLIC_IP], ["127.0.0.1"], ["127.0.0.1"]]
    lookups: list[str] = []

    async def rebinding(host: str) -> list[str]:
        lookups.append(host)
        return answers[len(lookups) - 1]

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"public page")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = fetch_url_tool(client=client, resolver=rebinding)

    result = await tool.run({"url": "http://rebind.example/"}, CTX)

    # One lookup per hop, and the request used exactly the address that was checked.
    assert lookups == ["rebind.example"]
    assert [request.url.host for request in requests] == [PUBLIC_IP]
    assert "public page" in result.content

    # The next fetch sees the loopback answer and is refused before connecting.
    with pytest.raises(ToolError, match="not a public address"):
        await tool.run({"url": "http://rebind.example/"}, CTX)
    assert len(requests) == 1


@pytest.mark.parametrize("url", ["http://[::1/", "http://example.com:notaport/"])
async def test_malformed_url_is_a_tool_error(url: str) -> None:
    tool, requests, _ = make_tool(text_page("x"))

    with pytest.raises(ToolError, match="invalid URL"):
        await tool.run({"url": url}, CTX)

    assert requests == []


async def test_resolver_returning_garbage_is_a_tool_error() -> None:
    tool, requests, _ = make_tool(text_page("x"), {"odd.example": ["not-an-ip"]})

    with pytest.raises(ToolError, match="invalid address"):
        await tool.run({"url": "http://odd.example/"}, CTX)

    assert requests == []


async def test_slow_server_is_cut_off_by_the_total_time_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch_url_module, "TOTAL_TIMEOUT", 0.05)

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"late")

    client = httpx.AsyncClient(transport=httpx.MockTransport(slow))
    tool = fetch_url_tool(client=client, resolver=FakeDNS())

    with pytest.raises(ToolError, match="took longer than"):
        await tool.run({"url": "https://slow.example/"}, CTX)


# --- reading the page -------------------------------------------------------


def test_html_to_text_keeps_content_and_drops_markup() -> None:
    html = """
    <html><head><title>  My   Page </title><style>body { color: red }</style></head>
    <body>
      <script>alert("ignore me")</script>
      <h1>Heading</h1>
      <p>First   paragraph with <b>bold</b> &amp; an entity.</p>
      <ul><li>one</li><li>two</li></ul>
      <noscript>enable javascript</noscript>
    </body></html>
    """

    title, text = html_to_text(html)

    assert title == "My Page"
    assert text.splitlines()[0] == "Heading"
    assert "First paragraph with bold & an entity." in text
    assert "one" in text and "two" in text
    assert text.index("one") < text.index("two")
    for dropped in ("alert", "color: red", "enable javascript", "My Page", "<"):
        assert dropped not in text
    assert "\n\n\n" not in text


async def test_html_page_is_returned_as_text_with_title_and_source() -> None:
    page = "<html><head><title>Docs</title></head><body><p>中文內容 and English.</p></body></html>"
    tool, _, _ = make_tool(text_page(page, "text/html; charset=utf-8"))

    result = await tool.run({"url": "https://example.com/docs"}, CTX)

    assert result.content.startswith("URL: https://example.com/docs\nTitle: Docs\n")
    assert "中文內容 and English." in result.content
    assert "<p>" not in result.content
    assert result.sources == ("https://example.com/docs",)


@pytest.mark.parametrize(
    "content_type",
    ["text/plain", "application/json", "application/xml", "text/markdown; charset=utf-8"],
)
async def test_plain_text_types_pass_through_unchanged(content_type: str) -> None:
    tool, _, _ = make_tool(text_page('{"key": "<b>value</b>"}', content_type))

    result = await tool.run({"url": "https://example.com/data"}, CTX)

    assert '{"key": "<b>value</b>"}' in result.content
    assert "Title:" not in result.content


@pytest.mark.parametrize("content_type", ["image/png", "application/pdf", "application/zip", ""])
async def test_binary_content_types_are_refused(content_type: str) -> None:
    headers = {"content-type": content_type} if content_type else {}
    tool, _, _ = make_tool(lambda request: httpx.Response(200, headers=headers, content=b"\x89PNG"))

    with pytest.raises(ToolError, match="unsupported content type"):
        await tool.run({"url": "https://example.com/file"}, CTX)


@pytest.mark.parametrize("status", [403, 404, 500])
async def test_http_error_status_is_a_tool_error(status: int) -> None:
    tool, _, _ = make_tool(lambda request: httpx.Response(status, content=b"nope"))

    with pytest.raises(ToolError, match=f"HTTP {status}"):
        await tool.run({"url": "https://example.com/missing"}, CTX)


async def test_network_failure_is_a_tool_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    tool, _, _ = make_tool(handler)

    with pytest.raises(ToolError, match="request failed"):
        await tool.run({"url": "https://example.com/"}, CTX)


async def test_download_stops_once_the_size_limit_is_passed() -> None:
    chunk = b"a" * 100_000
    produced = 0

    async def endless() -> AsyncIterator[bytes]:
        nonlocal produced
        for _ in range(200):  # 20 MB if read to the end
            produced += 1
            yield chunk

    tool, _, _ = make_tool(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/plain"}, content=endless()
        )
    )

    result = await tool.run({"url": "https://example.com/huge"}, CTX)

    assert produced * len(chunk) <= MAX_BYTES + 2 * len(chunk)
    assert "more characters" in result.content


# --- paging -----------------------------------------------------------------


async def test_long_pages_are_read_in_windows_with_offset() -> None:
    body = "".join(f"{index:07d}\n" for index in range(MAX_CHARS // 8 * 2 + 50))  # 8 chars per line
    tool, _, _ = make_tool(text_page(body))
    url = "https://example.com/long"

    pieces: list[str] = []
    offset = 0
    for _ in range(5):
        result = await tool.run({"url": url, "offset": offset}, CTX)
        header, _, rest = result.content.partition("\n\n")
        assert header == f"URL: {url}"
        window, marker, footer = rest.partition("\n[")
        pieces.append(window)
        if not marker:
            break
        offset += MAX_CHARS
        remaining = len(body) - offset
        assert footer == f"{remaining} more characters; call again with offset={offset}]"
        assert len(window) == MAX_CHARS
    else:
        pytest.fail("paging never reached the end of the page")

    assert len(pieces) == 3
    assert "".join(pieces) == body


async def test_short_page_has_no_paging_footer() -> None:
    tool, _, _ = make_tool(text_page("short"))

    result = await tool.run({"url": "https://example.com/"}, CTX)

    assert result.content == "URL: https://example.com/\n\nshort"


async def test_offset_past_the_end_returns_an_empty_window() -> None:
    tool, _, _ = make_tool(text_page("short"))

    result = await tool.run({"url": "https://example.com/", "offset": 1000}, CTX)

    assert result.content == "URL: https://example.com/\n\n"


# --- arguments and metadata -------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"url": ""},
        {"url": "   "},
        {"url": 42},
        {"url": "https://example.com/", "offset": "10"},
        {"url": "https://example.com/", "offset": True},
        {"url": "https://example.com/", "offset": 1.5},
    ],
)
async def test_invalid_arguments_are_tool_errors(args: dict[str, object]) -> None:
    tool, requests, _ = make_tool(text_page("x"))

    with pytest.raises(ToolError):
        await tool.run(args, CTX)

    assert requests == []


async def test_negative_offset_is_clamped_to_the_start() -> None:
    tool, _, _ = make_tool(text_page("short"))

    result = await tool.run({"url": "https://example.com/", "offset": -5}, CTX)

    assert result.content.endswith("short")


def test_tool_is_read_only_and_describes_its_arguments() -> None:
    tool, _, _ = make_tool(text_page("x"))

    assert tool.name == "fetch_url"
    assert tool.tier is Tier.READ_ONLY
    function = tool.spec()["function"]
    assert function["name"] == "fetch_url"
    assert function["parameters"]["required"] == ["url"]
    assert set(function["parameters"]["properties"]) == {"url", "offset"}
