from bot.gateway.streaming import EMPTY_REPLY, StreamingReply, _split
from fakes import FakeChannel


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_reply(limit: int = 2000) -> tuple[StreamingReply, FakeChannel, FakeClock]:
    channel = FakeChannel()
    clock = FakeClock()
    reply = StreamingReply(channel.send, edit_interval=1.0, limit=limit, clock=clock)
    return reply, channel, clock


async def test_first_delta_is_sent_immediately() -> None:
    reply, channel, _ = make_reply()
    await reply.feed("Hello")
    assert channel.texts == ["Hello"]


async def test_edits_are_throttled_to_the_interval() -> None:
    reply, channel, clock = make_reply()

    await reply.feed("a")
    clock.now = 0.3
    await reply.feed("b")
    clock.now = 0.6
    await reply.feed("c")
    assert channel.messages[0].edits == []

    clock.now = 1.0
    await reply.feed("d")
    assert channel.messages[0].edits == ["abcd"]

    clock.now = 1.5
    await reply.feed("e")
    assert channel.messages[0].edits == ["abcd"]

    await reply.finish()
    assert channel.messages[0].edits == ["abcd", "abcde"]
    assert len(channel.messages) == 1


async def test_finish_does_not_edit_when_nothing_changed() -> None:
    reply, channel, _ = make_reply()
    await reply.feed("done")
    await reply.finish()
    assert channel.texts == ["done"]
    assert channel.messages[0].edits == []


async def test_blank_output_sends_placeholder() -> None:
    reply, channel, _ = make_reply()
    await reply.feed("   \n")
    await reply.finish()
    assert channel.texts == [EMPTY_REPLY]


async def test_leading_whitespace_is_held_until_real_text_arrives() -> None:
    reply, channel, clock = make_reply()
    await reply.feed("\n")
    assert channel.messages == []
    clock.now = 2.0
    await reply.feed("text")
    assert channel.texts == ["\ntext"]


async def test_long_output_continues_in_new_messages() -> None:
    reply, channel, clock = make_reply(limit=20)

    await reply.feed("first line\n")
    clock.now = 5.0
    await reply.feed("second line that is long\nthird")
    await reply.finish()

    assert all(len(text) <= 20 for text in channel.texts)
    assert channel.texts == ["first line", "second line that is", "long\nthird"]


async def test_overflow_without_separators_is_cut_at_the_limit() -> None:
    reply, channel, _ = make_reply(limit=10)
    await reply.feed("x" * 25)
    await reply.finish()
    assert channel.texts == ["x" * 10, "x" * 10, "x" * 5]


def test_split_prefers_newline_then_space() -> None:
    assert _split("aaa bbb\nccc ddd", 12) == ("aaa bbb", "ccc ddd")
    assert _split("aaa bbb ccc", 9) == ("aaa bbb", "ccc")
    assert _split("abcdefgh", 4) == ("abcd", "efgh")
