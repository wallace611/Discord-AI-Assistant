import json

from bot.memory import InMemoryStore
from bot.memory.user_facts import FACTS_RESPONSE_FORMAT, UserFacts
from bot.providers.base import ProviderError
from fakes import FakeProvider, stored


def facts_json(*facts: object) -> str:
    return json.dumps({"facts": list(facts)})


MESSAGE = stored(1, "I mostly write Rust and I work as a backend engineer.")


async def test_extract_stores_new_facts_for_the_author_in_scope() -> None:
    store = InMemoryStore()
    provider = FakeProvider(reply=facts_json("Alice writes Rust.", "Alice is a backend engineer."))
    facts = UserFacts(store, provider, model="m")

    added = await facts.extract(MESSAGE)

    assert added == ["Alice writes Rust.", "Alice is a backend engineer."]
    assert [fact.fact for fact in await facts.view(7, 200)] == added
    assert await facts.view(7, 201) == []
    assert await facts.view(8, 200) == []
    call = provider.complete_calls[0]
    assert call["model"] == "m"
    assert call["response_format"] == FACTS_RESPONSE_FORMAT
    assert MESSAGE.content in call["messages"][1]["content"]


async def test_known_facts_are_shown_to_the_model_and_not_duplicated() -> None:
    store = InMemoryStore()
    await store.add_fact(7, 200, "Alice writes Rust.", None)
    provider = FakeProvider(reply=facts_json("alice writes rust.", "Alice likes tea."))
    facts = UserFacts(store, provider)

    added = await facts.extract(MESSAGE)

    assert added == ["Alice likes tea."]
    assert "- Alice writes Rust." in provider.complete_calls[0]["messages"][1]["content"]


async def test_short_and_bot_messages_are_not_sent_to_the_model() -> None:
    provider = FakeProvider(reply=facts_json("Should not be stored."))
    facts = UserFacts(InMemoryStore(), provider)

    assert await facts.extract(stored(1, "hi")) == []
    assert await facts.extract(stored(2, "I am a bot and I like Rust.", is_bot=True)) == []
    assert provider.complete_calls == []


async def test_bad_model_output_adds_nothing() -> None:
    store = InMemoryStore()
    for reply in ("not json", '{"facts": "a string"}', facts_json(1, None, "   ")):
        assert await UserFacts(store, FakeProvider(reply=reply)).extract(MESSAGE) == []
    failing = UserFacts(store, FakeProvider(error=ProviderError("down")))
    assert await failing.extract(MESSAGE) == []
    assert await store.list_facts(7, 200) == []


async def test_facts_are_capped_per_message_and_per_user() -> None:
    store = InMemoryStore()
    many = facts_json("Fact one.", "Fact two.", "Fact three.", "Fact four.")
    facts = UserFacts(store, FakeProvider(reply=many), max_facts=4, max_new_per_message=3)

    assert await facts.extract(MESSAGE) == ["Fact one.", "Fact two.", "Fact three."]

    more = UserFacts(store, FakeProvider(reply=facts_json("Fact five.", "Fact six.")), max_facts=4)
    assert await more.extract(MESSAGE) == ["Fact five."]

    full = FakeProvider(reply=facts_json("Fact seven."))
    assert await UserFacts(store, full, max_facts=4).extract(MESSAGE) == []
    assert full.complete_calls == []


async def test_facts_are_normalised_and_truncated() -> None:
    store = InMemoryStore()
    provider = FakeProvider(reply=facts_json("  Alice   likes\n tea. ", "x" * 500))
    added = await UserFacts(store, provider).extract(MESSAGE)
    assert added == ["Alice likes tea.", "x" * 200]


async def test_users_can_forget_only_their_own_facts() -> None:
    store = InMemoryStore()
    facts = UserFacts(store, FakeProvider())
    await store.add_fact(7, 200, "Alice writes Rust.", None)
    await store.add_fact(7, 200, "Alice likes tea.", None)
    await store.add_fact(8, 200, "Bob writes Go.", None)
    [rust, tea] = await facts.view(7, 200)

    assert await facts.forget(8, rust.fact_id) is False
    assert await facts.forget(7, rust.fact_id) is True
    assert await facts.view(7, 200) == [tea]

    assert await facts.forget_all(7, 200) == 1
    assert await facts.view(7, 200) == []
    assert len(await facts.view(8, 200)) == 1
