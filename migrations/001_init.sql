-- Memory schema: raw messages, RAG chunks, rolling summaries, user facts.

CREATE EXTENSION IF NOT EXISTS vector;

-- A chunk is a run of consecutive, close-in-time messages from one channel.
-- vector(1024) matches bge-m3; changing the embedding model needs a new migration.
CREATE TABLE chunks (
    chunk_id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    guild_id         BIGINT,                -- NULL for DMs
    channel_id       BIGINT      NOT NULL,
    first_message_id BIGINT      NOT NULL,
    last_message_id  BIGINT      NOT NULL,
    started_at       TIMESTAMPTZ NOT NULL,
    ended_at         TIMESTAMPTZ NOT NULL,
    content          TEXT        NOT NULL,
    embedding        vector(1024) NOT NULL
);

CREATE INDEX chunks_guild_idx ON chunks (guild_id);
CREATE INDEX chunks_channel_idx ON chunks (channel_id);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE messages (
    message_id  BIGINT PRIMARY KEY,         -- Discord snowflake
    guild_id    BIGINT,                     -- NULL for DMs
    channel_id  BIGINT      NOT NULL,       -- thread id when in a thread
    author_id   BIGINT      NOT NULL,
    author_name TEXT        NOT NULL,
    is_bot      BOOLEAN     NOT NULL DEFAULT FALSE,
    content     TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL,
    chunk_id    BIGINT REFERENCES chunks (chunk_id) ON DELETE SET NULL
);

CREATE INDEX messages_channel_idx ON messages (channel_id, message_id);
CREATE INDEX messages_unchunked_idx ON messages (channel_id, message_id) WHERE chunk_id IS NULL;

CREATE TABLE channel_summaries (
    channel_id         BIGINT PRIMARY KEY,
    guild_id           BIGINT,
    summary            TEXT        NOT NULL,
    covered_message_id BIGINT      NOT NULL,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE user_facts (
    fact_id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id           BIGINT      NOT NULL,
    guild_id          BIGINT,               -- NULL for facts learned in DMs
    fact              TEXT        NOT NULL,
    source_message_id BIGINT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX user_facts_unique_idx
    ON user_facts (user_id, COALESCE(guild_id, 0), lower(fact));
