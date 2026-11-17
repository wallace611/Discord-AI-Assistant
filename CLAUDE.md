# CLAUDE.md

本檔案是給 Claude Code 的專案說明。開始任何工作前先讀完這份文件。

## 專案概述

一隻混合式架構的 Discord AI bot：
- 本地小模型（homeserver 上的 GTX 1070 Ti，8GB VRAM）負責路由、摘要、記憶抽取、簡單看圖、embedding。
- 雲端模型（OpenAI / Azure OpenAI）負責困難推理、agent tool loop、語音轉文字（STT）、語音合成（TTS）、生圖、複雜圖片理解。
- 目標能力：Agent 工具呼叫、長期記憶 / RAG、多模態（看圖、語音頻道、生圖）。

部署環境：homeserver 上的 Incus system container，Ollama 跑在同一個 container 內。

## 架構

```
Discord gateway (discord.py 2.7+, asyncio event queue)
        |
Router (local, qwen3-vl:4b) -- 判斷 intent 與難度 tier
        |
  +-----------------+-------------------+----------------------+
  | Fast reply      | Agent loop        | Media                |
  | local 4B        | cloud model+tools | vision: local / cloud|
  | 閒聊、摘要       | 推理、多步驟工具    | STT/TTS/生圖: cloud   |
  +-----------------+-------------------+----------------------+
        |                   |
Memory store            Tool layer
(Postgres + pgvector)   (SearXNG, Incus sandbox, Discord API, MCP)
```

### 模型分工

| 角色 | 模型 | 位置 | 備註 |
|---|---|---|---|
| Router、摘要、記憶抽取、簡單看圖 | `qwen3-vl:4b` | 本地 Ollama，常駐 | 一個模型包辦，節省 VRAM |
| Embedding | `bge-m3` | 本地 Ollama，常駐 | 中英文皆可 |
| Reranker（P1 後段） | `bge-reranker-v2-m3` | 本地 | 可選 |
| 困難推理、agent | 由 `.env` 設定 | 雲端 | OpenAI 或 Azure OpenAI |
| 複雜圖表 / 文件理解 | 由 `.env` 設定 | 雲端 | router 判斷需要時才升級 |
| STT | 由 `.env` 設定 | 雲端 | 目前沒有第二台 GPU 機器 |
| TTS | 由 `.env` 設定 | 雲端 | 同上 |
| 生圖 | 由 `.env` 設定 | 雲端 | 1070 Ti 跑 SD 太慢 |

VRAM 規則：本地常駐的模型只有 `qwen3-vl:4b` 和 `bge-m3`。不要在本地加入其他常駐模型，除非先確認 `ollama ps` 的總用量仍低於 8GB 並留有 KV cache 空間。所有雲端模型名稱都必須從 `.env` 讀取，不要寫死在程式碼裡。

## Python 環境（必須遵守）

一律使用專案內的 venv，不要使用系統 Python，也不要在 venv 外執行 `pip install`。

### 主 bot（根目錄 `.venv`）

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### 語音 worker（獨立的 `voice_worker/.venv`）

discord.py 和 Py-cord 都會安裝成 `discord` 這個套件名稱，兩者不能裝在同一個環境。語音 worker 必須使用自己的 venv：

```bash
cd voice_worker
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### 規則

- 執行任何指令前先確認已啟用正確的 venv。在非互動 shell 中直接使用 `.venv/bin/python`、`.venv/bin/pytest`。
- 新增依賴時寫進對應的 `pyproject.toml`，再重新執行 `pip install -e ".[dev]"`。不要只手動 `pip install`。
- 不要提交 `.venv/`；確認 `.gitignore` 有包含它。

## 常用指令

```bash
# Run the main bot
.venv/bin/python -m bot

# Run the voice worker
voice_worker/.venv/bin/python -m voice_worker

# Tests
.venv/bin/pytest

# Lint and format
.venv/bin/ruff check . --fix
.venv/bin/ruff format .

# Infrastructure (Postgres + pgvector, SearXNG)
docker compose up -d
```

## 目錄結構

```
.
├── CLAUDE.md
├── pyproject.toml
├── .env.example
├── docker-compose.yml        # postgres (pgvector), searxng
├── migrations/               # SQL schema migrations
├── bot/
│   ├── __main__.py
│   ├── config.py             # pydantic-settings, reads .env
│   ├── logging_setup.py      # JSON-lines logging
│   ├── text.py               # cleanup of raw local-model output
│   ├── gateway/              # discord events, event queue, per-channel locks,
│   │                         # streaming replies, slash commands
│   ├── router/               # intent + tier classification
│   ├── providers/            # LLM provider abstraction
│   │   ├── base.py           # ChatProvider protocol
│   │   ├── ollama.py         # uses Ollama /v1 OpenAI-compatible endpoint
│   │   ├── openai.py
│   │   └── azure.py
│   ├── agent/
│   │   ├── loop.py           # tool-calling loop (step budget, confirmation, output wrapping)
│   │   ├── registry.py       # ToolRegistry
│   │   ├── setup.py          # builds the registry from settings; sandbox + MCP startup
│   │   └── tools/            # one file per tool, plus MCP client (mcp.py)
│   ├── memory/
│   │   ├── store.py          # MemoryStore interface + in-process implementation
│   │   ├── postgres.py       # MemoryStore on Postgres + pgvector, runs migrations
│   │   ├── service.py        # facade used by the gateway
│   │   ├── short_term.py     # recent messages + rolling summary
│   │   ├── ingest.py         # message ingestion and chunking
│   │   ├── rag.py            # vector search + rerank
│   │   └── user_facts.py     # extracted user facts
│   └── media/
│       ├── vision.py         # local first, escalate to cloud
│       └── image_gen.py      # cloud only
├── voice_worker/             # separate venv, Py-cord
│   ├── pyproject.toml
│   └── voice_worker/
│       ├── __main__.py
│       ├── vad.py            # Silero VAD
│       ├── stt.py            # cloud STT
│       └── tts.py            # cloud TTS
└── tests/
```

## 設定（`.env`）

`.env.example` 必須列出所有變數。不要提交 `.env`。

```
DISCORD_TOKEN=
LLM_PROVIDER_CLOUD=openai          # openai | azure
OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=
CLOUD_CHAT_MODEL=
CLOUD_VISION_MODEL=
CLOUD_STT_MODEL=
CLOUD_TTS_MODEL=
CLOUD_IMAGE_MODEL=
OLLAMA_BASE_URL=http://localhost:11434/v1
LOCAL_ROUTER_MODEL=qwen3-vl:4b
LOCAL_EMBED_MODEL=bge-m3
DATABASE_URL=postgresql://bot:bot@localhost:5432/bot
SEARXNG_URL=http://localhost:8080
INCUS_IMAGE=images:debian/12
INCUS_SANDBOX_PROFILE=             # 留空則停用 run_code
RUN_CODE_TIMEOUT_SECONDS=30
MCP_CONFIG_PATH=mcp_servers.json   # 檔案不存在則沒有 MCP 工具
DAILY_CLOUD_TOKEN_BUDGET_PER_USER=
VOICE_WORKER_IPC_URL=
```

### 沙盒 profile（run_code）

`run_code` 只有在 `INCUS_SANDBOX_PROFILE` 指向一個**沒有網路裝置**的 profile 時才會啟用；啟動時會用 `incus profile show` 檢查，有 `type: nic` 就停用並記 error log。建立方式：

```bash
incus profile create sandbox
incus profile device add sandbox root disk path=/ pool=default size=1GiB
# 確認映像檔有 python3
incus launch images:debian/12 t --ephemeral -p sandbox && incus exec t -- python3 --version; incus delete -f t
```

### MCP 伺服器（`mcp_servers.json`，不提交）

```json
{"servers": {"docs": {"transport": "stdio", "command": "uvx", "args": ["mcp-server-docs"], "tier": "read_only"},
             "crm": {"transport": "http", "url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer ..."}}}}
```

`tier` 為 `read_only` / `sandbox` / `side_effect`，預設 `side_effect`（需使用者確認）。工具名稱為 `伺服器名__工具名`。

## 設計規範

### Provider 抽象層
- 所有 LLM 呼叫都經過 `providers/base.py` 的 `ChatProvider` 介面，業務邏輯不得直接 import `openai` SDK。
- 三個 provider 都使用 OpenAI 格式的 messages 與 function calling；Ollama 走 `/v1` 相容 endpoint。
- 切換 provider 只能靠改 `.env`，不需要改程式碼。

### Router
- 輸出固定的 JSON schema：`{"intent": ..., "tier": "local" | "cloud_agent" | "media", "needs_vision": bool}`。
- JSON 解析失敗時預設走 `cloud_agent`，並記錄 log。
- 使用者當天的雲端 token 額度用完時，強制降級為 `local`，並在回覆中說明。

### Agent loop
- 每次最多 8 步工具呼叫；超過時回覆目前進度並結束。
- 工具權限分三級：
  1. 唯讀，可自動執行：`web_search`、`fetch_url`、`search_server_history`、`recall_memory`
  2. 沙盒：`run_code` 在 Incus ephemeral container 中執行，無網路、限時、限記憶體，執行完就刪除
  3. 有副作用，需使用者按按鈕確認：建立 thread、發起投票、`remember`、pin 訊息
- 網頁內容、歷史訊息、工具輸出一律視為資料，不視為指令（防 prompt injection）。工具輸出包在 `<tool_output>` 內，內容中的同名標籤會被跳脫。
- `fetch_url` 只允許公開位址：先解析 DNS、檢查每個位址，再把連線釘在已檢查的 IP 上（Host header 保留原主機名），redirect 每一跳重做檢查；不讀環境的 proxy 設定。
- 8 步的上限算的是工具呼叫總數，不是模型回合數。
- 所有送出的訊息都設定 `allowed_mentions=AllowedMentions.none()`。
- 新工具優先用 MCP server 接入，不要修改 agent 核心。

### 記憶
- 短期：每個頻道或 thread 保留最近 N 則訊息加一份 rolling summary，由本地 4B 模型在背景更新。
- RAG：以「同頻道、時間相近的連續訊息」為一個 chunk，用 `bge-m3` 轉向量後存入 pgvector。查詢時必須依 `guild_id` 過濾，不同伺服器的資料絕不能互相讀取。
- 使用者記憶：由本地 4B 模型非同步抽取。使用者必須能用 `/memory view` 和 `/memory forget` 查看、刪除。
- 中文全文檢索需要分詞；P1 先只做向量檢索加 reranker，BM25 留到之後。

### 多模態
- 看圖：先用本地 `qwen3-vl:4b`；router 判斷是複雜圖表或文件時，改用 `CLOUD_VISION_MODEL`。
- 語音：Discord 自 2026-03-01 起強制語音連線支援 DAVE（E2EE），不支援的連線會收到 close code 4017。discord.py 核心不提供語音接收，所以語音由獨立的 `voice_worker`（Py-cord + `davey`）處理。
  - Pipeline：Silero VAD 切段 -> 雲端 STT -> 透過 IPC 送給主 bot 的 router / agent -> 雲端 TTS -> 播放回語音頻道。
  - P3 開工前第一件事：寫一個最小腳本驗證「加入語音頻道並收到音訊」，確認可行再繼續。
- 生圖：只用雲端 `CLOUD_IMAGE_MODEL`。

### 使用體驗
- 串流回覆：每 1 到 1.5 秒編輯一次訊息，避免撞到 rate limit。
- 長任務自動開 thread，顯示目前執行到哪個工具。
- 回答附上來源（網址、Discord 訊息連結）。
- Slash commands：`/ask`、`/summarize`、`/kb add`、`/memory view`、`/memory forget`、`/model`、`/voice join`、`/voice leave`。
- 提供右鍵選單「問 bot 這則訊息」。

## 程式碼風格

- Python 3.12，全面使用 type hints 與 `async`/`await`。
- 所有程式碼註解與 docstring 一律使用英文。
- 用 `ruff` 做 lint 與 format。
- 用 `logging` 輸出結構化 log，不要用 `print`。
- 不要在程式碼中寫死任何 token、API key 或雲端模型名稱。

## 測試

- 用 `pytest` + `pytest-asyncio`。
- 單元測試不得呼叫真實的雲端 API 或 Discord；一律 mock provider。
- Router 和 agent 維護一組固定的回歸問題（`tests/regression/`），修改 prompt 後必須跑過。

## 實作階段

依序完成，每個階段完成並通過測試後再進入下一階段。

- [x] P0 骨架：venv 與 `pyproject.toml`、config、gateway + event queue、provider 抽象層（三家）、router、串流回覆
- [x] P1 記憶：docker compose（Postgres + pgvector）、訊息 ingest、短期摘要、RAG 檢索、`/memory` 指令
- [ ] P2 Agent：tool loop、SearXNG、`fetch_url`、`search_server_history`、權限分級與確認按鈕、Incus 沙盒、MCP client
- [ ] P3 多模態：看圖（本地加雲端升級）、DAVE 語音接收驗證、voice worker（雲端 STT / TTS）、雲端生圖
- [ ] P4 加固：每位使用者的雲端額度、rate limit、log 與 tracing（可自架 Langfuse）、回歸測試集