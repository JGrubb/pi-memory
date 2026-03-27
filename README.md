# pi-memory

A session and project memory extension for [pi](https://github.com/mariozechner/pi-coding-agent), a coding agent harness.

## Purpose

By default, each pi session starts fresh — the agent has no knowledge of what you worked on yesterday, last week, or in other projects. **pi-memory** solves this by automatically capturing a summary of each coding session and storing it in a local vector database. On the next session start, relevant past context is retrieved semantically and injected into the agent's system prompt before you type your first message.

This means the agent can answer questions like *"did we already add auth to this project?"*, recall decisions made in past sessions, and surface related work from other projects without you having to re-explain everything.

### How it works

1. **Session end** (`agent_end` hook): after each conversation turn, the full message thread is summarized by a small language model (Claude Haiku by default) and the summary is embedded into a 768-dim vector using Gemini's embedding model.
2. **Storage**: the summary, embedding, metadata (topics, files touched, tools used, cwd, timestamp) are stored in a local SQLite database using [sqlite-vec](https://github.com/asg017/sqlite-vec) for fast KNN vector search.
3. **Session start** (`session_start` hook): on the next session, recent memories from the current project and one representative memory from each other project are retrieved and formatted into a context block, injected automatically before the first agent turn.
4. **On-demand search**: the `memory_search` tool is available to the agent at any time for semantic search across all past sessions.

## Features

- **Automatic capture**: no manual action needed — sessions are summarized and stored in the background after each turn
- **Semantic search**: search past work using natural language via `vec0` KNN index
- **Project-aware context**: scopes recent history to the current working directory, plus cross-project highlights
- **Persistent storage**: local SQLite file, no external services required for storage
- **Multi-session safe**: WAL mode + `better-sqlite3` allows concurrent access from multiple pi sessions
- **Flexible AI backend**: embeddings via Vertex AI (Gemini) or a local Ollama model; summarization via Claude on Vertex, Gemini on Vertex, or Anthropic direct API
- **`memory_search` tool**: available in agent prompts for on-demand semantic search

## Installation

```bash
# Clone into pi's extensions directory
git clone https://github.com/JGrubb/pi-memory.git ~/.pi/agent/extensions/memory

# Install dependencies
cd ~/.pi/agent/extensions/memory
npm install
```

## Configuration

Two concerns are configured independently: **embedding** (where vectors are computed) and **summarization** (which LLM processes conversation text). Each has its own provider setting.

### Shared Vertex config

These env vars are shared with the [claude-vertex](../claude-vertex) provider extension — no duplication needed if you're already using it.

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_VERTEX_PROJECT_ID` | — | GCP project with Vertex AI enabled (falls back to `GOOGLE_CLOUD_PROJECT`) |
| `GOOGLE_CLOUD_LOCATION` | `global` | Vertex AI region (falls back to `CLOUD_ML_REGION`) |

### Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `PI_MEMORY_EMBED_PROVIDER` | `vertex` | `vertex` (Gemini via Vertex AI) or `ollama` (local model) |
| `PI_MEMORY_EMBED_MODEL` | `gemini-embedding-001` | Embedding model name |
| `PI_MEMORY_EMBED_DIMS` | `768` | Embedding dimensions — must match the model's output |
| `PI_MEMORY_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL (only used when `embedProvider=ollama`) |

### Summarization

| Variable | Default | Description |
|----------|---------|-------------|
| `PI_MEMORY_SUMMARIZE_PROVIDER` | `vertex-anthropic` | `vertex-anthropic`, `vertex-google`, or `anthropic` |
| `PI_MEMORY_SUMMARIZE_MODEL` | `claude-haiku-4-5@20251001` | Model name appropriate for the chosen provider |
| `ANTHROPIC_API_KEY` | — | Required only when `summarizeProvider=anthropic` |

### Other

| Variable | Default | Description |
|----------|---------|-------------|
| `PI_MEMORY_DB_PATH` | `~/.pi/agent/memory/memory.db` | SQLite database path |

### Common configurations

**Vertex only** — one auth mechanism, no API keys:
```sh
export ANTHROPIC_VERTEX_PROJECT_ID=my-gcp-project
# PI_MEMORY_EMBED_PROVIDER=vertex       (default)
# PI_MEMORY_SUMMARIZE_PROVIDER=vertex-anthropic  (default)
```

**Anthropic direct + Vertex embeddings** — use your Anthropic API key for summarization:
```sh
export ANTHROPIC_VERTEX_PROJECT_ID=my-gcp-project
export ANTHROPIC_API_KEY=sk-ant-...
export PI_MEMORY_SUMMARIZE_PROVIDER=anthropic
```

**Fully local** — no GCP required:
```sh
export PI_MEMORY_EMBED_PROVIDER=ollama
export PI_MEMORY_EMBED_MODEL=nomic-embed-text
export PI_MEMORY_EMBED_DIMS=768
export ANTHROPIC_API_KEY=sk-ant-...
export PI_MEMORY_SUMMARIZE_PROVIDER=anthropic
```

### GCP setup

1. Enable the Vertex AI API in your GCP project
2. Ensure your account has the `aiplatform.endpoints.predict` permission (e.g. `roles/aiplatform.user`)
3. Authenticate with Application Default Credentials:
   ```bash
   gcloud auth application-default login
   ```

## Usage

Once installed and configured, start/restart pi. The extension auto-loads on session start.

On first prompt you'll see a **🧠 Session Memory** block injected with relevant context from past sessions. A notification shows how many memories are loaded.

### `memory_search` tool

Available to the agent for semantic search across all past sessions:

```
memory_search(query: string, limit?: number, project_filter?: string)
```

Example uses in conversation:
- *"Did we already set up auth in this project?"*
- *"What was the approach we took for the DB schema?"*
- *"Search memory for anything related to rate limiting"*

### `/memory` command

Type `/memory` in pi to show memory system stats:

```
🧠 Memory System Stats
  Memories: 42
  Projects: 7
  Sessions: 18
  Oldest: 1/15/2026
  Newest: 3/18/2026
  DB: /Users/you/.pi/agent/memory/memory.db
```

## Architecture

- **index.ts**: Extension entry point — tool/command registration, session lifecycle hooks, message parsing, background storage
- **db.ts**: SQLite + sqlite-vec database layer with connection caching and KNN search
- **vertex.ts**: AI integrations — embedding (Vertex/Ollama) and summarization (Vertex-Anthropic, Vertex-Google, or Anthropic direct) with a shared `callLLM()` dispatcher
- **context.ts**: Session context builder and search result formatter
- **types.ts**: TypeScript type definitions

### Database schema

Stored at `~/.pi/agent/memory/memory.db`:

- **`memories`** table: id, sessionId, cwd, summary, topics, filesTouched, toolsUsed, userPrompt, responseSnippet, timestamp
- **`vec_memories`** virtual table: `vec0` with 768-dim float embeddings, cosine distance metric, `cwd` partition metadata for filtered KNN queries

### Session lifecycle

```
session_start    → init DB → build context from past memories → cache for injection
before_agent_start → inject cached context (first turn only)
agent_end        → summarize → embed → store (background, non-blocking)
session_shutdown → await any pending stores before exit
```

## Development

```bash
# Run tests (36 tests)
npx tsx --test test/*.test.ts
```

## License

MIT
