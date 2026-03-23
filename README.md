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
- **Hybrid AI backend**: embeddings via Vertex AI (Gemini); summarization via Anthropic API directly (Claude Haiku) — avoiding Vertex quota limits on Anthropic-hosted models
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

Set the following environment variables (e.g. in your shell profile):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_CLOUD_PROJECT` | **Yes** | — | GCP project with Vertex AI API enabled |
| `PI_MEMORY_REGION` | No | `global` | Vertex AI region for embeddings. **Note:** if using a Claude summarization model via Vertex, set this to a region that supports Anthropic models (e.g. `us-east5`) — `global` will not work for Claude on Vertex. |
| `PI_MEMORY_SUMMARIZE_MODEL` | No | `claude-haiku-4-5@20251001` | Summarization model name |
| `PI_MEMORY_SUMMARIZE_PROVIDER` | No | `vertex` | Summarization provider: `vertex` (uses GCP ADC, no extra key needed) or `anthropic` (requires `ANTHROPIC_API_KEY`) |
| `ANTHROPIC_API_KEY` | No | — | Only required when `PI_MEMORY_SUMMARIZE_PROVIDER=anthropic` |
| `PI_MEMORY_EMBED_MODEL` | No | `gemini-embedding-001` | Embedding model (always Vertex AI) |
| `PI_MEMORY_EMBED_DIMS` | No | `768` | Embedding dimensions |
| `PI_MEMORY_DB_PATH` | No | `~/.pi/agent/memory/memory.db` | Database file path |

**Recommended setup for GCP-only environments** (no Anthropic API key):

```sh
export GOOGLE_CLOUD_PROJECT=my-gcp-project
export PI_MEMORY_SUMMARIZE_PROVIDER=vertex
export PI_MEMORY_SUMMARIZE_MODEL=claude-haiku-4-5@20251001
export PI_MEMORY_REGION=us-east5   # required for Claude on Vertex
```

### GCP Setup (for embeddings)

1. Enable the Vertex AI API in your GCP project
2. Ensure your account has the `aiplatform.endpoints.predict` permission (e.g. `roles/aiplatform.user`)
3. Set up Application Default Credentials:
   ```bash
   gcloud auth application-default login
   ```
4. Export your project:
   ```bash
   export GOOGLE_CLOUD_PROJECT=your-gcp-project-id
   ```

### Anthropic API Setup (for summarization)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
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
- **vertex.ts**: AI integrations — Vertex AI (Gemini embeddings) and Anthropic API (Claude summarization)
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
