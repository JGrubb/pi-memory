# pi-memory

A session and project memory extension for [pi](https://github.com/mariozechner/pi-coding-agent), a coding agent harness.

This extension enables semantic search across coding sessions, automatically storing and retrieving relevant context from previous work using vector embeddings.

## Features

- **Session Memory**: Automatically captures summaries and topics from each coding session
- **Semantic Search**: Search past work using natural language queries via `vec0` KNN index
- **Project Filtering**: Scope searches to specific projects or search across all projects
- **Persistent Storage**: Uses SQLite with [sqlite-vec](https://github.com/asg017/sqlite-vec) for vector search
- **Multi-session Safe**: WAL mode + `better-sqlite3` allows concurrent access from multiple pi sessions
- **Custom Tool Integration**: Provides `memory_search` tool for use in agent prompts

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
| `PI_MEMORY_REGION` | No | `global` | Vertex AI region |
| `PI_MEMORY_EMBED_MODEL` | No | `gemini-embedding-001` | Embedding model |
| `PI_MEMORY_HAIKU_MODEL` | No | `claude-haiku-4-5@20251001` | Summarization model |
| `PI_MEMORY_EMBED_DIMS` | No | `768` | Embedding dimensions |
| `PI_MEMORY_DB_PATH` | No | `~/.pi/agent/memory/memory.db` | Database file path |

### GCP Setup

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

## Usage

Once installed and configured, start/restart pi. The extension auto-loads on session start.

### `memory_search` tool

Available in agent prompts for semantic search across past sessions:

```
memory_search(query: string, limit?: number, project_filter?: string)
```

### `/memory` command

Shows memory system stats (total memories, projects, sessions, date range).

## Architecture

- **index.ts**: Extension entry point, tool/command registration, session lifecycle hooks
- **db.ts**: SQLite + sqlite-vec database layer with connection caching and KNN search
- **vertex.ts**: Google Vertex AI integration (embeddings + summarization)
- **context.ts**: Session context builder and search result formatting
- **types.ts**: TypeScript type definitions

### Database

Memories are stored in SQLite at `~/.pi/agent/memory/memory.db`:

- **`memories`** table: metadata (id, summary, topics, timestamps, cwd, etc.)
- **`vec_memories`** virtual table: `vec0` with 768-dim float embeddings, cosine distance metric, `cwd` metadata column for filtered KNN queries

## Development

```bash
# Run tests (36 tests)
npx tsx --test test/*.test.ts
```

## License

MIT
