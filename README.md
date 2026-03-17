# pi-memory

A session and project memory extension for [pi](https://github.com/mariozechner/pi-coding-agent), a coding agent harness.

This extension enables semantic search across coding sessions, automatically storing and retrieving relevant context from previous work using vector embeddings.

## Features

- **Session Memory**: Automatically captures summaries and topics from each coding session
- **Semantic Search**: Search past work using natural language queries with vector embeddings
- **Project Filtering**: Scope searches to specific projects or search across all projects
- **Persistent Storage**: Uses Turso SQLite for reliable memory persistence
- **Custom Tool Integration**: Provides `memory_search` tool for use in agent prompts

## Installation

1. Place this extension in your pi extensions directory:
   ```bash
   cp -r pi-memory ~/.pi/agent/extensions/
   ```

2. Add to your `.pirc.yaml` or pi config:
   ```yaml
   extensions:
     - ./extensions/memory
   ```

3. Reload pi to activate the extension

## Usage

Once installed, the `memory_search` tool is available for use in your pi agent:

```
memory_search(query: string, limit?: number = 10, project_filter?: string)
```

### Examples

Search across all sessions:
```
memory_search("database performance optimization")
```

Search within a specific project:
```
memory_search("authentication issues", project_filter: "/Users/johngrubb/my-project")
```

Return top 5 most relevant memories:
```
memory_search("API integration", limit: 5)
```

## Architecture

- **index.ts**: Extension entry point, memory tool registration, session memory capture
- **db.ts**: Turso database management, vector storage, semantic search
- **vertex.ts**: Google Vertex AI integration for embedding generation
- **context.ts**: Memory context rendering for agent prompts
- **types.ts**: TypeScript type definitions

## Database

Memories are stored in a Turso SQLite database at `~/.pi/agent/memory/` with vector embeddings (768-dimensional via Vertex AI Embeddings API).

## Requirements

- Google Cloud credentials with Vertex AI access
- Turso database URL and auth token

## License

MIT
