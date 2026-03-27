# Changelog

All notable changes to pi-memory are documented here.

## Unreleased

### Changed
- **Provider config overhaul**: replaced implicit model-name sniffing with explicit provider enums
  - `embedProvider: "vertex" | "ollama"` — controls where embeddings are computed
  - `summarizeProvider: "vertex-anthropic" | "vertex-google" | "anthropic"` — controls which LLM backend handles summarization; no longer inferred from the model name
  - Added `ollamaUrl` config option for local Ollama embedding support
- **Env var alignment with claude-vertex extension**: memory extension now reads the same Vertex env vars used by the main provider extension
  - `ANTHROPIC_VERTEX_PROJECT_ID` (falls back to `GOOGLE_CLOUD_PROJECT`) for GCP project
  - `GOOGLE_CLOUD_LOCATION` (falls back to `CLOUD_ML_REGION`, default `global`) for region
  - Removed `PI_MEMORY_REGION` — region is no longer configured separately
- **Removed region fixup hack**: the silent `global → us-east5` override for Anthropic-on-Vertex is gone; region is passed through as-is
- **Renamed config fields**: `embeddingModel` → `embedModel`, `embeddingDims` → `embedDims` for naming consistency with `embedProvider`
- **New env vars**:
  - `PI_MEMORY_EMBED_PROVIDER` — `"vertex"` (default) or `"ollama"`
  - `PI_MEMORY_OLLAMA_URL` — Ollama base URL, default `http://localhost:11434`
  - `PI_MEMORY_SUMMARIZE_PROVIDER` — now accepts `"vertex-anthropic"` (default), `"vertex-google"`, or `"anthropic"` (previously `"vertex"` or `"anthropic"`)
- **Startup validation**: GCP project is only required when at least one provider targets Vertex, allowing a fully local config (Ollama embeddings + Anthropic direct) with no GCP dependency

### Refactored
- Extracted `callLLM()` dispatcher in `vertex.ts` — the identical 3-branch provider routing that was copy-pasted into `summarizeInteraction`, `summarizeSession`, and `nameSession` is now a single shared function
- Extracted `embedViaVertex()` and `embedViaOllama()` from `embedText()` for clarity

## 2026-03-26

### Added
- Resources-in-scope tracking: URLs, Jira issues, Confluence pages, Grafana queries, and BigQuery tables referenced during a session are captured automatically from tool calls and surfaced in context
- `pin_resource` tool for explicitly adding resources to session scope

### Fixed
- Show touched files relative to project root instead of basename only

## 2026-03-25

### Added
- LLM-generated session descriptions: at session end, a prose summary of the full session is generated from individual memory summaries and stored alongside the session record
- Automatic session naming at turn 5 with continuation detection (`cont. N` suffix when resuming the same topic in a new session)
- `save_artifact` tool for saving verbatim content (SQL queries, commands, findings) with semantic search

### Changed
- Replaced bullet-point context injection with LLM-generated session description paragraphs
- Backfill now uses `getBackfillCandidates` + `appendSessionInfoToJSONL` instead of scanning JSONL files

### Fixed
- NULL session_id exclusion in `getRecentForCwd`
- Retry logic for records that failed summarization or embedding in a previous session (`pending` / `pending_embed` status)
