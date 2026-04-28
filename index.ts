import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Box, Text } from "@mariozechner/pi-tui";
import { Type } from "typebox";
import { StringEnum } from "@mariozechner/pi-ai";
import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import * as os from "node:os";
import * as path from "node:path";

import { initDb, insertMemory, searchByVector, getStats, getPendingRecords, updateMemoryAfterRetry, upsertSession, updateSessionName, findSimilarSessions, getBackfillCandidates, appendSessionInfoToJSONL, getSessionMemoriesForSummary, updateSessionDescription, getAllNamedSessions, getMemoriesForSession, getSessionIdByFile } from "./db.js";
import { embedText, summarizeInteraction, nameSession, summarizeSession } from "./vertex.js";
import { buildSessionContext, formatSearchResults } from "./context.js";
import type { Config, MemoryRecord, ExtractedContent, Resource, ResourceType, SessionRecord, SearchResult } from "./types.js";
import * as fs from "node:fs";

// ---------------------------------------------------------------------------
// Resolve the effective project root for a given cwd.
// If the directory is inside a git repo, normalize to the repo root so that
// sessions started from any subdirectory share the same memory scope.
// Falls back to the raw cwd when not in a git repo (e.g. a scratch folder).
// ---------------------------------------------------------------------------
function resolveProjectCwd(cwd: string): string {
  try {
    const root = execFileSync("git", ["rev-parse", "--show-toplevel"], {
      cwd,
      stdio: ["pipe", "pipe", "pipe"],
      encoding: "utf8",
    }).trim();
    return root;
  } catch {
    return cwd;
  }
}

// ---------------------------------------------------------------------------
// Config — env vars. GCP vars are shared with the claude-vertex extension.
//
// Shared Vertex config (when either provider targets Vertex):
//   ANTHROPIC_VERTEX_PROJECT_ID  — GCP project (falls back to GOOGLE_CLOUD_PROJECT)
//   GOOGLE_CLOUD_LOCATION        — Vertex region (falls back to CLOUD_ML_REGION, default: "global")
//
// Embedding:
//   PI_MEMORY_EMBED_PROVIDER  — "vertex" (default) | "ollama"
//   PI_MEMORY_EMBED_MODEL     — default: "gemini-embedding-001"
//   PI_MEMORY_EMBED_DIMS      — default: 768
//   PI_MEMORY_OLLAMA_URL      — default: "http://localhost:11434"
//
// Summarization:
//   PI_MEMORY_SUMMARIZE_PROVIDER — "vertex-anthropic" (default) | "vertex-google" | "anthropic"
//   PI_MEMORY_SUMMARIZE_MODEL    — default: "claude-haiku-4-5@20251001"
//   ANTHROPIC_API_KEY            — required when summarizeProvider is "anthropic"
//
// Other:
//   PI_MEMORY_DB_PATH — default: ~/.pi/agent/memory/memory.db
// ---------------------------------------------------------------------------

const validSummarizeProviders = ["vertex-anthropic", "vertex-google", "anthropic"] as const;
type SummarizeProvider = typeof validSummarizeProviders[number];

function parseSummarizeProvider(val: string | undefined): SummarizeProvider {
  if (val && (validSummarizeProviders as readonly string[]).includes(val)) {
    return val as SummarizeProvider;
  }
  return "vertex-anthropic";
}

function getDefaultSummarizeModel(provider: SummarizeProvider): string {
  switch (provider) {
    case "anthropic":
      return "claude-haiku-4-5-20251001";
    case "vertex-anthropic":
    case "vertex-google":
      return "claude-haiku-4-5@20251001";
  }
}

const summarizeProvider = parseSummarizeProvider(process.env.PI_MEMORY_SUMMARIZE_PROVIDER);

const CONFIG: Config = {
  gcpProject: process.env.ANTHROPIC_VERTEX_PROJECT_ID ?? process.env.GOOGLE_CLOUD_PROJECT ?? "",
  region: process.env.GOOGLE_CLOUD_LOCATION ?? process.env.CLOUD_ML_REGION ?? "global",

  embedProvider: process.env.PI_MEMORY_EMBED_PROVIDER === "ollama" ? "ollama" : "vertex",
  embedModel: process.env.PI_MEMORY_EMBED_MODEL || "gemini-embedding-001",
  embedDims: Number(process.env.PI_MEMORY_EMBED_DIMS) || 768,
  ollamaUrl: process.env.PI_MEMORY_OLLAMA_URL || "http://localhost:11434",

  summarizeProvider,
  summarizeModel: process.env.PI_MEMORY_SUMMARIZE_MODEL || getDefaultSummarizeModel(summarizeProvider),

  dbPath: process.env.PI_MEMORY_DB_PATH || path.join(os.homedir(), ".pi", "agent", "memory", "memory.db"),
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let dbReady = false;
let injectedThisSession = false;
let cachedContext: string | null = null;
let currentSessionId: string | null = null;
let previousSessionId: string | null = null;
let currentCwd: string | null = null;

// Track in-flight stores so we can await them on shutdown
const pendingStores: Set<Promise<void>> = new Set();

// Session naming state — reset on each session start/switch
let agentEndCount = 0;
let conversationBuffer: any[] = [];
let namingComplete = false;

// ---------------------------------------------------------------------------
// Message parsing
// ---------------------------------------------------------------------------

function extractTextParts(content: unknown): string[] {
  if (typeof content === "string") return [content];
  if (!Array.isArray(content)) return [];
  return content
    .filter((p: any) => p?.type === "text" && typeof p.text === "string")
    .map((p: any) => p.text);
}

// ---------------------------------------------------------------------------
// Resource detection helpers
// ---------------------------------------------------------------------------

/**
 * Infer a Resource from a tool name + its arguments. Returns null if the tool
 * call doesn't reference a remote resource we care about.
 */
function detectResource(toolName: string, args: Record<string, any>): Resource | null {
  switch (toolName) {
    case "fetch":
    case "web_search": {
      const url: string = args.url ?? args.query ?? "";
      if (!url) return null;
      return { type: "url", uri: url };
    }

    case "grafana_query":
    case "grafana_query_range": {
      const query: string = args.query ?? "";
      const label = query.length > 60 ? query.slice(0, 60) + "…" : query;
      return { type: "grafana", uri: `grafana:${query}`, label };
    }

    case "bash": {
      const cmd: string = args.command ?? "";
      // BigQuery CLI — match table refs like project.dataset.table
      const bqMatch = cmd.match(/\bbq\b/);
      if (bqMatch) {
        // Extract all fully-qualified table references
        const tableRefs = [...cmd.matchAll(/`?([a-z0-9_-]+\.[a-z0-9_-]+\.[a-z0-9_-]+)`?/gi)];
        if (tableRefs.length > 0) {
          // Return the first one; caller can handle multiples via repeated detection
          return { type: "bigquery", uri: tableRefs[0][1] };
        }
      }
      return null;
    }

    default:
      return null;
  }
}

/**
 * Detect resources from tool name + args, returning potentially multiple
 * (e.g. a bash bq command with several table refs).
 */
function detectResources(toolName: string, args: Record<string, any>): Resource[] {
  // Special case: bash bq commands may touch multiple tables
  if (toolName === "bash") {
    const cmd: string = args.command ?? "";
    if (/\bbq\b/.test(cmd)) {
      const tableRefs = [...cmd.matchAll(/`?([a-z0-9_-]+\.[a-z0-9_-]+\.[a-z0-9_-]+)`?/gi)];
      if (tableRefs.length > 0) {
        return tableRefs.map((m) => ({ type: "bigquery" as ResourceType, uri: m[1] }));
      }
    }
    return [];
  }

  const single = detectResource(toolName, args);
  return single ? [single] : [];
}

function extractFromMessages(messages: any[]): ExtractedContent {
  let userPrompt = "";
  let assistantResponse = "";
  const filesTouched = new Set<string>();
  const resourceMap = new Map<string, Resource>(); // keyed by uri for dedup
  const toolsUsed = new Set<string>();

  for (const msg of messages) {
    if (!msg || !msg.role) continue;

    if (msg.role === "user") {
      const parts = extractTextParts(msg.content);
      if (parts.length > 0) userPrompt = parts.join("\n").trim();
    } else if (msg.role === "assistant") {
      const parts = extractTextParts(msg.content);
      assistantResponse += parts.join("\n").trim() + "\n";

      // Extract tool calls from assistant content
      if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part?.type === "toolCall" && part.name) {
            toolsUsed.add(part.name);
            const args = part.arguments ?? part.input ?? {};
            if (args.path) filesTouched.add(args.path);
            // Detect remote resources
            for (const res of detectResources(part.name, args)) {
              resourceMap.set(res.uri, res);
            }
          }
        }
      }
    } else if (msg.role === "toolResult") {
      if (msg.toolName) toolsUsed.add(msg.toolName);
    }
  }

  return {
    userPrompt,
    assistantResponse: assistantResponse.trim().slice(0, 2000),
    filesTouched: Array.from(filesTouched),
    resources: Array.from(resourceMap.values()),
    toolsUsed: Array.from(toolsUsed),
  };
}

/**
 * Build a truncated conversation string suitable for sending to Haiku.
 */
function buildConversationText(messages: any[]): string {
  const parts: string[] = [];

  for (const msg of messages) {
    if (!msg?.role) continue;

    if (msg.role === "user") {
      const text = extractTextParts(msg.content).join("\n").trim();
      if (text) parts.push(`User: ${text}`);
    } else if (msg.role === "assistant") {
      const text = extractTextParts(msg.content).join("\n").trim();
      if (text) parts.push(`Assistant: ${text}`);

      if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part?.type === "toolCall" && part.name) {
            const argsStr = JSON.stringify(part.arguments ?? part.input ?? {}).slice(0, 200);
            parts.push(`[Tool: ${part.name}(${argsStr})]`);
          }
        }
      }
    }
  }

  const full = parts.join("\n\n");
  return full.length > 6000 ? full.slice(0, 6000) + "\n...[truncated]" : full;
}

// ---------------------------------------------------------------------------
// Background memory storage
// ---------------------------------------------------------------------------

async function processAndStore(
  messages: any[],
  cwd: string,
  sessionId: string,
): Promise<void> {
  const extracted = extractFromMessages(messages);

  // Skip trivial interactions (very short prompts with no tool use)
  if (extracted.userPrompt.length < 20 && extracted.toolsUsed.length === 0) {
    return;
  }

  const conversationText = buildConversationText(messages);
  if (conversationText.trim().length < 50) return;

  // Step 1: Summarize
  let summary: string;
  let topics: string[];
  let status: MemoryRecord["status"] = "complete";
  try {
    const result = await summarizeInteraction(conversationText, CONFIG);
    summary = result.summary;
    topics = result.topics;
  } catch (err) {
    console.error("[memory] Summarization failed, will retry on next session start:", err);
    summary = extracted.userPrompt.slice(0, 300);
    topics = [];
    status = "pending";
  }

  // Step 2: Embed the summary
  let embedding: Float32Array | null;
  try {
    embedding = await embedText(summary, CONFIG, "RETRIEVAL_DOCUMENT");
  } catch (err) {
    console.error("[memory] Embedding failed, will retry on next session start:", err);
    embedding = null;
    if (status === "complete") status = "pending_embed";
  }

  // Step 3: Store in DB
  const record: MemoryRecord = {
    id: randomUUID(),
    sessionId,
    timestamp: Date.now(),
    cwd,
    summary,
    topics,
    filesTouched: extracted.filesTouched,
    resources: extracted.resources,
    toolsUsed: extracted.toolsUsed,
    userPrompt: extracted.userPrompt.slice(0, 500),
    responseSnippet: extracted.assistantResponse.slice(0, 500),
    status,
    rawText: conversationText,
    type: "memory",
    content: null,
  };

  await insertMemory(CONFIG.dbPath, record, embedding);
}

// ---------------------------------------------------------------------------
// Backfill — retry records that failed summarization or embedding
// ---------------------------------------------------------------------------

async function retryPendingRecords(): Promise<void> {
  if (!dbReady) return;
  const pending = await getPendingRecords(CONFIG.dbPath);
  if (pending.length === 0) return;

  console.log(`[memory] Retrying ${pending.length} pending record(s)...`);

  for (const record of pending) {
    try {
      let summary = record.summary;
      let topics: string[] = [];

      if (record.status === "pending") {
        // Need to re-summarize (and then embed)
        const text = record.rawText || record.summary; // rawText preferred
        const result = await summarizeInteraction(text, CONFIG);
        summary = result.summary;
        topics = result.topics;
      }

      const embedding = await embedText(summary, CONFIG, "RETRIEVAL_DOCUMENT");
      await updateMemoryAfterRetry(CONFIG.dbPath, record.id, record.rowid, summary, topics, embedding);
    } catch (err) {
      console.error(`[memory] Retry failed for record ${record.id}, will try again next session:`, err);
    }
  }
}

// ---------------------------------------------------------------------------
// Session naming + backfill
// ---------------------------------------------------------------------------

/**
 * If the previous session has memories but no description yet, generate one.
 * Called at session start/switch so it runs once per new session.
 */
async function summarizePreviousSession(previousSessionId: string): Promise<void> {
  const { summaries, filesTouched, resources } = await getSessionMemoriesForSummary(CONFIG.dbPath, previousSessionId);
  if (summaries.length === 0) return;

  const description = await summarizeSession(summaries, CONFIG);
  await updateSessionDescription(CONFIG.dbPath, previousSessionId, description, filesTouched, resources);
  console.log(`[memory] Summarized previous session (${summaries.length} memories)`);
}

async function runBackfill(sessionsDir: string): Promise<void> {
  const candidates = await getBackfillCandidates(CONFIG.dbPath, sessionsDir, 5);
  if (candidates.length === 0) return;

  console.log(`[memory] Backfilling names for ${candidates.length} session(s)...`);

  for (const candidate of candidates) {
    try {
      // Ensure session row exists before naming it
      await upsertSession(CONFIG.dbPath, {
        id: candidate.sessionId,
        cwd: candidate.cwd,
        sessionFile: candidate.filePath,
        name: null,
        mainTopic: null,
        subTopic: null,
        timestamp: candidate.timestamp,
        namedAt: null,
      });

      const { mainTopic, subTopic } = await nameSession(candidate.conversationText, CONFIG);
      if (!mainTopic) continue;

      const embedding = await embedText(mainTopic, CONFIG, "RETRIEVAL_DOCUMENT");

      // Continuation detection — works correctly because we process oldest first
      const similar = await findSimilarSessions(CONFIG.dbPath, candidate.cwd, candidate.sessionId, embedding, 0.85);
      const sameTopicCount = similar.filter(
        (s) => s.mainTopic?.toLowerCase() === mainTopic.toLowerCase(),
      ).length;

      const name = sameTopicCount > 0
        ? `${mainTopic} - ${subTopic} (cont. ${sameTopicCount + 1})`
        : subTopic ? `${mainTopic} - ${subTopic}` : mainTopic;

      await updateSessionName(CONFIG.dbPath, candidate.sessionId, mainTopic, subTopic, name, embedding);
      await appendSessionInfoToJSONL(candidate.filePath, name);

      console.log(`[memory] Named: "${name}" (${candidate.userTurnCount} turns)`);
    } catch (err) {
      console.error(`[memory] Backfill failed for session ${candidate.sessionId}:`, err);
    }
  }
}


async function performSessionNaming(
  messages: any[],
  cwd: string,
  sessionId: string,
  setSessionName: (name: string) => void,
): Promise<void> {
  const conversationText = buildConversationText(messages);
  if (conversationText.trim().length < 50) return;

  // Step 1: Get main_topic and sub_topic from the LLM
  const { mainTopic, subTopic } = await nameSession(conversationText, CONFIG);
  if (!mainTopic) return;

  // Step 2: Embed the main_topic for continuation detection
  const embedding = await embedText(mainTopic, CONFIG, "RETRIEVAL_DOCUMENT");

  // Step 3: Check for similar sessions in the same cwd
  const similar = await findSimilarSessions(CONFIG.dbPath, cwd, sessionId, embedding, 0.85);

  // Count how many prior sessions share this main_topic
  const sameTopicCount = similar.filter(
    (s) => s.mainTopic?.toLowerCase() === mainTopic.toLowerCase(),
  ).length;

  // Step 4: Build the final name
  let name: string;
  if (sameTopicCount > 0) {
    name = `${mainTopic} - ${subTopic} (cont. ${sameTopicCount + 1})`;
  } else {
    name = subTopic ? `${mainTopic} - ${subTopic}` : mainTopic;
  }

  // Step 5: Persist to sessions table and set the pi session name
  await updateSessionName(CONFIG.dbPath, sessionId, mainTopic, subTopic, name, embedding);
  setSessionName(name);
}

// ---------------------------------------------------------------------------
// Obsidian backup
// ---------------------------------------------------------------------------

function buildObsidianNote(session: SessionRecord, memories: SearchResult[], project: string): string {
  // Aggregate topics, files, resources across all memories
  const allTopics = new Set<string>(session.mainTopic ? [session.mainTopic] : []);
  const allFiles = new Set<string>(session.filesTouched);
  const allResources = new Map<string, Resource>();
  for (const r of session.resources) allResources.set(r.uri, r);

  for (const m of memories) {
    for (const t of m.topics) allTopics.add(t);
    for (const f of m.filesTouched) allFiles.add(f);
    for (const r of m.resources) allResources.set(r.uri, r);
  }

  const date = new Date(session.timestamp).toISOString().slice(0, 10);

  // --- Frontmatter ---
  const fm: string[] = ["---"];
  fm.push(`date: ${date}`);
  fm.push(`project: ${project}`);
  fm.push(`session_id: ${session.id}`);
  if (session.name) fm.push(`name: "${session.name.replace(/"/g, "'")}"`); 
  if (session.mainTopic) fm.push(`main_topic: ${session.mainTopic}`);
  if (session.subTopic) fm.push(`sub_topic: ${session.subTopic}`);

  if (allTopics.size > 0) {
    const topicList = Array.from(allTopics).map((t) => `"${t}"`).join(", ");
    fm.push(`topics: [${topicList}]`);
  }

  if (allFiles.size > 0) {
    fm.push("files:");
    for (const f of allFiles) fm.push(`  - ${f}`);
  }

  if (allResources.size > 0) {
    fm.push("resources:");
    for (const r of allResources.values()) {
      fm.push(`  - type: ${r.type}`);
      fm.push(`    uri: "${r.uri}"`);
      if (r.label) fm.push(`    label: "${r.label.replace(/"/g, "'")}"`);
    }
  }

  fm.push("---");

  // --- Body ---
  const lines: string[] = [...fm, ""];

  lines.push(`# ${session.name ?? "Unnamed Session"}`, "");

  if (session.description) {
    lines.push(session.description, "");
  }

  lines.push("## Memories", "");

  for (const m of memories) {
    const time = new Date(m.timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
    lines.push(`### ${time}`, "");
    lines.push(m.summary, "");

    if (m.userPrompt) {
      const prompt = m.userPrompt.slice(0, 500).replace(/\n+/g, " ").trim();
      lines.push(`**Prompt:** ${prompt}`, "");
    }

    lines.push("---", "");
  }

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
  // -------------------------------------------------------------------------
  // Session start: init DB, build context from previous sessions
  // -------------------------------------------------------------------------

  pi.on("session_start", async (_event, ctx) => {
    injectedThisSession = false;
    cachedContext = null;
    previousSessionId = currentSessionId;
    currentSessionId = ctx.sessionManager.getSessionId() ?? randomUUID();
    currentCwd = resolveProjectCwd(ctx.cwd);
    agentEndCount = 0;
    conversationBuffer = [];
    namingComplete = !!pi.getSessionName(); // skip if already named from a prior run

    const needsVertex = CONFIG.embedProvider === "vertex"
      || CONFIG.summarizeProvider === "vertex-anthropic"
      || CONFIG.summarizeProvider === "vertex-google";
    if (needsVertex && !CONFIG.gcpProject) {
      console.error(
        "[memory] ANTHROPIC_VERTEX_PROJECT_ID (or GOOGLE_CLOUD_PROJECT) is not set. " +
        "Required when embedProvider or summarizeProvider targets Vertex. " +
        "Memory extension disabled.",
      );
      dbReady = false;
      return;
    }

    try {
      await initDb(CONFIG.dbPath);
      dbReady = true;
    } catch (err) {
      console.error("[memory] DB init failed:", err);
      dbReady = false;
      return;
    }

    // Register this session in the sessions table (no-op if already present)
    await upsertSession(CONFIG.dbPath, {
      id: currentSessionId,
      cwd: currentCwd,
      sessionFile: ctx.sessionManager.getSessionFile() ?? null,
      name: null,
      mainTopic: null,
      subTopic: null,
      timestamp: Date.now(),
      namedAt: null,
    }).catch((err) => console.error("[memory] Session upsert failed:", err));

    // Summarize the previous session if it has memories but no description yet.
    // We use event.previousSessionFile (set by pi on /new, /resume, /fork) to look up
    // the session ID reliably, since the extension is re-instantiated on every session
    // switch so module-level previousSessionId is always null at this point.
    const prevFile = ((_event as any).previousSessionFile as string | undefined) ?? null;
    const resolvedPreviousSessionId = prevFile
      ? getSessionIdByFile(CONFIG.dbPath, prevFile)
      : previousSessionId; // fallback for in-process switches (shouldn't happen post-v0.65)
    if (resolvedPreviousSessionId) {
      summarizePreviousSession(resolvedPreviousSessionId).catch((err) =>
        console.error("[memory] Previous session summarization failed:", err),
      );
    }

    try {
      cachedContext = await buildSessionContext(CONFIG.dbPath, currentCwd, currentSessionId);
      if (cachedContext) {
        const stats = await getStats(CONFIG.dbPath);
        ctx.ui.notify(
          `🧠 Memory loaded: ${stats.totalMemories} memories across ${stats.distinctProjects} projects`,
          "info",
        );
      }
    } catch (err) {
      console.error("[memory] Context build failed:", err);
    }

    // Backfill: name any existing sessions that hit the 5-turn threshold
    // but haven't been named yet. Fire-and-forget — fast after first run.
    const piSessionsDir = path.join(os.homedir(), ".pi", "agent", "sessions");
    runBackfill(piSessionsDir).catch((err) =>
      console.error("[memory] Session backfill failed:", err),
    );

    // Retry any failed memory records from previous sessions
    retryPendingRecords().catch((err) =>
      console.error("[memory] Backfill failed:", err),
    );
  });

  // -------------------------------------------------------------------------
  // Before agent start: inject memory context on first prompt only
  // -------------------------------------------------------------------------

  pi.on("before_agent_start", async (_event, _ctx) => {
    if (injectedThisSession || !cachedContext) return;
    injectedThisSession = true;

    return {
      message: {
        customType: "memory-context",
        content: cachedContext,
        display: true,
      },
    };
  });

  // -------------------------------------------------------------------------
  // Renderer: show memory context as a labeled block
  // -------------------------------------------------------------------------

  pi.registerMessageRenderer("memory-context", (message, _options, theme) => {
    const header = theme.fg("accent", theme.bold("🧠 Session Memory"));
    const body = String(message.content);
    const lines = `${header}\n${body}`;

    const box = new Box(1, 1, (t: string) => theme.bg("customMessageBg", t));
    box.addChild(new Text(lines, 0, 0));
    return box;
  });

  // -------------------------------------------------------------------------
  // Agent end: fire-and-forget memory extraction
  // -------------------------------------------------------------------------

  pi.on("agent_end", async (event, ctx) => {
    if (!dbReady) return;

    const messages = event.messages;
    const cwd = currentCwd ?? ctx.cwd;
    const sessionId = currentSessionId ?? "unknown";

    // Accumulate messages for session naming
    conversationBuffer.push(...messages);
    agentEndCount++;

    // Store memory for this turn (fire and forget)
    const promise = processAndStore(messages, cwd, sessionId).catch((err) =>
      console.error("[memory] Background store failed:", err),
    );
    pendingStores.add(promise);
    promise.finally(() => pendingStores.delete(promise));

    // Trigger session naming at turn 5 — enough context to be meaningful,
    // enough friction to filter out throwaway sessions
    if (agentEndCount === 5 && !namingComplete) {
      namingComplete = true; // prevent double-firing
      const snapshot = [...conversationBuffer];
      const sessionIdSnapshot = sessionId;

      const namingPromise = performSessionNaming(snapshot, cwd, sessionIdSnapshot, (name) => pi.setSessionName(name)).catch((err) =>
        console.error("[memory] Session naming failed:", err),
      );
      pendingStores.add(namingPromise);
      namingPromise.finally(() => pendingStores.delete(namingPromise));
    }
  });

  // -------------------------------------------------------------------------
  // Shutdown: wait for pending stores
  // -------------------------------------------------------------------------

  pi.on("session_shutdown", async () => {
    if (pendingStores.size > 0) {
      await Promise.allSettled(Array.from(pendingStores));
    }
  });

  // -------------------------------------------------------------------------
  // Tool: memory_search
  // -------------------------------------------------------------------------

  pi.registerTool({
    name: "memory_search",
    label: "Search Memory",
    description: "Search past coding session memories by semantic similarity",
    promptSnippet:
      "Search past coding session memories for relevant context, decisions, and patterns",
    promptGuidelines: [
      "Use memory_search when the user references past work, asks 'did we already...', or when you need context about previous decisions in this or related projects.",
    ],
    parameters: Type.Object({
      query: Type.String({ description: "What to search for — describe the topic or concept" }),
      limit: Type.Optional(
        Type.Number({ description: "Max results to return (default 10)" }),
      ),
      project_filter: Type.Optional(
        Type.String({
          description: "Filter to a specific project directory path, or omit for all projects",
        }),
      ),
    }),

    async execute(_toolCallId, params, _signal) {
      if (!dbReady) {
        throw new Error("Memory database not initialized");
      }

      let queryEmbedding: Float32Array;
      try {
        queryEmbedding = await embedText(params.query, CONFIG, "RETRIEVAL_QUERY");
      } catch (err) {
        throw new Error(`Failed to embed query: ${err}`);
      }

      const results = await searchByVector(
        CONFIG.dbPath,
        queryEmbedding,
        params.limit ?? 10,
        params.project_filter,
      );

      return {
        content: [{ type: "text", text: formatSearchResults(results) }],
        details: { count: results.length },
      };
    },
  });

  // -------------------------------------------------------------------------
  // Tool: save_artifact
  // -------------------------------------------------------------------------

  pi.registerTool({
    name: "save_artifact",
    label: "Save Artifact",
    description:
      "Save a specific artifact (SQL query, research finding, command, config snippet, etc.) verbatim for precise future recall. Use this when the user asks to remember or save something specific.",
    promptSnippet: "Save a specific artifact verbatim for future recall",
    promptGuidelines: [
      "Use save_artifact when the user says 'remember this', 'save this query', 'save this finding', or similar.",
      "Write a clear, searchable title as the summary — e.g. 'CPU utilization query across dbserver fleet' not just 'query'.",
      "Include the full verbatim content: SQL, results, commands, findings — whatever needs to be precisely recreated.",
    ],
    parameters: Type.Object({
      summary: Type.String({
        description: "Short descriptive title for the artifact — used for display and search",
      }),
      content: Type.String({
        description: "The full verbatim content to save (SQL, results, commands, findings, etc.)",
      }),
      topics: Type.Optional(
        Type.Array(Type.String(), {
          description: "Short lowercase tags (e.g. 'bigquery', 'grafana', 'sql')",
        }),
      ),
    }),

    async execute(_toolCallId, params, _signal) {
      if (!dbReady) {
        throw new Error("Memory database not initialized");
      }

      // Embed title + content together for rich semantic search
      const textToEmbed = `${params.summary}\n\n${params.content}`;
      let embedding: Float32Array;
      try {
        embedding = await embedText(textToEmbed, CONFIG, "RETRIEVAL_DOCUMENT");
      } catch (err) {
        throw new Error(`Failed to embed artifact: ${err}`);
      }

      const record: MemoryRecord = {
        id: randomUUID(),
        sessionId: currentSessionId ?? "unknown",
        timestamp: Date.now(),
        cwd: currentCwd ?? "",
        summary: params.summary,
        topics: params.topics ?? [],
        filesTouched: [],
        resources: [],
        toolsUsed: [],
        userPrompt: "",
        responseSnippet: "",
        status: "complete",
        rawText: "",
        type: "artifact",
        content: params.content,
      };

      await insertMemory(CONFIG.dbPath, record, embedding);

      return {
        content: [{ type: "text", text: `Artifact saved: "${params.summary}"` }],
        details: { id: record.id },
      };
    },
  });

  // -------------------------------------------------------------------------
  // Tool: pin_resource
  // -------------------------------------------------------------------------

  pi.registerTool({
    name: "pin_resource",
    label: "Pin Resource",
    description:
      "Explicitly pin a remote resource (URL, Jira issue, Confluence page, BigQuery table, Grafana query, etc.) to the current session scope so it appears in future session context alongside files in scope.",
    promptSnippet: "Pin a remote resource to the current session scope",
    promptGuidelines: [
      "Use pin_resource when the user asks to 'remember this page', 'keep track of this issue', or 'add this to scope'.",
      "Use it for resources that are semantically important to the session but weren't auto-detected from tool calls.",
      "The uri should be canonical: a full URL, a Jira key like INFRA-1234, a BQ table like project.dataset.table, etc.",
    ],
    parameters: Type.Object({
      uri: Type.String({ description: "Canonical identifier: full URL, Jira key, BQ table name, etc." }),
      type: StringEnum(["url", "confluence", "jira", "metabase", "grafana", "bigquery", "other"] as const, {
        description: "Resource type",
      }),
      label: Type.Optional(
        Type.String({ description: "Human-readable title, e.g. 'INFRA-1234: Fix provisioner bug'" }),
      ),
    }),

    async execute(_toolCallId, params, _signal) {
      if (!dbReady) throw new Error("Memory database not initialized");
      if (!currentSessionId) throw new Error("No active session");

      const resource: Resource = {
        type: params.type as ResourceType,
        uri: params.uri,
        ...(params.label ? { label: params.label } : {}),
      };

      // Store as a lightweight memory record with no conversation content,
      // just so the resource is associated with this session.
      const display = params.label ? `${params.label} (${params.uri})` : params.uri;
      const summaryText = `Pinned resource: ${display}`;

      let embedding: Float32Array | null;
      try {
        embedding = await embedText(summaryText, CONFIG, "RETRIEVAL_DOCUMENT");
      } catch {
        embedding = null;
      }

      const record: MemoryRecord = {
        id: randomUUID(),
        sessionId: currentSessionId,
        timestamp: Date.now(),
        cwd: currentCwd ?? "",
        summary: summaryText,
        topics: [params.type],
        filesTouched: [],
        resources: [resource],
        toolsUsed: [],
        userPrompt: "",
        responseSnippet: "",
        status: "complete",
        rawText: "",
        type: "memory",
        content: null,
      };

      await insertMemory(CONFIG.dbPath, record, embedding);

      return {
        content: [{ type: "text", text: `Pinned: ${display}` }],
        details: { uri: params.uri, type: params.type },
      };
    },
  });

  // -------------------------------------------------------------------------
  // Command: /memory
  // -------------------------------------------------------------------------

  pi.registerCommand("memory-backup", {
    description: "Backup all pi memories to Obsidian vault as markdown files",
    handler: async (_args, ctx) => {
      if (!dbReady) {
        ctx.ui.notify("Memory system not initialized", "warning");
        return;
      }

      const vaultPath =
        process.env.PI_MEMORY_OBSIDIAN_PATH ??
        path.join(os.homedir(), "Documents", "obsidian", "v2025");
      const backupDir = path.join(vaultPath, "pi-memories");

      try {
        const sessions = await getAllNamedSessions(CONFIG.dbPath);
        if (sessions.length === 0) {
          ctx.ui.notify("No named sessions to back up", "info");
          return;
        }

        let written = 0;
        let skipped = 0;

        for (const session of sessions) {
          const memories = await getMemoriesForSession(CONFIG.dbPath, session.id);
          if (memories.length === 0) {
            skipped++;
            continue;
          }

          const project = path.basename(session.cwd);
          const projectDir = path.join(backupDir, project);
          fs.mkdirSync(projectDir, { recursive: true });

          const dateStr = new Date(session.timestamp).toISOString().slice(0, 10);
          const safeName = (session.name ?? session.id.slice(0, 8))
            .replace(/[\/\\:*?"<>|]/g, "-")
            .slice(0, 80)
            .trim();
          const filePath = path.join(projectDir, `${dateStr} - ${safeName}.md`);

          fs.writeFileSync(filePath, buildObsidianNote(session, memories, project), "utf-8");
          written++;
        }

        ctx.ui.notify(
          `✅ Backed up ${written} session${written === 1 ? "" : "s"} to ${backupDir}` +
          (skipped > 0 ? ` (${skipped} skipped — no memories yet)` : ""),
          "info",
        );
      } catch (err) {
        ctx.ui.notify(`❌ Backup failed: ${err}`, "error");
      }
    },
  });

  pi.registerCommand("memory", {
    description: "Show memory system stats and recent memories",
    handler: async (_args, ctx) => {
      if (!dbReady) {
        ctx.ui.notify("Memory system not initialized", "warning");
        return;
      }

      try {
        const stats = await getStats(CONFIG.dbPath);

        const lines = [
          `🧠 Memory System Stats`,
          `  Memories: ${stats.totalMemories}`,
          `  Projects: ${stats.distinctProjects}`,
          `  Sessions: ${stats.distinctSessions}`,
        ];

        if (stats.oldestTimestamp) {
          lines.push(`  Oldest: ${new Date(stats.oldestTimestamp).toLocaleDateString()}`);
        }
        if (stats.newestTimestamp) {
          lines.push(`  Newest: ${new Date(stats.newestTimestamp).toLocaleDateString()}`);
        }
        lines.push(`  DB: ${CONFIG.dbPath}`);

        ctx.ui.notify(lines.join("\n"), "info");
      } catch (err) {
        ctx.ui.notify(`Memory error: ${err}`, "error");
      }
    },
  });
}
