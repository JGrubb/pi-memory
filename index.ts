import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Box, Text } from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";
import { randomUUID } from "node:crypto";
import * as os from "node:os";
import * as path from "node:path";

import { initDb, insertMemory, searchByVector, getStats, getPendingRecords, updateMemoryAfterRetry, upsertSession, updateSessionName, findSimilarSessions, backfillSessionsFromJSONL } from "./db.js";
import { embedText, summarizeInteraction, nameSession } from "./vertex.js";
import { buildSessionContext, formatSearchResults } from "./context.js";
import type { Config, MemoryRecord, ExtractedContent } from "./types.js";

// ---------------------------------------------------------------------------
// Config — set via env vars. GOOGLE_CLOUD_PROJECT is required.
//
//   GOOGLE_CLOUD_PROJECT  — GCP project with Vertex AI enabled (required)
//   PI_MEMORY_REGION      — Vertex AI region (default: "global")
//   PI_MEMORY_EMBED_MODEL — embedding model (default: "gemini-embedding-001")
//   PI_MEMORY_SUMMARIZE_MODEL    — summarization model (default: "claude-haiku-4-5@20251001")
//   PI_MEMORY_SUMMARIZE_PROVIDER — "vertex" (default) or "anthropic"
//   PI_MEMORY_EMBED_DIMS  — embedding dimensions (default: 768)
//   PI_MEMORY_DB_PATH     — database file path (default: ~/.pi/agent/memory/memory.db)
// ---------------------------------------------------------------------------

const CONFIG: Config = {
  gcpProject: process.env.GOOGLE_CLOUD_PROJECT ?? "",
  region: process.env.PI_MEMORY_REGION || "global",
  embeddingModel: process.env.PI_MEMORY_EMBED_MODEL || "gemini-embedding-001",
  summarizeModel: process.env.PI_MEMORY_SUMMARIZE_MODEL || "claude-haiku-4-5@20251001",
  summarizeProvider: (process.env.PI_MEMORY_SUMMARIZE_PROVIDER === "anthropic" ? "anthropic" : "vertex") as "vertex" | "anthropic",
  embeddingDims: Number(process.env.PI_MEMORY_EMBED_DIMS) || 768,
  dbPath:
    process.env.PI_MEMORY_DB_PATH ||
    path.join(os.homedir(), ".pi", "agent", "memory", "memory.db"),
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let dbReady = false;
let injectedThisSession = false;
let cachedContext: string | null = null;
let currentSessionId: string | null = null;
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

function extractFromMessages(messages: any[]): ExtractedContent {
  let userPrompt = "";
  let assistantResponse = "";
  const filesTouched = new Set<string>();
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
  let embedding: Float32Array;
  try {
    embedding = await embedText(summary, CONFIG, "RETRIEVAL_DOCUMENT");
  } catch (err) {
    console.error("[memory] Embedding failed, will retry on next session start:", err);
    embedding = new Float32Array(CONFIG.embeddingDims);
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
// Session naming
// ---------------------------------------------------------------------------

async function performSessionNaming(
  messages: any[],
  cwd: string,
  sessionId: string,
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
  pi.setSessionName(name);
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
    currentSessionId = ctx.sessionManager.getSessionId() ?? randomUUID();
    currentCwd = ctx.cwd;
    agentEndCount = 0;
    conversationBuffer = [];
    namingComplete = !!pi.getSessionName(); // skip if already named from a prior run

    if (!CONFIG.gcpProject) {
      console.error(
        "[memory] GOOGLE_CLOUD_PROJECT env var is not set. " +
        "Set it to a GCP project with Vertex AI enabled. " +
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
      cwd: ctx.cwd,
      sessionFile: ctx.sessionManager.getSessionFile() ?? null,
      name: null,
      mainTopic: null,
      subTopic: null,
      timestamp: Date.now(),
      namedAt: null,
    }).catch((err) => console.error("[memory] Session upsert failed:", err));

    try {
      cachedContext = await buildSessionContext(CONFIG.dbPath, ctx.cwd, currentSessionId);
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

    // One-time backfill of existing named sessions from JSONL files
    const piSessionsDir = path.join(os.homedir(), ".pi", "agent", "sessions");
    backfillSessionsFromJSONL(CONFIG.dbPath, piSessionsDir).catch((err) =>
      console.error("[memory] Session backfill failed:", err),
    );

    // Retry any failed memory records from previous sessions
    retryPendingRecords().catch((err) =>
      console.error("[memory] Backfill failed:", err),
    );
  });

  // -------------------------------------------------------------------------
  // Session switch (/new, /resume): rebuild context
  // -------------------------------------------------------------------------

  pi.on("session_switch", async (_event, ctx) => {
    injectedThisSession = false;
    currentSessionId = ctx.sessionManager.getSessionId() ?? randomUUID();
    currentCwd = ctx.cwd;
    agentEndCount = 0;
    conversationBuffer = [];
    namingComplete = !!pi.getSessionName();

    if (!dbReady) return;

    // Register the new session (no-op if already present, preserves existing name)
    await upsertSession(CONFIG.dbPath, {
      id: currentSessionId,
      cwd: ctx.cwd,
      sessionFile: ctx.sessionManager.getSessionFile() ?? null,
      name: null,
      mainTopic: null,
      subTopic: null,
      timestamp: Date.now(),
      namedAt: null,
    }).catch((err) => console.error("[memory] Session upsert failed:", err));

    try {
      cachedContext = await buildSessionContext(CONFIG.dbPath, ctx.cwd, currentSessionId);
    } catch (err) {
      console.error("[memory] Context rebuild failed:", err);
      cachedContext = null;
    }
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
    const cwd = ctx.cwd;
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

      const namingPromise = performSessionNaming(snapshot, cwd, sessionIdSnapshot).catch((err) =>
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
  // Command: /memory
  // -------------------------------------------------------------------------

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
