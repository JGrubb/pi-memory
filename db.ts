import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import * as fs from "node:fs";
import * as path from "node:path";
import { randomBytes } from "node:crypto";
import type { MemoryRecord, MemoryStatus, MemoryType, Resource, SearchResult, SessionRecord } from "./types.js";

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const MEMORIES_TABLE = `
CREATE TABLE IF NOT EXISTS memories (
    rowid           INTEGER PRIMARY KEY AUTOINCREMENT,
    id              TEXT UNIQUE NOT NULL,
    session_id      TEXT,
    timestamp       INTEGER NOT NULL,
    cwd             TEXT NOT NULL,
    summary         TEXT NOT NULL,
    topics          TEXT,
    files_touched   TEXT,
    resources       TEXT,
    tools_used      TEXT,
    user_prompt     TEXT,
    response_snippet TEXT,
    status          TEXT NOT NULL DEFAULT 'complete',
    raw_text        TEXT,
    type            TEXT NOT NULL DEFAULT 'memory',
    content         TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_cwd ON memories(cwd);
CREATE INDEX IF NOT EXISTS idx_memories_ts ON memories(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
`;

// Migrations for existing databases that predate new columns
const MIGRATIONS = [
  `ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'complete'`,
  `ALTER TABLE memories ADD COLUMN raw_text TEXT`,
  `CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)`,
  `ALTER TABLE memories ADD COLUMN type TEXT NOT NULL DEFAULT 'memory'`,
  `ALTER TABLE memories ADD COLUMN content TEXT`,
  `ALTER TABLE memories ADD COLUMN resources TEXT`,
];

const SESSIONS_TABLE = `
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    cwd          TEXT NOT NULL,
    session_file TEXT,
    name         TEXT,
    main_topic   TEXT,
    sub_topic    TEXT,
    description  TEXT,
    files_touched TEXT NOT NULL DEFAULT '[]',
    resources    TEXT NOT NULL DEFAULT '[]',
    embedding    BLOB,
    timestamp    INTEGER NOT NULL,
    named_at     INTEGER
);

-- Add columns if upgrading from older schema
CREATE TABLE IF NOT EXISTS _migrations (key TEXT PRIMARY KEY);


CREATE INDEX IF NOT EXISTS idx_sessions_cwd ON sessions(cwd);
CREATE INDEX IF NOT EXISTS idx_sessions_ts ON sessions(timestamp DESC);
`;

const SESSIONS_MIGRATIONS = [
  `ALTER TABLE sessions ADD COLUMN description TEXT`,
  `ALTER TABLE sessions ADD COLUMN files_touched TEXT NOT NULL DEFAULT '[]'`,
  `ALTER TABLE sessions ADD COLUMN resources TEXT NOT NULL DEFAULT '[]'`,
];

// vec0 virtual table — cosine distance, with cwd as a metadata column for
// filtered KNN queries. Rowids match the memories table.
const VEC_TABLE = `
CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
    embedding float[768] distance_metric=cosine,
    cwd text
);
`;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Convert a Float32Array to a Buffer that better-sqlite3 / sqlite-vec accept. */
function vecBuf(vec: Float32Array): Buffer {
  return Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength);
}

function safeJsonParse<T>(val: string | null | undefined, fallback: T): T {
  if (!val) return fallback;
  try {
    return JSON.parse(val);
  } catch {
    return fallback;
  }
}

// ---------------------------------------------------------------------------
// Connection cache
//
// Keep one connection per database path open for the lifetime of the process.
// better-sqlite3 with WAL mode supports concurrent multi-process access,
// so a single long-lived connection per process is safe and performs better
// than open/close on every call.
//
// This also works around a sqlite-vec bug where vec0 DELETE operations fail
// with SQLITE_DONE when the extension is loaded on a fresh connection to an
// existing database (observed under node:test async contexts).
// ---------------------------------------------------------------------------

const _connections = new Map<string, Database.Database>();

function getDb(dbPath: string): Database.Database {
  let db = _connections.get(dbPath);
  if (db && db.open) {
    return db;
  }

  db = new Database(dbPath);
  sqliteVec.load(db);
  db.pragma("journal_mode = WAL");
  db.pragma("busy_timeout = 5000");
  _connections.set(dbPath, db);
  return db;
}

/** Close all cached connections (useful for test cleanup). */
export function closeAll(): void {
  for (const db of _connections.values()) {
    if (db.open) db.close();
  }
  _connections.clear();
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

export async function initDb(dbPath: string): Promise<void> {
  const dir = path.dirname(dbPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  const db = getDb(dbPath);
  db.exec(MEMORIES_TABLE);
  db.exec(VEC_TABLE);
  db.exec(SESSIONS_TABLE);

  // Run migrations — each is safe to ignore if the column already exists
  for (const sql of [...MIGRATIONS, ...SESSIONS_MIGRATIONS]) {
    try {
      db.exec(sql);
    } catch (e: any) {
      if (!e.message?.includes("duplicate column name") && !e.message?.includes("already exists")) {
        throw e;
      }
    }
  }

  // One-time data fix: older versions stored the full .jsonl file path as
  // session_id in memories (before commit a74db4b switched to UUID).
  // Normalize them by extracting the UUID from the filename suffix.
  const legacyCount = (db.prepare(
    `SELECT COUNT(*) as n FROM memories WHERE session_id LIKE '%.jsonl'`,
  ).get() as { n: number }).n;
  if (legacyCount > 0) {
    db.prepare(
      `UPDATE memories
       SET session_id = REPLACE(
         substr(session_id, instr(session_id, '_') + 1),
         '.jsonl', ''
       )
       WHERE session_id LIKE '%.jsonl'`,
    ).run();
    console.log(`[memory] Migrated ${legacyCount} legacy session_id(s) from file path to UUID`);
  }
}

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------

export async function insertMemory(
  dbPath: string,
  record: MemoryRecord,
  embedding: Float32Array,
): Promise<void> {
  const db = getDb(dbPath);
  const trx = db.transaction(() => {
    // Upsert into memories table
    db.prepare(
      `INSERT INTO memories
       (id, session_id, timestamp, cwd, summary, topics, files_touched, resources, tools_used, user_prompt, response_snippet, status, raw_text, type, content)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         summary = excluded.summary,
         topics = excluded.topics,
         files_touched = excluded.files_touched,
         resources = excluded.resources,
         tools_used = excluded.tools_used,
         user_prompt = excluded.user_prompt,
         response_snippet = excluded.response_snippet,
         status = excluded.status,
         raw_text = excluded.raw_text,
         type = excluded.type,
         content = excluded.content`,
    ).run(
      record.id,
      record.sessionId,
      record.timestamp,
      record.cwd,
      record.summary,
      JSON.stringify(record.topics),
      JSON.stringify(record.filesTouched),
      JSON.stringify(record.resources),
      JSON.stringify(record.toolsUsed),
      record.userPrompt,
      record.responseSnippet,
      record.status,
      record.rawText,
      record.type,
      record.content,
    );

    // Get the rowid (works for both insert and update)
    const row = db
      .prepare("SELECT rowid FROM memories WHERE id = ?")
      .get(record.id) as { rowid: number };

    // Upsert into vec0 table — vec0 doesn't support UPDATE, so delete + re-insert.
    // The DELETE may throw SQLITE_DONE under certain runtime contexts (a sqlite-vec
    // bug when the native extension is loaded in node:test + tsx). This is safe to
    // ignore — it just means "no rows deleted", which happens on the first insert.
    try {
      db.prepare("DELETE FROM vec_memories WHERE rowid = ?").run(row.rowid);
    } catch (e: any) {
      if (e.code !== "SQLITE_DONE") throw e;
    }
    // If DELETE failed silently (SQLITE_DONE bug), the old row may still exist.
    // Try INSERT; if it fails with UNIQUE constraint, the embedding is unchanged
    // which is acceptable for an upsert with the same embedding.
    try {
      db.prepare(
        "INSERT INTO vec_memories(rowid, embedding, cwd) VALUES (?, ?, ?)",
      ).run(BigInt(row.rowid), vecBuf(embedding), record.cwd);
    } catch (e: any) {
      // If the row already exists (DELETE didn't actually remove it), that's OK
      // for the upsert case — the embedding data is the same.
      if (!e.message?.includes("UNIQUE constraint")) throw e;
    }
  });

  trx();
}

// ---------------------------------------------------------------------------
// Semantic search — uses vec0 KNN index
// ---------------------------------------------------------------------------

export async function searchByVector(
  dbPath: string,
  queryEmbedding: Float32Array,
  limit: number = 10,
  cwdFilter?: string,
): Promise<SearchResult[]> {
  const db = getDb(dbPath);
  let knnSql: string;
  let knnArgs: any[];

  if (cwdFilter) {
    knnSql = `
      SELECT rowid, distance
      FROM vec_memories
      WHERE embedding MATCH ?
        AND k = ?
        AND cwd = ?
    `;
    knnArgs = [vecBuf(queryEmbedding), limit, cwdFilter];
  } else {
    knnSql = `
      SELECT rowid, distance
      FROM vec_memories
      WHERE embedding MATCH ?
        AND k = ?
    `;
    knnArgs = [vecBuf(queryEmbedding), limit];
  }

  const knnRows = db.prepare(knnSql).all(...knnArgs) as Array<{
    rowid: number;
    distance: number;
  }>;

  if (knnRows.length === 0) return [];

  // Fetch full memory records for the matched rowids
  const placeholders = knnRows.map(() => "?").join(",");
  const memRows = db
    .prepare(
      `SELECT rowid, id, summary, cwd, timestamp, topics, files_touched, resources, user_prompt, type, content
       FROM memories
       WHERE rowid IN (${placeholders})`,
    )
    .all(...knnRows.map((r) => r.rowid)) as any[];

  // Build a lookup map
  const memMap = new Map<number, any>();
  for (const r of memRows) memMap.set(r.rowid, r);

  // Merge KNN distances with memory data, preserving KNN order
  return knnRows
    .filter((kr) => memMap.has(kr.rowid))
    .map((kr) => {
      const m = memMap.get(kr.rowid)!;
      return {
        id: m.id,
        summary: m.summary,
        cwd: m.cwd,
        timestamp: m.timestamp,
        topics: safeJsonParse(m.topics, []),
        filesTouched: safeJsonParse(m.files_touched, []),
        resources: safeJsonParse(m.resources, []),
        userPrompt: m.user_prompt ?? "",
        distance: kr.distance,
        type: (m.type ?? "memory") as MemoryType,
        content: m.content ?? null,
      };
    });
}

// ---------------------------------------------------------------------------
// Chronological queries (for session context injection)
// ---------------------------------------------------------------------------

export async function getRecentForCwd(
  dbPath: string,
  cwd: string,
  excludeSessionId: string | null,
  limit: number = 15,
): Promise<SearchResult[]> {
  const db = getDb(dbPath);
  let sql: string;
  const args: any[] = [cwd];

  if (excludeSessionId) {
    sql = `
      SELECT id, session_id, summary, cwd, timestamp, topics, files_touched, resources, user_prompt, type, content
      FROM memories
      WHERE cwd = ? AND (session_id IS NULL OR session_id != ?) AND type = 'memory'
      ORDER BY timestamp DESC
      LIMIT ?
    `;
    args.push(excludeSessionId, limit);
  } else {
    sql = `
      SELECT id, session_id, summary, cwd, timestamp, topics, files_touched, resources, user_prompt, type, content
      FROM memories
      WHERE cwd = ? AND type = 'memory'
      ORDER BY timestamp DESC
      LIMIT ?
    `;
    args.push(limit);
  }

  const rows = db.prepare(sql).all(...args) as any[];

  return rows.map((r: any) => ({
    id: r.id,
    sessionId: r.session_id ?? null,
    summary: r.summary,
    cwd: r.cwd,
    timestamp: r.timestamp,
    topics: safeJsonParse(r.topics, []),
    filesTouched: safeJsonParse(r.files_touched, []),
    resources: safeJsonParse(r.resources, []),
    userPrompt: r.user_prompt ?? "",
    distance: 0,
    type: "memory" as MemoryType,
    content: null,
  }));
}

export async function getRecentCrossProject(
  dbPath: string,
  excludeCwd: string,
  limit: number = 10,
): Promise<SearchResult[]> {
  const db = getDb(dbPath);
  const rows = db
    .prepare(
      `
      SELECT m.id, m.summary, m.cwd, m.timestamp, m.topics, m.files_touched, m.user_prompt
      FROM memories m
      INNER JOIN (
        SELECT cwd, MAX(timestamp) as max_ts
        FROM memories
        WHERE cwd != ? AND type = 'memory'
        GROUP BY cwd
      ) latest ON m.cwd = latest.cwd AND m.timestamp = latest.max_ts
      ORDER BY m.timestamp DESC
      LIMIT ?
    `,
    )
    .all(excludeCwd, limit) as any[];

  return rows.map((r: any) => ({
    id: r.id,
    summary: r.summary,
    cwd: r.cwd,
    timestamp: r.timestamp,
    topics: safeJsonParse(r.topics, []),
    filesTouched: safeJsonParse(r.files_touched, []),
    resources: safeJsonParse(r.resources, []),
    userPrompt: r.user_prompt ?? "",
    distance: 0,
    type: "memory" as MemoryType,
    content: null,
  }));
}

// ---------------------------------------------------------------------------
// Pending record retry support
// ---------------------------------------------------------------------------

export interface PendingRecord {
  id: string;
  rowid: number;
  rawText: string;
  summary: string;  // existing (may be fallback) — used if status is pending_embed
  status: MemoryStatus;
}

export async function getPendingRecords(dbPath: string): Promise<PendingRecord[]> {
  const db = getDb(dbPath);
  const rows = db
    .prepare(
      `SELECT rowid, id, raw_text, summary, status
       FROM memories
       WHERE status != 'complete'
       ORDER BY timestamp ASC`,
    )
    .all() as any[];

  return rows.map((r) => ({
    id: r.id,
    rowid: r.rowid,
    rawText: r.raw_text ?? "",
    summary: r.summary,
    status: r.status as MemoryStatus,
  }));
}

export async function updateMemoryAfterRetry(
  dbPath: string,
  id: string,
  rowid: number,
  summary: string,
  topics: string[],
  embedding: Float32Array,
): Promise<void> {
  const db = getDb(dbPath);
  const trx = db.transaction(() => {
    db.prepare(
      `UPDATE memories SET summary = ?, topics = ?, status = 'complete' WHERE id = ?`,
    ).run(summary, JSON.stringify(topics), id);

    try {
      db.prepare("DELETE FROM vec_memories WHERE rowid = ?").run(rowid);
    } catch (e: any) {
      if (e.code !== "SQLITE_DONE") throw e;
    }
    const cwdRow = db.prepare("SELECT cwd FROM memories WHERE id = ?").get(id) as { cwd: string } | undefined;
    try {
      db.prepare(
        "INSERT INTO vec_memories(rowid, embedding, cwd) VALUES (?, ?, ?)",
      ).run(BigInt(rowid), vecBuf(embedding), cwdRow?.cwd ?? "");
    } catch (e: any) {
      if (!e.message?.includes("UNIQUE constraint")) throw e;
    }
  });
  trx();
}

// ---------------------------------------------------------------------------
// Sessions table
// ---------------------------------------------------------------------------

/** Convert a Buffer (stored BLOB) back to Float32Array. */
function bufToVec(buf: Buffer): Float32Array {
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

/** Dot product of two unit-normalized vectors = cosine similarity. */
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

/**
 * Upsert a session record. Uses INSERT OR IGNORE so existing rows (including
 * their names and embeddings) are never overwritten by a bare upsert.
 */
export async function upsertSession(
  dbPath: string,
  record: SessionRecord,
): Promise<void> {
  const db = getDb(dbPath);
  db.prepare(
    `INSERT OR IGNORE INTO sessions (id, cwd, session_file, name, main_topic, sub_topic, embedding, timestamp, named_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    record.id,
    record.cwd,
    record.sessionFile,
    record.name,
    record.mainTopic,
    record.subTopic,
    null,
    record.timestamp,
    record.namedAt,
  );
}

/**
 * Set the name, topics, and embedding on a session after the naming LLM call.
 */
export async function updateSessionName(
  dbPath: string,
  id: string,
  mainTopic: string,
  subTopic: string,
  name: string,
  embedding: Float32Array,
): Promise<void> {
  const db = getDb(dbPath);
  db.prepare(
    `UPDATE sessions
     SET name = ?, main_topic = ?, sub_topic = ?, embedding = ?, named_at = ?
     WHERE id = ?`,
  ).run(name, mainTopic, subTopic, vecBuf(embedding), Date.now(), id);
}

/**
 * Return all memory summaries and files for a given session, for use in
 * generating a holistic session description.
 */
export async function getSessionMemoriesForSummary(
  dbPath: string,
  sessionId: string,
): Promise<{ summaries: string[]; filesTouched: string[]; resources: Resource[] }> {
  const db = getDb(dbPath);
  const rows = db
    .prepare(
      `SELECT summary, files_touched, resources FROM memories
       WHERE session_id = ? AND type = 'memory'
       ORDER BY timestamp ASC`,
    )
    .all(sessionId) as any[];

  const summaries: string[] = rows.map((r) => r.summary);
  const allFiles = new Set<string>();
  const allResources = new Map<string, Resource>(); // keyed by uri for dedup
  for (const r of rows) {
    for (const f of safeJsonParse(r.files_touched, [])) allFiles.add(f);
    for (const res of safeJsonParse<Resource[]>(r.resources, [])) {
      allResources.set(res.uri, res);
    }
  }

  return { summaries, filesTouched: Array.from(allFiles), resources: Array.from(allResources.values()) };
}

/**
 * Look up a session ID by its session file path.
 * Returns null if no matching session is found.
 */
export function getSessionIdByFile(dbPath: string, sessionFile: string): string | null {
  const db = getDb(dbPath);
  const row = db
    .prepare(`SELECT id FROM sessions WHERE session_file = ? LIMIT 1`)
    .get(sessionFile) as { id: string } | undefined;
  return row?.id ?? null;
}

/**
 * Store a generated description and aggregate files_touched on a session.
 */
export async function updateSessionDescription(
  dbPath: string,
  sessionId: string,
  description: string,
  filesTouched: string[],
  resources: Resource[],
): Promise<void> {
  const db = getDb(dbPath);
  db.prepare(
    `UPDATE sessions SET description = ?, files_touched = ?, resources = ? WHERE id = ?`,
  ).run(description, JSON.stringify(filesTouched), JSON.stringify(resources), sessionId);
}

/**
 * Return recent named sessions for a given cwd, excluding the current session.
 * If a session's files_touched is empty (not yet summarized), falls back to
 * aggregating files from individual memory records for that session.
 */
export async function getRecentSessions(
  dbPath: string,
  cwd: string,
  excludeId: string,
  limit: number = 10,
): Promise<SessionRecord[]> {
  const db = getDb(dbPath);
  const rows = db
    .prepare(
      `SELECT id, cwd, session_file, name, main_topic, sub_topic, description, files_touched, resources, timestamp, named_at
       FROM sessions
       WHERE cwd = ? AND id != ? AND name IS NOT NULL
       ORDER BY timestamp DESC
       LIMIT ?`,
    )
    .all(cwd, excludeId, limit) as any[];

  return rows.map((r) => {
    let filesTouched: string[] = safeJsonParse(r.files_touched, []);
    let resources: Resource[] = safeJsonParse(r.resources, []);

    // If the session hasn't been summarized yet, aggregate files+resources from memories
    if (filesTouched.length === 0 && resources.length === 0) {
      const memRows = db
        .prepare(
          `SELECT files_touched, resources FROM memories WHERE session_id = ? AND type = 'memory'`,
        )
        .all(r.id) as any[];
      const allFiles = new Set<string>();
      const allResources = new Map<string, Resource>();
      for (const m of memRows) {
        for (const f of safeJsonParse(m.files_touched, [])) allFiles.add(f);
        for (const res of safeJsonParse<Resource[]>(m.resources, [])) {
          allResources.set(res.uri, res);
        }
      }
      filesTouched = Array.from(allFiles);
      resources = Array.from(allResources.values());
    }

    return {
      id: r.id,
      cwd: r.cwd,
      sessionFile: r.session_file ?? null,
      name: r.name,
      mainTopic: r.main_topic ?? null,
      subTopic: r.sub_topic ?? null,
      description: r.description ?? null,
      filesTouched,
      resources,
      timestamp: r.timestamp,
      namedAt: r.named_at ?? null,
    };
  });
}

/**
 * Return the most recent named session per project cwd, excluding the current cwd.
 * Used for cross-project chapter headings in the startup context.
 */
export async function getRecentSessionsCrossProject(
  dbPath: string,
  excludeCwd: string,
  limit: number = 8,
): Promise<SessionRecord[]> {
  const db = getDb(dbPath);
  const rows = db
    .prepare(
      `SELECT s.id, s.cwd, s.session_file, s.name, s.main_topic, s.sub_topic, s.description, s.files_touched, s.resources, s.timestamp, s.named_at
       FROM sessions s
       INNER JOIN (
         SELECT cwd, MAX(timestamp) as max_ts
         FROM sessions
         WHERE cwd != ? AND name IS NOT NULL
         GROUP BY cwd
       ) latest ON s.cwd = latest.cwd AND s.timestamp = latest.max_ts AND s.name IS NOT NULL
       ORDER BY s.timestamp DESC
       LIMIT ?`,
    )
    .all(excludeCwd, limit) as any[];

  return rows.map((r) => ({
    id: r.id,
    cwd: r.cwd,
    sessionFile: r.session_file ?? null,
    name: r.name,
    mainTopic: r.main_topic ?? null,
    subTopic: r.sub_topic ?? null,
    description: r.description ?? null,
    filesTouched: safeJsonParse(r.files_touched, []),
    resources: safeJsonParse(r.resources, []),
    timestamp: r.timestamp,
    namedAt: r.named_at ?? null,
  }));
}

/**
 * Find sessions in the same cwd whose main_topic embedding is similar to the
 * given embedding. Results are ordered by similarity descending and filtered
 * to those above the threshold. Sessions without embeddings are skipped.
 */
export async function findSimilarSessions(
  dbPath: string,
  cwd: string,
  excludeId: string,
  queryEmbedding: Float32Array,
  threshold: number = 0.85,
): Promise<SessionRecord[]> {
  const db = getDb(dbPath);
  const rows = db
    .prepare(
      `SELECT id, cwd, session_file, name, main_topic, sub_topic, timestamp, named_at, embedding
       FROM sessions
       WHERE cwd = ? AND id != ? AND name IS NOT NULL AND embedding IS NOT NULL`,
    )
    .all(cwd, excludeId) as any[];

  const scored = rows
    .map((r) => {
      const vec = bufToVec(r.embedding as Buffer);
      const similarity = cosineSimilarity(queryEmbedding, vec);
      return { record: r, similarity };
    })
    .filter(({ similarity }) => similarity >= threshold)
    .sort((a, b) => b.similarity - a.similarity);

  return scored.map(({ record: r }) => ({
    id: r.id,
    cwd: r.cwd,
    sessionFile: r.session_file ?? null,
    name: r.name,
    mainTopic: r.main_topic ?? null,
    subTopic: r.sub_topic ?? null,
    description: null,
    filesTouched: [],
    resources: [],
    timestamp: r.timestamp,
    namedAt: r.named_at ?? null,
  }));
}

// ---------------------------------------------------------------------------
// Backfill — find unnamed sessions and write names back to JSONL
// ---------------------------------------------------------------------------

export interface BackfillCandidate {
  filePath: string;
  sessionId: string;
  cwd: string;
  timestamp: number;
  conversationText: string;
  userTurnCount: number;
}

function extractTextFromContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((p: any) => p?.type === "text" && typeof p.text === "string")
    .map((p: any) => p.text)
    .join("\n");
}

function buildConversationFromEntries(entries: any[]): { text: string; userTurns: number } {
  const parts: string[] = [];
  let userTurns = 0;

  for (const entry of entries) {
    if (entry.type !== "message" || !entry.message) continue;
    const msg = entry.message;

    if (msg.role === "user") {
      const text = extractTextFromContent(msg.content).trim();
      if (text) { parts.push(`User: ${text}`); userTurns++; }
    } else if (msg.role === "assistant") {
      const text = extractTextFromContent(msg.content).trim();
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
  return {
    text: full.length > 6000 ? full.slice(0, 6000) + "\n...[truncated]" : full,
    userTurns,
  };
}

/**
 * Scan the pi sessions directory and return sessions that:
 * - Have at least minTurns user turns
 * - Are not already named (no session_info in JSONL, no name in sessions table)
 * Results are sorted oldest → newest so continuation detection works correctly
 * when the caller processes them in order.
 */
export async function getBackfillCandidates(
  dbPath: string,
  sessionsDir: string,
  minTurns: number = 5,
): Promise<BackfillCandidate[]> {
  if (!fs.existsSync(sessionsDir)) return [];

  const db = getDb(dbPath);
  const candidates: BackfillCandidate[] = [];

  const projectDirs = fs.readdirSync(sessionsDir).filter((d) =>
    fs.statSync(path.join(sessionsDir, d)).isDirectory(),
  );

  for (const projectDir of projectDirs) {
    const projectPath = path.join(sessionsDir, projectDir);
    const files = fs.readdirSync(projectPath).filter((f) => f.endsWith(".jsonl"));

    for (const file of files) {
      const filePath = path.join(projectPath, file);

      let sessionId: string | null = null;
      let sessionCwd: string | null = null;
      let sessionTimestamp = Date.now();
      let hasSessionInfo = false;
      const entries: any[] = [];

      try {
        const content = fs.readFileSync(filePath, "utf8");
        for (const line of content.trim().split("\n")) {
          if (!line.trim()) continue;
          let entry: any;
          try { entry = JSON.parse(line); } catch { continue; }

          if (entry.type === "session") {
            sessionId = entry.id ?? null;
            sessionCwd = entry.cwd ?? null;
            sessionTimestamp = entry.timestamp ? new Date(entry.timestamp).getTime() : Date.now();
          } else if (entry.type === "session_info") {
            hasSessionInfo = true;
          } else {
            entries.push(entry);
          }
        }
      } catch {
        continue;
      }

      if (!sessionId || !sessionCwd || hasSessionInfo) continue;

      // Skip if already named in the sessions table
      const existing = db
        .prepare("SELECT name FROM sessions WHERE id = ?")
        .get(sessionId) as { name: string | null } | undefined;
      if (existing?.name) continue;

      const { text, userTurns } = buildConversationFromEntries(entries);
      if (userTurns < minTurns || text.trim().length < 50) continue;

      candidates.push({ filePath, sessionId, cwd: sessionCwd, timestamp: sessionTimestamp, conversationText: text, userTurnCount: userTurns });
    }
  }

  candidates.sort((a, b) => a.timestamp - b.timestamp);
  return candidates;
}

/**
 * Append a session_info entry to an existing JSONL session file so that
 * pi's /resume list shows the name instead of the first prompt.
 */
export async function appendSessionInfoToJSONL(
  filePath: string,
  name: string,
): Promise<void> {
  const content = fs.readFileSync(filePath, "utf8");
  const lines = content.trim().split("\n").filter((l) => l.trim());

  // Find the last entry with an id to use as parentId
  let lastId: string | null = null;
  for (let i = lines.length - 1; i >= 0; i--) {
    try {
      const entry = JSON.parse(lines[i]);
      if (entry.id) { lastId = entry.id; break; }
    } catch { continue; }
  }

  const newEntry = {
    type: "session_info",
    id: randomBytes(4).toString("hex"),
    parentId: lastId,
    timestamp: new Date().toISOString(),
    name,
  };

  fs.appendFileSync(filePath, JSON.stringify(newEntry) + "\n");
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

export async function getStats(dbPath: string): Promise<{
  totalMemories: number;
  distinctProjects: number;
  distinctSessions: number;
  oldestTimestamp: number | null;
  newestTimestamp: number | null;
}> {
  const db = getDb(dbPath);
  const total = db
    .prepare("SELECT COUNT(*) as c FROM memories")
    .get() as { c: number };
  const projects = db
    .prepare("SELECT COUNT(DISTINCT cwd) as c FROM memories")
    .get() as { c: number };
  const sessions = db
    .prepare("SELECT COUNT(DISTINCT session_id) as c FROM memories")
    .get() as { c: number };
  const oldest = db
    .prepare("SELECT MIN(timestamp) as t FROM memories")
    .get() as { t: number | null };
  const newest = db
    .prepare("SELECT MAX(timestamp) as t FROM memories")
    .get() as { t: number | null };

  return {
    totalMemories: total.c,
    distinctProjects: projects.c,
    distinctSessions: sessions.c,
    oldestTimestamp: oldest.t,
    newestTimestamp: newest.t,
  };
}

// ---------------------------------------------------------------------------
// Obsidian backup — fetch all named sessions and their memories
// ---------------------------------------------------------------------------

/**
 * Return all named sessions across all projects, ordered by project then time.
 * Used for Obsidian vault backup.
 */
export async function getAllNamedSessions(dbPath: string): Promise<SessionRecord[]> {
  const db = getDb(dbPath);
  const rows = db
    .prepare(
      `SELECT id, cwd, session_file, name, main_topic, sub_topic, description, files_touched, resources, timestamp, named_at
       FROM sessions
       WHERE name IS NOT NULL
       ORDER BY cwd, timestamp ASC`,
    )
    .all() as any[];

  return rows.map((r) => ({
    id: r.id,
    cwd: r.cwd,
    sessionFile: r.session_file ?? null,
    name: r.name,
    mainTopic: r.main_topic ?? null,
    subTopic: r.sub_topic ?? null,
    description: r.description ?? null,
    filesTouched: safeJsonParse(r.files_touched, []),
    resources: safeJsonParse(r.resources, []),
    timestamp: r.timestamp,
    namedAt: r.named_at ?? null,
  }));
}

/**
 * Return all complete memories for a given session, ordered by time.
 * Used for Obsidian vault backup.
 */
export async function getMemoriesForSession(dbPath: string, sessionId: string): Promise<SearchResult[]> {
  const db = getDb(dbPath);
  const rows = db
    .prepare(
      `SELECT id, session_id, summary, cwd, timestamp, topics, files_touched, resources, user_prompt, type, content
       FROM memories
       WHERE session_id = ? AND type = 'memory' AND status = 'complete'
       ORDER BY timestamp ASC`,
    )
    .all(sessionId) as any[];

  return rows.map((r) => ({
    id: r.id,
    sessionId: r.session_id ?? null,
    summary: r.summary,
    cwd: r.cwd,
    timestamp: r.timestamp,
    topics: safeJsonParse(r.topics, []),
    filesTouched: safeJsonParse(r.files_touched, []),
    resources: safeJsonParse<Resource[]>(r.resources, []),
    userPrompt: r.user_prompt ?? "",
    distance: 0,
    type: r.type as MemoryType,
    content: r.content ?? null,
  }));
}
