import { connect } from "@tursodatabase/database";
import * as fs from "node:fs";
import * as path from "node:path";
import type { MemoryRecord, SearchResult } from "./types.js";

type Database = Awaited<ReturnType<typeof connect>>;

const SCHEMA = `
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    session_id      TEXT,
    timestamp       INTEGER NOT NULL,
    cwd             TEXT NOT NULL,
    summary         TEXT NOT NULL,
    topics          TEXT,
    files_touched   TEXT,
    tools_used      TEXT,
    user_prompt     TEXT,
    response_snippet TEXT,
    embedding       BLOB
);

CREATE INDEX IF NOT EXISTS idx_memories_cwd ON memories(cwd);
CREATE INDEX IF NOT EXISTS idx_memories_ts ON memories(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
`;

/** Wrap Float32Array as Buffer so the Turso driver preserves all bytes. */
function vecBuf(vec: Float32Array): Buffer {
  return Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength);
}

/**
 * Open a short-lived connection, execute fn, then close.
 * Follows memelord's pattern: Turso's embedded driver locks the file at
 * connect() time, so we keep connections brief and retry on lock contention.
 */
async function withDb<T>(dbPath: string, fn: (db: Database) => Promise<T>): Promise<T> {
  const maxRetries = 10;
  const baseDelay = 50;

  let db: Database;
  for (let attempt = 0; ; attempt++) {
    try {
      db = await connect(dbPath);
      break;
    } catch (e: any) {
      if (
        attempt >= maxRetries ||
        (!e.message?.includes("locked") && !e.message?.includes("Locking"))
      ) {
        throw e;
      }
      const delay = baseDelay * (1 + Math.random()) * Math.min(attempt + 1, 5);
      await new Promise((r) => setTimeout(r, delay));
    }
  }

  await db.exec("PRAGMA busy_timeout = 5000");
  try {
    return await fn(db);
  } finally {
    db.close();
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

export async function initDb(dbPath: string): Promise<void> {
  const dir = path.dirname(dbPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  await withDb(dbPath, async (db) => {
    await db.exec(SCHEMA);
  });
}

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------

export async function insertMemory(
  dbPath: string,
  record: MemoryRecord,
  embedding: Float32Array,
): Promise<void> {
  await withDb(dbPath, async (db) => {
    await db
      .prepare(
        `INSERT INTO memories
         (id, session_id, timestamp, cwd, summary, topics, files_touched, tools_used, user_prompt, response_snippet, embedding)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           summary = excluded.summary,
           topics = excluded.topics,
           files_touched = excluded.files_touched,
           tools_used = excluded.tools_used,
           user_prompt = excluded.user_prompt,
           response_snippet = excluded.response_snippet,
           embedding = excluded.embedding`,
      )
      .run(
        record.id,
        record.sessionId,
        record.timestamp,
        record.cwd,
        record.summary,
        JSON.stringify(record.topics),
        JSON.stringify(record.filesTouched),
        JSON.stringify(record.toolsUsed),
        record.userPrompt,
        record.responseSnippet,
        vecBuf(embedding),
      );
  });
}

// ---------------------------------------------------------------------------
// Semantic search
// ---------------------------------------------------------------------------

export async function searchByVector(
  dbPath: string,
  queryEmbedding: Float32Array,
  limit: number = 10,
  cwdFilter?: string,
): Promise<SearchResult[]> {
  return withDb(dbPath, async (db) => {
    let sql: string;
    const args: any[] = [vecBuf(queryEmbedding), vecBuf(queryEmbedding)];

    if (cwdFilter) {
      sql = `
        SELECT id, summary, cwd, timestamp, topics, files_touched, user_prompt,
               vector_distance_cos(vector32(embedding), vector32(?)) AS distance
        FROM memories
        WHERE embedding IS NOT NULL AND cwd = ?
        ORDER BY vector_distance_cos(vector32(embedding), vector32(?)) ASC
        LIMIT ?
      `;
      args.splice(1, 0, cwdFilter); // insert cwdFilter after first embedding
      args.push(limit);
    } else {
      sql = `
        SELECT id, summary, cwd, timestamp, topics, files_touched, user_prompt,
               vector_distance_cos(vector32(embedding), vector32(?)) AS distance
        FROM memories
        WHERE embedding IS NOT NULL
        ORDER BY vector_distance_cos(vector32(embedding), vector32(?)) ASC
        LIMIT ?
      `;
      args.push(limit);
    }

    const rows = (await db.prepare(sql).all(...args)) as any[];

    return rows.map((r: any) => ({
      id: r.id,
      summary: r.summary,
      cwd: r.cwd,
      timestamp: r.timestamp,
      topics: safeJsonParse(r.topics, []),
      filesTouched: safeJsonParse(r.files_touched, []),
      userPrompt: r.user_prompt ?? "",
      distance: r.distance,
    }));
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
  return withDb(dbPath, async (db) => {
    let sql: string;
    const args: any[] = [cwd];

    if (excludeSessionId) {
      sql = `
        SELECT id, summary, cwd, timestamp, topics, files_touched, user_prompt
        FROM memories
        WHERE cwd = ? AND session_id != ?
        ORDER BY timestamp DESC
        LIMIT ?
      `;
      args.push(excludeSessionId, limit);
    } else {
      sql = `
        SELECT id, summary, cwd, timestamp, topics, files_touched, user_prompt
        FROM memories
        WHERE cwd = ?
        ORDER BY timestamp DESC
        LIMIT ?
      `;
      args.push(limit);
    }

    const rows = (await db.prepare(sql).all(...args)) as any[];

    return rows.map((r: any) => ({
      id: r.id,
      summary: r.summary,
      cwd: r.cwd,
      timestamp: r.timestamp,
      topics: safeJsonParse(r.topics, []),
      filesTouched: safeJsonParse(r.files_touched, []),
      userPrompt: r.user_prompt ?? "",
      distance: 0,
    }));
  });
}

export async function getRecentCrossProject(
  dbPath: string,
  excludeCwd: string,
  limit: number = 10,
): Promise<SearchResult[]> {
  return withDb(dbPath, async (db) => {
    // Get the most recent memory per distinct cwd, excluding current project
    const rows = (await db
      .prepare(
        `
        SELECT m.id, m.summary, m.cwd, m.timestamp, m.topics, m.files_touched, m.user_prompt
        FROM memories m
        INNER JOIN (
          SELECT cwd, MAX(timestamp) as max_ts
          FROM memories
          WHERE cwd != ?
          GROUP BY cwd
        ) latest ON m.cwd = latest.cwd AND m.timestamp = latest.max_ts
        ORDER BY m.timestamp DESC
        LIMIT ?
      `,
      )
      .all(excludeCwd, limit)) as any[];

    return rows.map((r: any) => ({
      id: r.id,
      summary: r.summary,
      cwd: r.cwd,
      timestamp: r.timestamp,
      topics: safeJsonParse(r.topics, []),
      filesTouched: safeJsonParse(r.files_touched, []),
      userPrompt: r.user_prompt ?? "",
      distance: 0,
    }));
  });
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
  return withDb(dbPath, async (db) => {
    const total = (await db
      .prepare("SELECT COUNT(*) as c FROM memories")
      .get()) as { c: number };
    const projects = (await db
      .prepare("SELECT COUNT(DISTINCT cwd) as c FROM memories")
      .get()) as { c: number };
    const sessions = (await db
      .prepare("SELECT COUNT(DISTINCT session_id) as c FROM memories")
      .get()) as { c: number };
    const oldest = (await db
      .prepare("SELECT MIN(timestamp) as t FROM memories")
      .get()) as { t: number | null };
    const newest = (await db
      .prepare("SELECT MAX(timestamp) as t FROM memories")
      .get()) as { t: number | null };

    return {
      totalMemories: total.c,
      distinctProjects: projects.c,
      distinctSessions: sessions.c,
      oldestTimestamp: oldest.t,
      newestTimestamp: newest.t,
    };
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function safeJsonParse<T>(val: string | null | undefined, fallback: T): T {
  if (!val) return fallback;
  try {
    return JSON.parse(val);
  } catch {
    return fallback;
  }
}
