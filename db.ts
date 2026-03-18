import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import * as fs from "node:fs";
import * as path from "node:path";
import type { MemoryRecord, SearchResult } from "./types.js";

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
    tools_used      TEXT,
    user_prompt     TEXT,
    response_snippet TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_cwd ON memories(cwd);
CREATE INDEX IF NOT EXISTS idx_memories_ts ON memories(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
`;

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
// Connection helper — short-lived, WAL mode, multi-process safe
// ---------------------------------------------------------------------------

function openDb(dbPath: string): Database.Database {
  const db = new Database(dbPath);
  sqliteVec.load(db);
  db.pragma("journal_mode = WAL");
  db.pragma("busy_timeout = 5000");
  return db;
}

/**
 * Open a short-lived connection, execute fn, then close.
 *
 * better-sqlite3 + WAL mode handles multi-process access natively —
 * no advisory lock file needed.
 */
function withDb<T>(dbPath: string, fn: (db: Database.Database) => T): T {
  const db = openDb(dbPath);
  try {
    return fn(db);
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
  withDb(dbPath, (db) => {
    db.exec(MEMORIES_TABLE);
    db.exec(VEC_TABLE);
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
  withDb(dbPath, (db) => {
    const trx = db.transaction(() => {
      // Upsert into memories table
      const info = db
        .prepare(
          `INSERT INTO memories
           (id, session_id, timestamp, cwd, summary, topics, files_touched, tools_used, user_prompt, response_snippet)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             summary = excluded.summary,
             topics = excluded.topics,
             files_touched = excluded.files_touched,
             tools_used = excluded.tools_used,
             user_prompt = excluded.user_prompt,
             response_snippet = excluded.response_snippet`,
        )
        .run(
          record.id,
          record.sessionId,
          record.timestamp,
          record.cwd,
          JSON.stringify(record.topics),
          JSON.stringify(record.filesTouched),
          JSON.stringify(record.toolsUsed),
          record.userPrompt,
          record.responseSnippet,
        );

      // Get the rowid (works for both insert and update)
      const row = db
        .prepare("SELECT rowid FROM memories WHERE id = ?")
        .get(record.id) as { rowid: number };

      // Upsert into vec0 table — delete + re-insert since vec0 doesn't support UPDATE
      // vec0 requires BigInt for explicit rowid values
      db.prepare("DELETE FROM vec_memories WHERE rowid = ?").run(row.rowid);
      db.prepare(
        "INSERT INTO vec_memories(rowid, embedding, cwd) VALUES (?, ?, ?)",
      ).run(BigInt(row.rowid), vecBuf(embedding), record.cwd);
    });

    trx();
  });
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
  return withDb(dbPath, (db) => {
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
        `SELECT rowid, id, summary, cwd, timestamp, topics, files_touched, user_prompt
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
          userPrompt: m.user_prompt ?? "",
          distance: kr.distance,
        };
      });
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
  return withDb(dbPath, (db) => {
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

    const rows = db.prepare(sql).all(...args) as any[];

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
  return withDb(dbPath, (db) => {
    const rows = db
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
      .all(excludeCwd, limit) as any[];

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
  return withDb(dbPath, (db) => {
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
  });
}
