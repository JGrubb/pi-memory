#!/usr/bin/env node
/**
 * Migrate from Turso/libSQL memory.db to better-sqlite3 + sqlite-vec format.
 *
 * - Reads old DB (which better-sqlite3 can open since it's standard SQLite + WAL)
 * - Creates new DB with vec0 virtual table
 * - Copies all memories + embeddings
 *
 * Usage: node migrate.cjs [old_db_path] [new_db_path]
 */

const Database = require("better-sqlite3");
const sqliteVec = require("sqlite-vec");
const path = require("path");
const fs = require("fs");

const OLD_PATH =
  process.argv[2] ||
  path.join(
    require("os").homedir(),
    ".pi",
    "agent",
    "memory",
    "memory.db",
  );
const NEW_PATH =
  process.argv[3] ||
  path.join(
    require("os").homedir(),
    ".pi",
    "agent",
    "memory",
    "memory_v2.db",
  );

console.log(`Migrating: ${OLD_PATH} → ${NEW_PATH}`);

// Check old DB exists
if (!fs.existsSync(OLD_PATH)) {
  console.error("Old database not found:", OLD_PATH);
  process.exit(1);
}

// Check new DB doesn't exist (safety)
if (fs.existsSync(NEW_PATH)) {
  console.error("New database already exists:", NEW_PATH);
  console.error("Delete it first if you want to re-migrate.");
  process.exit(1);
}

// Open old DB read-only
const oldDb = new Database(OLD_PATH, { readonly: true });
const rows = oldDb.prepare("SELECT * FROM memories").all();
console.log(`Found ${rows.length} memories to migrate.`);

// Create new DB
const newDb = new Database(NEW_PATH);
sqliteVec.load(newDb);
newDb.pragma("journal_mode = WAL");

// Create schema
newDb.exec(`
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
`);

newDb.exec(`
CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
    embedding float[768] distance_metric=cosine,
    cwd text
);
`);

// Migrate
const insertMem = newDb.prepare(`
  INSERT INTO memories (id, session_id, timestamp, cwd, summary, topics, files_touched, tools_used, user_prompt, response_snippet)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
`);

const insertVec = newDb.prepare(`
  INSERT INTO vec_memories (rowid, embedding, cwd) VALUES (?, ?, ?)
`);

const trx = newDb.transaction(() => {
  let migrated = 0;
  let skippedEmbed = 0;

  for (const row of rows) {
    const info = insertMem.run(
      row.id,
      row.session_id,
      row.timestamp,
      row.cwd,
      row.summary,
      row.topics,
      row.files_touched,
      row.tools_used,
      row.user_prompt,
      row.response_snippet,
    );

    const newRowid = info.lastInsertRowid;

    if (row.embedding && row.embedding.length > 0) {
      // The old embedding is stored as a raw Float32Array buffer (BLOB)
      // Verify it's the right size for 768 dims
      const expectedBytes = 768 * 4; // 768 floats * 4 bytes each
      if (row.embedding.length === expectedBytes) {
        insertVec.run(BigInt(newRowid), row.embedding, row.cwd);
        migrated++;
      } else {
        console.warn(
          `  Unexpected embedding size for ${row.id}: ${row.embedding.length} bytes (expected ${expectedBytes})`,
        );
        skippedEmbed++;
      }
    } else {
      skippedEmbed++;
    }
  }

  console.log(`  Migrated: ${migrated} memories with embeddings`);
  if (skippedEmbed > 0) {
    console.log(`  Skipped embeddings: ${skippedEmbed}`);
  }
});

trx();

// Verify
const count = newDb.prepare("SELECT COUNT(*) as c FROM memories").get();
const vecCount = newDb.prepare("SELECT COUNT(*) as c FROM vec_memories").get();
console.log(`\nVerification:`);
console.log(`  memories table: ${count.c} rows`);
console.log(`  vec_memories table: ${vecCount.c} rows`);

// Quick KNN test
const testRow = newDb
  .prepare(
    "SELECT rowid, cwd FROM memories ORDER BY timestamp DESC LIMIT 1",
  )
  .get();
if (testRow) {
  const testVec = newDb
    .prepare("SELECT embedding FROM vec_memories WHERE rowid = ?")
    .get(testRow.rowid);
  if (testVec) {
    const results = newDb
      .prepare(
        "SELECT rowid, distance FROM vec_memories WHERE embedding MATCH ? AND k = 3",
      )
      .all(testVec.embedding);
    console.log(`  KNN test (3 nearest to most recent): ${results.length} results ✓`);
  }
}

oldDb.close();
newDb.close();

console.log(`\nDone! New database at: ${NEW_PATH}`);
console.log(`\nNext steps:`);
console.log(`  1. Back up old DB:  mv ${OLD_PATH} ${OLD_PATH}.bak`);
console.log(`  2. Move new DB:    mv ${NEW_PATH} ${OLD_PATH}`);
console.log(`  3. Run /reload in all pi sessions`);
