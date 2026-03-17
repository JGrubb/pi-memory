import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

import {
  initDb,
  insertMemory,
  searchByVector,
  getRecentForCwd,
  getRecentCrossProject,
  getStats,
} from "../db.js";
import type { MemoryRecord } from "../types.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let tmpDir: string;
let dbPath: string;

function makeDbPath(): string {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-memory-test-"));
  return path.join(tmpDir, "test.db");
}

function makeRecord(overrides: Partial<MemoryRecord> = {}): MemoryRecord {
  return {
    id: `mem-${Math.random().toString(36).slice(2, 8)}`,
    sessionId: "session-1",
    timestamp: Date.now(),
    cwd: "/Users/test/project-a",
    summary: "Refactored the billing pipeline to use hourly grain",
    topics: ["dbt", "bigquery", "refactor"],
    filesTouched: ["models/billing.sql", "tests/test_billing.sql"],
    toolsUsed: ["read", "edit", "bash"],
    userPrompt: "Refactor the billing model to hourly grain",
    responseSnippet: "I've updated the billing model...",
    ...overrides,
  };
}

/** Generate a random Float32Array that looks like an embedding. */
function makeEmbedding(dims: number = 768, seed: number = 42): Float32Array {
  const vec = new Float32Array(dims);
  // Simple deterministic pseudo-random based on seed
  let x = seed;
  for (let i = 0; i < dims; i++) {
    x = ((x * 1103515245 + 12345) & 0x7fffffff) >>> 0;
    vec[i] = (x / 0x7fffffff) * 2 - 1; // range [-1, 1]
  }
  // Normalize to unit length
  let norm = 0;
  for (let i = 0; i < dims; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  for (let i = 0; i < dims; i++) vec[i] /= norm;
  return vec;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("Database", () => {
  beforeEach(() => {
    dbPath = makeDbPath();
  });

  afterEach(() => {
    if (tmpDir && fs.existsSync(tmpDir)) {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  describe("initDb", () => {
    it("creates the database file and tables", async () => {
      await initDb(dbPath);
      assert.ok(fs.existsSync(dbPath), "DB file should exist");
    });

    it("is idempotent — can be called multiple times", async () => {
      await initDb(dbPath);
      await initDb(dbPath);
      // Should not throw
    });

    it("creates parent directories if missing", async () => {
      const deepPath = path.join(tmpDir, "a", "b", "c", "test.db");
      await initDb(deepPath);
      assert.ok(fs.existsSync(deepPath));
    });
  });

  describe("insertMemory + getStats", () => {
    it("inserts a memory and reflects in stats", async () => {
      await initDb(dbPath);

      const record = makeRecord();
      const embedding = makeEmbedding(768, 1);
      await insertMemory(dbPath, record, embedding);

      const stats = await getStats(dbPath);
      assert.equal(stats.totalMemories, 1);
      assert.equal(stats.distinctProjects, 1);
      assert.equal(stats.distinctSessions, 1);
    });

    it("handles multiple inserts across projects and sessions", async () => {
      await initDb(dbPath);

      await insertMemory(
        dbPath,
        makeRecord({ id: "m1", sessionId: "s1", cwd: "/project-a" }),
        makeEmbedding(768, 1),
      );
      await insertMemory(
        dbPath,
        makeRecord({ id: "m2", sessionId: "s1", cwd: "/project-a" }),
        makeEmbedding(768, 2),
      );
      await insertMemory(
        dbPath,
        makeRecord({ id: "m3", sessionId: "s2", cwd: "/project-b" }),
        makeEmbedding(768, 3),
      );

      const stats = await getStats(dbPath);
      assert.equal(stats.totalMemories, 3);
      assert.equal(stats.distinctProjects, 2);
      assert.equal(stats.distinctSessions, 2);
    });

    it("upserts on duplicate ID", async () => {
      await initDb(dbPath);

      const record = makeRecord({ id: "dup-1", summary: "original" });
      await insertMemory(dbPath, record, makeEmbedding(768, 1));

      const updated = makeRecord({ id: "dup-1", summary: "updated" });
      await insertMemory(dbPath, updated, makeEmbedding(768, 1));

      const stats = await getStats(dbPath);
      assert.equal(stats.totalMemories, 1);
    });
  });

  describe("searchByVector", () => {
    it("returns results ordered by cosine distance", async () => {
      await initDb(dbPath);

      // Insert 3 memories with different embeddings
      const baseEmbed = makeEmbedding(768, 100);
      const similarEmbed = makeEmbedding(768, 101); // close seed = similar
      const differentEmbed = makeEmbedding(768, 999); // far seed = different

      await insertMemory(
        dbPath,
        makeRecord({ id: "similar", summary: "Similar memory" }),
        similarEmbed,
      );
      await insertMemory(
        dbPath,
        makeRecord({ id: "different", summary: "Different memory" }),
        differentEmbed,
      );
      await insertMemory(
        dbPath,
        makeRecord({ id: "exact", summary: "Exact match" }),
        baseEmbed,
      );

      const results = await searchByVector(dbPath, baseEmbed, 10);
      assert.ok(results.length >= 2, "Should return results");
      // The exact match should be first (distance ≈ 0)
      assert.equal(results[0].id, "exact");
      assert.ok(results[0].distance < 0.01, `Exact match distance should be ~0, got ${results[0].distance}`);
    });

    it("respects the limit parameter", async () => {
      await initDb(dbPath);

      for (let i = 0; i < 5; i++) {
        await insertMemory(
          dbPath,
          makeRecord({ id: `m-${i}` }),
          makeEmbedding(768, i * 10),
        );
      }

      const results = await searchByVector(dbPath, makeEmbedding(768, 0), 2);
      assert.equal(results.length, 2);
    });

    it("filters by cwd when provided", async () => {
      await initDb(dbPath);

      await insertMemory(
        dbPath,
        makeRecord({ id: "a1", cwd: "/project-a" }),
        makeEmbedding(768, 1),
      );
      await insertMemory(
        dbPath,
        makeRecord({ id: "b1", cwd: "/project-b" }),
        makeEmbedding(768, 2),
      );

      const results = await searchByVector(dbPath, makeEmbedding(768, 1), 10, "/project-a");
      assert.equal(results.length, 1);
      assert.equal(results[0].cwd, "/project-a");
    });

    it("returns empty array when no memories exist", async () => {
      await initDb(dbPath);
      const results = await searchByVector(dbPath, makeEmbedding(768, 1), 10);
      assert.equal(results.length, 0);
    });
  });

  describe("getRecentForCwd", () => {
    it("returns memories for matching cwd in reverse chronological order", async () => {
      await initDb(dbPath);

      const now = Date.now();
      await insertMemory(
        dbPath,
        makeRecord({ id: "old", cwd: "/project", timestamp: now - 60000, summary: "Old" }),
        makeEmbedding(768, 1),
      );
      await insertMemory(
        dbPath,
        makeRecord({ id: "new", cwd: "/project", timestamp: now, summary: "New" }),
        makeEmbedding(768, 2),
      );
      await insertMemory(
        dbPath,
        makeRecord({ id: "other", cwd: "/other-project", timestamp: now }),
        makeEmbedding(768, 3),
      );

      const results = await getRecentForCwd(dbPath, "/project", null, 10);
      assert.equal(results.length, 2);
      assert.equal(results[0].id, "new", "Most recent should be first");
      assert.equal(results[1].id, "old");
    });

    it("excludes the specified session", async () => {
      await initDb(dbPath);

      await insertMemory(
        dbPath,
        makeRecord({ id: "current", sessionId: "current-session", cwd: "/project" }),
        makeEmbedding(768, 1),
      );
      await insertMemory(
        dbPath,
        makeRecord({ id: "previous", sessionId: "old-session", cwd: "/project" }),
        makeEmbedding(768, 2),
      );

      const results = await getRecentForCwd(dbPath, "/project", "current-session", 10);
      assert.equal(results.length, 1);
      assert.equal(results[0].id, "previous");
    });

    it("respects the limit", async () => {
      await initDb(dbPath);

      for (let i = 0; i < 5; i++) {
        await insertMemory(
          dbPath,
          makeRecord({ id: `m${i}`, cwd: "/project", timestamp: Date.now() + i }),
          makeEmbedding(768, i),
        );
      }

      const results = await getRecentForCwd(dbPath, "/project", null, 2);
      assert.equal(results.length, 2);
    });
  });

  describe("getRecentCrossProject", () => {
    it("returns one memory per project, excluding current cwd", async () => {
      await initDb(dbPath);

      const now = Date.now();
      await insertMemory(
        dbPath,
        makeRecord({ id: "a1", cwd: "/project-a", timestamp: now }),
        makeEmbedding(768, 1),
      );
      await insertMemory(
        dbPath,
        makeRecord({ id: "a2", cwd: "/project-a", timestamp: now - 1000 }),
        makeEmbedding(768, 2),
      );
      await insertMemory(
        dbPath,
        makeRecord({ id: "b1", cwd: "/project-b", timestamp: now - 500 }),
        makeEmbedding(768, 3),
      );
      await insertMemory(
        dbPath,
        makeRecord({ id: "current", cwd: "/current-project", timestamp: now }),
        makeEmbedding(768, 4),
      );

      const results = await getRecentCrossProject(dbPath, "/current-project", 10);

      // Should get one per project (a and b), not current
      assert.equal(results.length, 2);
      const cwds = results.map((r) => r.cwd);
      assert.ok(!cwds.includes("/current-project"), "Should exclude current project");
      assert.ok(cwds.includes("/project-a"));
      assert.ok(cwds.includes("/project-b"));

      // Should pick the most recent from project-a
      const aResult = results.find((r) => r.cwd === "/project-a");
      assert.equal(aResult?.id, "a1");
    });

    it("returns empty when only current project exists", async () => {
      await initDb(dbPath);

      await insertMemory(
        dbPath,
        makeRecord({ id: "m1", cwd: "/only-project" }),
        makeEmbedding(768, 1),
      );

      const results = await getRecentCrossProject(dbPath, "/only-project", 10);
      assert.equal(results.length, 0);
    });
  });
});
