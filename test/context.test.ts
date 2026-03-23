import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

import { initDb, insertMemory } from "../db.js";
import { buildSessionContext, formatSearchResults } from "../context.js";
import type { MemoryRecord, SearchResult } from "../types.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let tmpDir: string;
let dbPath: string;

function setup(): string {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-memory-ctx-test-"));
  return path.join(tmpDir, "test.db");
}

function makeRecord(overrides: Partial<MemoryRecord> = {}): MemoryRecord {
  return {
    id: `mem-${Math.random().toString(36).slice(2, 8)}`,
    sessionId: "session-1",
    timestamp: Date.now(),
    cwd: "/Users/test/project-a",
    summary: "Refactored billing pipeline",
    topics: ["dbt", "bigquery"],
    filesTouched: ["models/billing.sql"],
    toolsUsed: ["read", "edit"],
    userPrompt: "Refactor billing",
    responseSnippet: "Done.",
    status: "complete",
    rawText: "User: Refactor billing\nAssistant: Done.",
    ...overrides,
  };
}

function makeEmbedding(dims: number = 768): Float32Array {
  const vec = new Float32Array(dims);
  for (let i = 0; i < dims; i++) vec[i] = Math.random() * 2 - 1;
  return vec;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("buildSessionContext", () => {
  beforeEach(() => {
    dbPath = setup();
  });

  afterEach(() => {
    if (tmpDir && fs.existsSync(tmpDir)) {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it("returns null when there are no memories", async () => {
    await initDb(dbPath);
    const ctx = await buildSessionContext(dbPath, "/some/project", null);
    assert.equal(ctx, null);
  });

  it("includes same-cwd memories from previous sessions", async () => {
    await initDb(dbPath);

    await insertMemory(
      dbPath,
      makeRecord({
        id: "prev",
        sessionId: "old-session",
        cwd: "/my/project",
        summary: "Fixed the auth middleware",
        topics: ["auth", "middleware"],
      }),
      makeEmbedding(),
    );

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("Recent work in this directory"), "Should have same-cwd section");
    assert.ok(ctx!.includes("Fixed the auth middleware"), "Should include the summary");
    assert.ok(ctx!.includes("auth"), "Should include topics");
  });

  it("excludes memories from the current session", async () => {
    await initDb(dbPath);

    await insertMemory(
      dbPath,
      makeRecord({
        id: "current",
        sessionId: "this-session",
        cwd: "/my/project",
        summary: "Current session work",
      }),
      makeEmbedding(),
    );
    await insertMemory(
      dbPath,
      makeRecord({
        id: "previous",
        sessionId: "old-session",
        cwd: "/my/project",
        summary: "Previous session work",
      }),
      makeEmbedding(),
    );

    const ctx = await buildSessionContext(dbPath, "/my/project", "this-session");
    assert.ok(ctx !== null);
    assert.ok(!ctx!.includes("Current session work"), "Should NOT include current session");
    assert.ok(ctx!.includes("Previous session work"), "Should include previous session");
  });

  it("includes cross-project memories", async () => {
    await initDb(dbPath);

    await insertMemory(
      dbPath,
      makeRecord({
        id: "other-project",
        sessionId: "s1",
        cwd: "/other/repo",
        summary: "Deployed new terraform module",
        topics: ["terraform"],
      }),
      makeEmbedding(),
    );

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("Recent work in other projects"), "Should have cross-project section");
    assert.ok(ctx!.includes("Deployed new terraform module"));
    assert.ok(ctx!.includes("other/repo"), "Should show shortened path");
  });

  it("handles both same-cwd and cross-project in one context", async () => {
    await initDb(dbPath);

    await insertMemory(
      dbPath,
      makeRecord({
        id: "same",
        sessionId: "old",
        cwd: "/my/project",
        summary: "Updated dbt models",
      }),
      makeEmbedding(),
    );
    await insertMemory(
      dbPath,
      makeRecord({
        id: "cross",
        sessionId: "old",
        cwd: "/other/project",
        summary: "Fixed CI pipeline",
      }),
      makeEmbedding(),
    );

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("Recent work in this directory"));
    assert.ok(ctx!.includes("Recent work in other projects"));
    assert.ok(ctx!.includes("Updated dbt models"));
    assert.ok(ctx!.includes("Fixed CI pipeline"));
  });
});

describe("formatSearchResults", () => {
  it("returns a message when no results", () => {
    const output = formatSearchResults([]);
    assert.ok(output.includes("No matching memories"));
  });

  it("formats results with similarity percentage and metadata", () => {
    const results: SearchResult[] = [
      {
        id: "r1",
        summary: "Migrated billing to hourly grain",
        cwd: "/Users/john/finops",
        timestamp: Date.now() - 86400000, // yesterday
        topics: ["dbt", "billing"],
        filesTouched: ["models/billing.sql"],
        userPrompt: "migrate billing",
        distance: 0.15, // 85% similar
      },
    ];

    const output = formatSearchResults(results);
    assert.ok(output.includes("1 matching memories"));
    assert.ok(output.includes("Migrated billing to hourly grain"));
    assert.ok(output.includes("85%"), "Should show similarity percentage");
    assert.ok(output.includes("dbt"), "Should show topics");
    assert.ok(output.includes("billing.sql"), "Should show files");
    assert.ok(output.includes("john/finops") || output.includes("finops"), "Should show project");
  });

  it("handles results with no topics or files", () => {
    const results: SearchResult[] = [
      {
        id: "r1",
        summary: "Some work",
        cwd: "/project",
        timestamp: Date.now(),
        topics: [],
        filesTouched: [],
        userPrompt: "do stuff",
        distance: 0.3,
      },
    ];

    const output = formatSearchResults(results);
    assert.ok(output.includes("Some work"));
    assert.ok(!output.includes("Files:"), "Should not show Files section when empty");
  });
});
