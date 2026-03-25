import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

import { initDb, insertMemory, upsertSession, updateSessionName } from "../db.js";
import { buildSessionContext, formatSearchResults } from "../context.js";
import type { MemoryRecord, SearchResult, SessionRecord } from "../types.js";

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
    type: "memory",
    content: null,
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

    // Cross-project section now comes from named sessions, not raw memories.
    // Create both a memory and a named session for the other project.
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
    await upsertSession(dbPath, makeSessionRecord({ id: "s1", cwd: "/other/repo" }));
    await updateSessionName(dbPath, "s1", "Terraform", "module deployment", "Terraform - module deployment", makeSessionEmbedding());

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("Recent work in other projects"), "Should have cross-project section");
    assert.ok(ctx!.includes("other/repo"), "Should show shortened path");
  });

  it("handles both same-cwd and cross-project in one context", async () => {
    await initDb(dbPath);

    // Same-cwd memory
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

    // Cross-project: needs a named session
    await insertMemory(
      dbPath,
      makeRecord({
        id: "cross",
        sessionId: "cross-session",
        cwd: "/other/project",
        summary: "Fixed CI pipeline",
      }),
      makeEmbedding(),
    );
    await upsertSession(dbPath, makeSessionRecord({ id: "cross-session", cwd: "/other/project" }));
    await updateSessionName(dbPath, "cross-session", "CI Pipeline", "fix flaky tests", "CI Pipeline - fix flaky tests", makeSessionEmbedding());

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("Recent work in this directory"));
    assert.ok(ctx!.includes("Recent work in other projects"));
    assert.ok(ctx!.includes("Updated dbt models"));
    assert.ok(ctx!.includes("CI Pipeline - fix flaky tests"), "Cross-project should show session name");
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

// ---------------------------------------------------------------------------
// Session chapter headings in buildSessionContext
// ---------------------------------------------------------------------------

function makeSessionRecord(overrides: Partial<SessionRecord> = {}): SessionRecord {
  return {
    id: `sess-${Math.random().toString(36).slice(2, 8)}`,
    cwd: "/Users/test/project-a",
    sessionFile: null,
    name: null,
    mainTopic: null,
    subTopic: null,
    timestamp: Date.now(),
    namedAt: null,
    ...overrides,
  };
}

function makeSessionEmbedding(dims = 768): Float32Array {
  const vec = new Float32Array(dims);
  for (let i = 0; i < dims; i++) vec[i] = Math.random() * 2 - 1;
  return vec;
}

describe("buildSessionContext — session chapter headings", () => {
  beforeEach(() => { dbPath = setup(); });
  afterEach(() => {
    if (tmpDir && fs.existsSync(tmpDir)) fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it("uses session name as chapter heading instead of time-proximity grouping", async () => {
    await initDb(dbPath);

    // Named session with two memories
    const s = makeSessionRecord({ id: "old-session", cwd: "/my/project", timestamp: Date.now() - 86400000 });
    await upsertSession(dbPath, s);
    await updateSessionName(dbPath, "old-session", "GCP Billing", "cost export fix", "GCP Billing - cost export fix", makeSessionEmbedding());

    await insertMemory(dbPath, makeRecord({
      id: "m1", sessionId: "old-session", cwd: "/my/project",
      summary: "Rewrote the export query", timestamp: Date.now() - 86400000,
    }), makeEmbedding());
    await insertMemory(dbPath, makeRecord({
      id: "m2", sessionId: "old-session", cwd: "/my/project",
      summary: "Added partition pruning", timestamp: Date.now() - 86300000,
    }), makeEmbedding());

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("GCP Billing - cost export fix"), "Session name should be the chapter heading");
    assert.ok(ctx!.includes("Rewrote the export query"));
    assert.ok(ctx!.includes("Added partition pruning"));
  });

  it("falls back to date heading for unnamed sessions", async () => {
    await initDb(dbPath);

    // Session in DB but not yet named
    await upsertSession(dbPath, makeSessionRecord({ id: "old-session", cwd: "/my/project" }));

    await insertMemory(dbPath, makeRecord({
      id: "m1", sessionId: "old-session", cwd: "/my/project",
      summary: "Did some exploratory work",
    }), makeEmbedding());

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("Did some exploratory work"));
    // Should fall back to a date-based heading, not crash or show null
    assert.ok(!ctx!.includes("null"), "Should not render 'null' as a heading");
  });

  it("shows session names for cross-project recent work", async () => {
    await initDb(dbPath);

    const s = makeSessionRecord({ id: "other-session", cwd: "/other/project", timestamp: Date.now() - 3600000 });
    await upsertSession(dbPath, s);
    await updateSessionName(dbPath, "other-session", "Metabase", "Cloud Run upgrade", "Metabase - Cloud Run upgrade", makeSessionEmbedding());

    await insertMemory(dbPath, makeRecord({
      id: "m1", sessionId: "other-session", cwd: "/other/project",
      summary: "Upgraded Metabase to v0.59",
    }), makeEmbedding());

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("Metabase - Cloud Run upgrade"), "Cross-project entry should show session name");
  });

  it("groups memories by real session id, not time proximity", async () => {
    await initDb(dbPath);
    const now = Date.now();

    // Two sessions whose memories interleave in timestamp but belong to different sessions
    const sA = makeSessionRecord({ id: "session-a", cwd: "/my/project", timestamp: now - 10000 });
    const sB = makeSessionRecord({ id: "session-b", cwd: "/my/project", timestamp: now - 5000 });
    await upsertSession(dbPath, sA);
    await upsertSession(dbPath, sB);
    await updateSessionName(dbPath, "session-a", "Topic A", "alpha", "Topic A - alpha", makeSessionEmbedding());
    await updateSessionName(dbPath, "session-b", "Topic B", "beta", "Topic B - beta", makeSessionEmbedding());

    // Memories from session A and B with interleaved timestamps
    await insertMemory(dbPath, makeRecord({ id: "a1", sessionId: "session-a", cwd: "/my/project", timestamp: now - 9000 }), makeEmbedding());
    await insertMemory(dbPath, makeRecord({ id: "b1", sessionId: "session-b", cwd: "/my/project", timestamp: now - 8000 }), makeEmbedding());
    await insertMemory(dbPath, makeRecord({ id: "a2", sessionId: "session-a", cwd: "/my/project", timestamp: now - 7000 }), makeEmbedding());

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    // Both chapter headings should appear
    assert.ok(ctx!.includes("Topic A - alpha"), "Session A heading should appear");
    assert.ok(ctx!.includes("Topic B - beta"), "Session B heading should appear");
  });
});

// ---------------------------------------------------------------------------
// Files-in-scope filter: only project-relative paths
// ---------------------------------------------------------------------------

describe("buildSessionContext — files in scope filtering", () => {
  beforeEach(() => { dbPath = setup(); });
  afterEach(() => {
    if (tmpDir && fs.existsSync(tmpDir)) fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it("shows files that are under the project cwd", async () => {
    await initDb(dbPath);

    await insertMemory(dbPath, makeRecord({
      id: "m1",
      sessionId: "old",
      cwd: "/my/project",
      filesTouched: ["/my/project/src/billing.ts", "/my/project/tests/billing.test.ts"],
      summary: "Updated billing module",
    }), makeEmbedding());

    const ctx = await buildSessionContext(dbPath, "/my/project", "current");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("billing.ts"), "Project files should appear");
  });

  it("excludes files outside the project cwd (tmp, downloads, home config)", async () => {
    await initDb(dbPath);

    await insertMemory(dbPath, makeRecord({
      id: "m1",
      sessionId: "old",
      cwd: "/my/project",
      filesTouched: [
        "/my/project/src/billing.ts",  // keep
        "/tmp/fetch-abc123.html",       // exclude
        "/var/folders/xyz/cache.json",  // exclude
        `${os.homedir()}/.pi/agent/memory/memory.db`, // exclude
      ],
      summary: "Did some work",
    }), makeEmbedding());

    const ctx = await buildSessionContext(dbPath, "/my/project", "current");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("billing.ts"), "Project file should be present");
    assert.ok(!ctx!.includes("/tmp/"), "Temp files should be excluded");
    assert.ok(!ctx!.includes("/var/"), "Var files should be excluded");
    assert.ok(!ctx!.includes(".pi/agent/memory"), "Home config files should be excluded");
  });

  it("shows no files section when all files are outside the project", async () => {
    await initDb(dbPath);

    await insertMemory(dbPath, makeRecord({
      id: "m1",
      sessionId: "old",
      cwd: "/my/project",
      filesTouched: ["/tmp/fetch-abc123.html", "/tmp/result.json"],
      summary: "Fetched some web content",
    }), makeEmbedding());

    const ctx = await buildSessionContext(dbPath, "/my/project", "current");
    assert.ok(ctx !== null);
    assert.ok(!ctx!.includes("Files in scope"), "Should not show files section when all are external");
  });
});
