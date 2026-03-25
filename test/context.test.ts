import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

import { initDb, insertMemory, upsertSession, updateSessionName, updateSessionDescription } from "../db.js";
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

  it("returns null when there are no sessions", async () => {
    await initDb(dbPath);
    const ctx = await buildSessionContext(dbPath, "/some/project", null);
    assert.equal(ctx, null);
  });

  it("shows named session with description and files", async () => {
    await initDb(dbPath);

    await upsertSession(dbPath, makeSessionRecord({ id: "s1", cwd: "/my/project" }));
    await updateSessionName(dbPath, "s1", "Auth", "middleware fix", "Auth - middleware fix", makeSessionEmbedding());
    await updateSessionDescription(dbPath, "s1", "Fixed the auth middleware to handle token expiry correctly.", ["/my/project/src/auth.ts"]);

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("Recent work in this directory"), "Should have same-cwd section");
    assert.ok(ctx!.includes("Auth - middleware fix"), "Should show session name");
    assert.ok(ctx!.includes("Fixed the auth middleware"), "Should show description");
    assert.ok(ctx!.includes("auth.ts"), "Should show files");
  });

  it("shows session name without description when description not yet generated", async () => {
    await initDb(dbPath);

    await upsertSession(dbPath, makeSessionRecord({ id: "s1", cwd: "/my/project" }));
    await updateSessionName(dbPath, "s1", "Auth", "middleware fix", "Auth - middleware fix", makeSessionEmbedding());

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("Auth - middleware fix"), "Should show session name even without description");
  });

  it("excludes current session", async () => {
    await initDb(dbPath);

    await upsertSession(dbPath, makeSessionRecord({ id: "current", cwd: "/my/project" }));
    await updateSessionName(dbPath, "current", "Topic", "current work", "Topic - current work", makeSessionEmbedding());

    await upsertSession(dbPath, makeSessionRecord({ id: "previous", cwd: "/my/project" }));
    await updateSessionName(dbPath, "previous", "Topic", "previous work", "Topic - previous work", makeSessionEmbedding());

    const ctx = await buildSessionContext(dbPath, "/my/project", "current");
    assert.ok(ctx !== null);
    assert.ok(!ctx!.includes("current work"), "Should NOT include current session");
    assert.ok(ctx!.includes("previous work"), "Should include previous session");
  });

  it("includes cross-project named sessions", async () => {
    await initDb(dbPath);

    await upsertSession(dbPath, makeSessionRecord({ id: "s1", cwd: "/other/repo" }));
    await updateSessionName(dbPath, "s1", "Terraform", "module deployment", "Terraform - module deployment", makeSessionEmbedding());

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("Recent work in other projects"), "Should have cross-project section");
    assert.ok(ctx!.includes("other/repo"), "Should show shortened path");
    assert.ok(ctx!.includes("Terraform - module deployment"), "Should show session name");
  });

  it("handles both same-cwd and cross-project in one context", async () => {
    await initDb(dbPath);

    await upsertSession(dbPath, makeSessionRecord({ id: "same", cwd: "/my/project" }));
    await updateSessionName(dbPath, "same", "DBT", "model updates", "DBT - model updates", makeSessionEmbedding());
    await updateSessionDescription(dbPath, "same", "Updated dbt models for billing pipeline.", []);

    await upsertSession(dbPath, makeSessionRecord({ id: "cross", cwd: "/other/project" }));
    await updateSessionName(dbPath, "cross", "CI Pipeline", "fix flaky tests", "CI Pipeline - fix flaky tests", makeSessionEmbedding());

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("Recent work in this directory"));
    assert.ok(ctx!.includes("Recent work in other projects"));
    assert.ok(ctx!.includes("DBT - model updates"));
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
    description: null,
    filesTouched: [],
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

  it("uses session name as chapter heading with description body", async () => {
    await initDb(dbPath);

    const s = makeSessionRecord({ id: "old-session", cwd: "/my/project", timestamp: Date.now() - 86400000 });
    await upsertSession(dbPath, s);
    await updateSessionName(dbPath, "old-session", "GCP Billing", "cost export fix", "GCP Billing - cost export fix", makeSessionEmbedding());
    await updateSessionDescription(dbPath, "old-session",
      "Rewrote the export query and added partition pruning to fix billing cost export.",
      ["/my/project/models/billing.sql"],
    );

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("GCP Billing - cost export fix"), "Session name should be the heading");
    assert.ok(ctx!.includes("Rewrote the export query"), "Description should appear");
    assert.ok(ctx!.includes("billing.sql"), "Files should appear");
  });

  it("falls back gracefully for unnamed sessions (not shown)", async () => {
    await initDb(dbPath);

    // Unnamed session — should not appear since we only show named sessions now
    await upsertSession(dbPath, makeSessionRecord({ id: "old-session", cwd: "/my/project" }));

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.equal(ctx, null, "Unnamed sessions without cross-project data should produce null");
  });

  it("shows session names for cross-project recent work", async () => {
    await initDb(dbPath);

    const s = makeSessionRecord({ id: "other-session", cwd: "/other/project", timestamp: Date.now() - 3600000 });
    await upsertSession(dbPath, s);
    await updateSessionName(dbPath, "other-session", "Metabase", "Cloud Run upgrade", "Metabase - Cloud Run upgrade", makeSessionEmbedding());

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("Metabase - Cloud Run upgrade"), "Cross-project entry should show session name");
  });

  it("shows multiple named sessions in recency order", async () => {
    await initDb(dbPath);
    const now = Date.now();

    const sA = makeSessionRecord({ id: "session-a", cwd: "/my/project", timestamp: now - 10000 });
    const sB = makeSessionRecord({ id: "session-b", cwd: "/my/project", timestamp: now - 5000 });
    await upsertSession(dbPath, sA);
    await upsertSession(dbPath, sB);
    await updateSessionName(dbPath, "session-a", "Topic A", "alpha", "Topic A - alpha", makeSessionEmbedding());
    await updateSessionName(dbPath, "session-b", "Topic B", "beta", "Topic B - beta", makeSessionEmbedding());

    const ctx = await buildSessionContext(dbPath, "/my/project", "current-session");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("Topic A - alpha"), "Session A heading should appear");
    assert.ok(ctx!.includes("Topic B - beta"), "Session B heading should appear");
    // B is more recent — should appear before A
    assert.ok(ctx!.indexOf("Topic B") < ctx!.indexOf("Topic A"), "More recent session should appear first");
  });
});

// ---------------------------------------------------------------------------
// Files in context: stored on session via updateSessionDescription
// ---------------------------------------------------------------------------

describe("buildSessionContext — files in scope", () => {
  beforeEach(() => { dbPath = setup(); });
  afterEach(() => {
    if (tmpDir && fs.existsSync(tmpDir)) fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it("shows files stored on the session", async () => {
    await initDb(dbPath);

    await upsertSession(dbPath, makeSessionRecord({ id: "s1", cwd: "/my/project" }));
    await updateSessionName(dbPath, "s1", "Billing", "module update", "Billing - module update", makeSessionEmbedding());
    await updateSessionDescription(dbPath, "s1", "Updated the billing module.", [
      "/my/project/src/billing.ts",
      "/my/project/tests/billing.test.ts",
    ]);

    const ctx = await buildSessionContext(dbPath, "/my/project", "current");
    assert.ok(ctx !== null);
    assert.ok(ctx!.includes("billing.ts"), "Project files should appear");
  });

  it("shows no files line when session has no files", async () => {
    await initDb(dbPath);

    await upsertSession(dbPath, makeSessionRecord({ id: "s1", cwd: "/my/project" }));
    await updateSessionName(dbPath, "s1", "Research", "exploration", "Research - exploration", makeSessionEmbedding());
    await updateSessionDescription(dbPath, "s1", "Did some exploratory research.", []);

    const ctx = await buildSessionContext(dbPath, "/my/project", "current");
    assert.ok(ctx !== null);
    assert.ok(!ctx!.includes("Files:"), "Should not show files line when empty");
  });
});
