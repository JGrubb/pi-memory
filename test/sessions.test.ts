import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

import {
  initDb,
  upsertSession,
  updateSessionName,
  getRecentSessions,
  findSimilarSessions,
  backfillSessionsFromJSONL,
  closeAll,
} from "../db.js";
import type { SessionRecord } from "../types.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let tmpDir: string;
let dbPath: string;

function makeDbPath(): string {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-sessions-test-"));
  return path.join(tmpDir, "test.db");
}

/** Generate a deterministic unit-length Float32Array. */
function makeEmbedding(dims: number = 768, seed: number = 42): Float32Array {
  const vec = new Float32Array(dims);
  let x = seed;
  for (let i = 0; i < dims; i++) {
    x = ((x * 1103515245 + 12345) & 0x7fffffff) >>> 0;
    vec[i] = (x / 0x7fffffff) * 2 - 1;
  }
  let norm = 0;
  for (let i = 0; i < dims; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  for (let i = 0; i < dims; i++) vec[i] /= norm;
  return vec;
}

/** Perturb a unit vector by adding small noise then re-normalizing — produces a "close" vector. */
function perturbEmbedding(base: Float32Array, noise: number = 0.1, seed: number = 7): Float32Array {
  let x = seed;
  const vec = new Float32Array(base.length);
  for (let i = 0; i < base.length; i++) {
    x = ((x * 1103515245 + 12345) & 0x7fffffff) >>> 0;
    vec[i] = base[i] + noise * ((x / 0x7fffffff) * 2 - 1);
  }
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  return vec;
}

function makeSession(overrides: Partial<SessionRecord> = {}): SessionRecord {
  return {
    id: `sess-${Math.random().toString(36).slice(2, 8)}`,
    cwd: "/Users/test/project-a",
    sessionFile: "/Users/test/.pi/sessions/project-a/123_abc.jsonl",
    name: null,
    mainTopic: null,
    subTopic: null,
    timestamp: Date.now(),
    namedAt: null,
    ...overrides,
  };
}

/**
 * Write a minimal JSONL session file with a session header, two messages,
 * and optionally a session_info entry with a name.
 */
function writeSessionFile(
  dir: string,
  filename: string,
  opts: {
    cwd: string;
    sessionId: string;
    firstPrompt: string;
    name?: string;
  },
): string {
  const filePath = path.join(dir, filename);
  const lines: object[] = [
    {
      type: "session",
      version: 3,
      id: opts.sessionId,
      timestamp: new Date().toISOString(),
      cwd: opts.cwd,
    },
    {
      type: "message",
      id: "aaa00001",
      parentId: null,
      timestamp: new Date().toISOString(),
      message: {
        role: "user",
        content: opts.firstPrompt,
        timestamp: Date.now(),
      },
    },
    {
      type: "message",
      id: "aaa00002",
      parentId: "aaa00001",
      timestamp: new Date().toISOString(),
      message: {
        role: "assistant",
        content: [{ type: "text", text: "Sure, I can help with that." }],
        timestamp: Date.now(),
      },
    },
  ];

  if (opts.name) {
    lines.push({
      type: "session_info",
      id: "aaa00003",
      parentId: "aaa00002",
      timestamp: new Date().toISOString(),
      name: opts.name,
    });
  }

  fs.writeFileSync(filePath, lines.map((l) => JSON.stringify(l)).join("\n") + "\n");
  return filePath;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("sessions table — upsertSession", () => {
  beforeEach(() => { dbPath = makeDbPath(); });
  afterEach(() => {
    closeAll();
    if (tmpDir && fs.existsSync(tmpDir)) fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it("inserts a new session row (visible after naming)", async () => {
    await initDb(dbPath);
    const s = makeSession({ id: "sess-1", cwd: "/project" });
    await upsertSession(dbPath, s);
    // Unnamed sessions are registered but not yet visible in getRecentSessions
    // (which only returns named sessions for the chapter heading display).
    // Name it, then verify it appears.
    await updateSessionName(dbPath, "sess-1", "Topic", "sub", "Topic - sub", makeEmbedding());

    const recent = await getRecentSessions(dbPath, "/project", "other-id", 10);
    assert.equal(recent.length, 1);
    assert.equal(recent[0].id, "sess-1");
  });

  it("is idempotent — calling twice with the same id does not duplicate", async () => {
    await initDb(dbPath);
    const s = makeSession({ id: "sess-1", cwd: "/project" });
    await upsertSession(dbPath, s);
    await upsertSession(dbPath, s);
    await updateSessionName(dbPath, "sess-1", "Topic", "sub", "Topic - sub", makeEmbedding());

    const recent = await getRecentSessions(dbPath, "/project", "other-id", 10);
    assert.equal(recent.length, 1);
  });

  it("does not overwrite an existing name when called again", async () => {
    await initDb(dbPath);
    const s = makeSession({ id: "sess-1", cwd: "/project" });
    await upsertSession(dbPath, s);
    await updateSessionName(dbPath, "sess-1", "GCP Billing", "cost export fix", "GCP Billing - cost export fix", makeEmbedding());

    // Upsert again (as happens on session_switch back to this session)
    await upsertSession(dbPath, s);

    const recent = await getRecentSessions(dbPath, "/project", "other-id", 10);
    assert.equal(recent.length, 1);
    assert.equal(recent[0].name, "GCP Billing - cost export fix", "Name should be preserved");
  });
});

describe("sessions table — updateSessionName", () => {
  beforeEach(() => { dbPath = makeDbPath(); });
  afterEach(() => {
    closeAll();
    if (tmpDir && fs.existsSync(tmpDir)) fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it("sets name, mainTopic, subTopic, and namedAt", async () => {
    await initDb(dbPath);
    await upsertSession(dbPath, makeSession({ id: "sess-1", cwd: "/project" }));
    await updateSessionName(dbPath, "sess-1", "GCP Billing", "Pub/Sub backlog investigation", "GCP Billing - Pub/Sub backlog investigation", makeEmbedding());

    const recent = await getRecentSessions(dbPath, "/project", "other-id", 10);
    assert.equal(recent[0].name, "GCP Billing - Pub/Sub backlog investigation");
    assert.equal(recent[0].mainTopic, "GCP Billing");
    assert.equal(recent[0].subTopic, "Pub/Sub backlog investigation");
    assert.ok(recent[0].namedAt !== null, "namedAt should be set");
  });

  it("stores an embedding that can be used in findSimilarSessions", async () => {
    await initDb(dbPath);
    const embedding = makeEmbedding(768, 10);
    await upsertSession(dbPath, makeSession({ id: "sess-1", cwd: "/project" }));
    await updateSessionName(dbPath, "sess-1", "GCP Billing", "cost export", "GCP Billing - cost export", embedding);

    // A nearly identical embedding should find it
    const results = await findSimilarSessions(dbPath, "/project", "sess-2", embedding, 0.5);
    assert.equal(results.length, 1);
    assert.equal(results[0].id, "sess-1");
  });
});

describe("getRecentSessions", () => {
  beforeEach(() => { dbPath = makeDbPath(); });
  afterEach(() => {
    closeAll();
    if (tmpDir && fs.existsSync(tmpDir)) fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it("returns sessions for the matching cwd in reverse chronological order", async () => {
    await initDb(dbPath);
    const now = Date.now();

    await upsertSession(dbPath, makeSession({ id: "old", cwd: "/project", timestamp: now - 10000 }));
    await upsertSession(dbPath, makeSession({ id: "new", cwd: "/project", timestamp: now }));
    await upsertSession(dbPath, makeSession({ id: "other", cwd: "/other-project", timestamp: now }));

    await updateSessionName(dbPath, "old", "Topic", "old work", "Topic - old work", makeEmbedding(768, 1));
    await updateSessionName(dbPath, "new", "Topic", "new work", "Topic - new work", makeEmbedding(768, 2));
    await updateSessionName(dbPath, "other", "Topic", "other work", "Topic - other work", makeEmbedding(768, 3));

    const results = await getRecentSessions(dbPath, "/project", "excluded-id", 10);
    assert.equal(results.length, 2);
    assert.equal(results[0].id, "new", "Most recent first");
    assert.equal(results[1].id, "old");
  });

  it("excludes the current session id", async () => {
    await initDb(dbPath);
    await upsertSession(dbPath, makeSession({ id: "current", cwd: "/project" }));
    await upsertSession(dbPath, makeSession({ id: "previous", cwd: "/project", timestamp: Date.now() - 1000 }));
    await updateSessionName(dbPath, "previous", "Topic", "sub", "Topic - sub", makeEmbedding());

    const results = await getRecentSessions(dbPath, "/project", "current", 10);
    assert.equal(results.length, 1);
    assert.equal(results[0].id, "previous");
  });

  it("only returns sessions that have been named", async () => {
    await initDb(dbPath);
    await upsertSession(dbPath, makeSession({ id: "named", cwd: "/project", timestamp: Date.now() - 1000 }));
    await upsertSession(dbPath, makeSession({ id: "unnamed", cwd: "/project", timestamp: Date.now() - 2000 }));
    await updateSessionName(dbPath, "named", "Topic", "sub", "Topic - sub", makeEmbedding());

    const results = await getRecentSessions(dbPath, "/project", "current", 10);
    assert.equal(results.length, 1);
    assert.equal(results[0].id, "named");
  });

  it("respects the limit", async () => {
    await initDb(dbPath);
    for (let i = 0; i < 6; i++) {
      const s = makeSession({ id: `sess-${i}`, cwd: "/project", timestamp: Date.now() + i });
      await upsertSession(dbPath, s);
      await updateSessionName(dbPath, s.id, "Topic", `sub ${i}`, `Topic - sub ${i}`, makeEmbedding(768, i));
    }

    const results = await getRecentSessions(dbPath, "/project", "other", 4);
    assert.equal(results.length, 4);
  });

  it("returns empty when no other sessions exist for this cwd", async () => {
    await initDb(dbPath);
    await upsertSession(dbPath, makeSession({ id: "current", cwd: "/project" }));

    const results = await getRecentSessions(dbPath, "/project", "current", 10);
    assert.equal(results.length, 0);
  });
});

describe("findSimilarSessions", () => {
  beforeEach(() => { dbPath = makeDbPath(); });
  afterEach(() => {
    closeAll();
    if (tmpDir && fs.existsSync(tmpDir)) fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it("returns sessions above the similarity threshold", async () => {
    await initDb(dbPath);
    const embedding = makeEmbedding(768, 42);

    await upsertSession(dbPath, makeSession({ id: "match", cwd: "/project", timestamp: Date.now() - 1000 }));
    await updateSessionName(dbPath, "match", "GCP Billing", "cost export", "GCP Billing - cost export", embedding);

    // Exact same embedding should be above any reasonable threshold
    const results = await findSimilarSessions(dbPath, "/project", "current", embedding, 0.9);
    assert.equal(results.length, 1);
    assert.equal(results[0].id, "match");
  });

  it("excludes sessions below the similarity threshold", async () => {
    await initDb(dbPath);
    const queryEmbedding = makeEmbedding(768, 1);
    const veryDifferentEmbedding = makeEmbedding(768, 999);

    await upsertSession(dbPath, makeSession({ id: "unrelated", cwd: "/project", timestamp: Date.now() - 1000 }));
    await updateSessionName(dbPath, "unrelated", "Metabase", "dashboard upgrade", "Metabase - dashboard upgrade", veryDifferentEmbedding);

    // High threshold — should exclude the unrelated session
    const results = await findSimilarSessions(dbPath, "/project", "current", queryEmbedding, 0.9);
    assert.equal(results.length, 0);
  });

  it("only searches within the same cwd", async () => {
    await initDb(dbPath);
    const embedding = makeEmbedding(768, 42);

    await upsertSession(dbPath, makeSession({ id: "other-project", cwd: "/other-project", timestamp: Date.now() - 1000 }));
    await updateSessionName(dbPath, "other-project", "GCP Billing", "cost export", "GCP Billing - cost export", embedding);

    const results = await findSimilarSessions(dbPath, "/project", "current", embedding, 0.9);
    assert.equal(results.length, 0, "Should not match sessions in other cwds");
  });

  it("excludes the current session", async () => {
    await initDb(dbPath);
    const embedding = makeEmbedding(768, 42);

    await upsertSession(dbPath, makeSession({ id: "current", cwd: "/project" }));
    // Even if current is named (e.g. from a previous run), it should be excluded
    await updateSessionName(dbPath, "current", "GCP Billing", "cost export", "GCP Billing - cost export", embedding);

    const results = await findSimilarSessions(dbPath, "/project", "current", embedding, 0.5);
    assert.equal(results.length, 0);
  });

  it("returns results ordered by similarity descending", async () => {
    await initDb(dbPath);
    const targetEmbedding = makeEmbedding(768, 42);

    // "close": target + small noise → high dot product with target
    const closeEmbedding = perturbEmbedding(targetEmbedding, 0.1, 7);

    // "far": independently random → near-zero dot product with target
    const farEmbedding = makeEmbedding(768, 999);

    await upsertSession(dbPath, makeSession({ id: "close", cwd: "/project", timestamp: Date.now() - 1000 }));
    await upsertSession(dbPath, makeSession({ id: "far", cwd: "/project", timestamp: Date.now() - 2000 }));
    await updateSessionName(dbPath, "close", "Topic", "close sub", "Topic - close sub", closeEmbedding);
    await updateSessionName(dbPath, "far", "Topic", "far sub", "Topic - far sub", farEmbedding);

    const results = await findSimilarSessions(dbPath, "/project", "current", targetEmbedding, -1.0);
    assert.ok(results.length >= 1);
    // Most similar should be first
    assert.equal(results[0].id, "close");
  });

  it("returns the count of existing sessions with the same mainTopic", async () => {
    await initDb(dbPath);
    const embedding = makeEmbedding(768, 42);

    // Three sessions with the same main topic
    for (let i = 1; i <= 3; i++) {
      const s = makeSession({ id: `billing-${i}`, cwd: "/project", timestamp: Date.now() - i * 1000 });
      await upsertSession(dbPath, s);
      await updateSessionName(dbPath, s.id, "GCP Billing", `sub ${i}`, `GCP Billing - sub ${i}`, embedding);
    }

    const results = await findSimilarSessions(dbPath, "/project", "current", embedding, 0.5);
    assert.equal(results.length, 3);
    // The count of sessions with mainTopic = "GCP Billing" should be 3
    const gcpBillingCount = results.filter(r => r.mainTopic === "GCP Billing").length;
    assert.equal(gcpBillingCount, 3);
  });
});

describe("backfillSessionsFromJSONL", () => {
  beforeEach(() => { dbPath = makeDbPath(); });
  afterEach(() => {
    closeAll();
    if (tmpDir && fs.existsSync(tmpDir)) fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it("imports named sessions from JSONL files into the sessions table", async () => {
    await initDb(dbPath);

    // Create a fake pi sessions directory structure
    const sessionsDir = path.join(tmpDir, "sessions");
    const projectDir = path.join(sessionsDir, "--Users-test-project--");
    fs.mkdirSync(projectDir, { recursive: true });

    writeSessionFile(projectDir, "session1.jsonl", {
      sessionId: "uuid-1",
      cwd: "/Users/test/project",
      firstPrompt: "Help me with billing",
      name: "GCP Billing - cost export fix",
    });

    await backfillSessionsFromJSONL(dbPath, sessionsDir);

    const recent = await getRecentSessions(dbPath, "/Users/test/project", "other", 10);
    assert.equal(recent.length, 1);
    assert.equal(recent[0].id, "uuid-1");
    assert.equal(recent[0].name, "GCP Billing - cost export fix");
  });

  it("skips unnamed sessions (no session_info entry)", async () => {
    await initDb(dbPath);

    const sessionsDir = path.join(tmpDir, "sessions");
    const projectDir = path.join(sessionsDir, "--Users-test-project--");
    fs.mkdirSync(projectDir, { recursive: true });

    writeSessionFile(projectDir, "unnamed.jsonl", {
      sessionId: "uuid-unnamed",
      cwd: "/Users/test/project",
      firstPrompt: "Quick question",
      // no name
    });

    await backfillSessionsFromJSONL(dbPath, sessionsDir);

    const recent = await getRecentSessions(dbPath, "/Users/test/project", "other", 10);
    assert.equal(recent.length, 0, "Unnamed sessions should not appear");
  });

  it("skips sessions already in the table", async () => {
    await initDb(dbPath);

    const sessionsDir = path.join(tmpDir, "sessions");
    const projectDir = path.join(sessionsDir, "--Users-test-project--");
    fs.mkdirSync(projectDir, { recursive: true });

    const filePath = writeSessionFile(projectDir, "session1.jsonl", {
      sessionId: "uuid-1",
      cwd: "/Users/test/project",
      firstPrompt: "Help me",
      name: "GCP Billing - cost export fix",
    });

    // Pre-insert the session record
    await upsertSession(dbPath, makeSession({ id: "uuid-1", cwd: "/Users/test/project", sessionFile: filePath }));

    await backfillSessionsFromJSONL(dbPath, sessionsDir);

    // Should still be exactly one row
    const recent = await getRecentSessions(dbPath, "/Users/test/project", "other", 10);
    assert.equal(recent.length, 1);
  });

  it("handles multiple project directories", async () => {
    await initDb(dbPath);

    const sessionsDir = path.join(tmpDir, "sessions");
    fs.mkdirSync(path.join(sessionsDir, "--Users-test-project-a--"), { recursive: true });
    fs.mkdirSync(path.join(sessionsDir, "--Users-test-project-b--"), { recursive: true });

    writeSessionFile(path.join(sessionsDir, "--Users-test-project-a--"), "s1.jsonl", {
      sessionId: "uuid-a",
      cwd: "/Users/test/project-a",
      firstPrompt: "Help with A",
      name: "Project A - some work",
    });
    writeSessionFile(path.join(sessionsDir, "--Users-test-project-b--"), "s2.jsonl", {
      sessionId: "uuid-b",
      cwd: "/Users/test/project-b",
      firstPrompt: "Help with B",
      name: "Project B - other work",
    });

    await backfillSessionsFromJSONL(dbPath, sessionsDir);

    const resultsA = await getRecentSessions(dbPath, "/Users/test/project-a", "other", 10);
    const resultsB = await getRecentSessions(dbPath, "/Users/test/project-b", "other", 10);
    assert.equal(resultsA.length, 1);
    assert.equal(resultsB.length, 1);
    assert.equal(resultsA[0].id, "uuid-a");
    assert.equal(resultsB[0].id, "uuid-b");
  });

  it("uses the last session_info entry as the canonical name", async () => {
    await initDb(dbPath);

    const sessionsDir = path.join(tmpDir, "sessions");
    const projectDir = path.join(sessionsDir, "--Users-test-project--");
    fs.mkdirSync(projectDir, { recursive: true });

    // Write a session with two session_info entries (e.g. user renamed it)
    const filePath = path.join(projectDir, "renamed.jsonl");
    const lines = [
      { type: "session", version: 3, id: "uuid-renamed", timestamp: new Date().toISOString(), cwd: "/Users/test/project" },
      { type: "message", id: "a1", parentId: null, timestamp: new Date().toISOString(), message: { role: "user", content: "start", timestamp: Date.now() } },
      { type: "session_info", id: "b1", parentId: "a1", timestamp: new Date().toISOString(), name: "First name" },
      { type: "message", id: "a2", parentId: "b1", timestamp: new Date().toISOString(), message: { role: "user", content: "more", timestamp: Date.now() } },
      { type: "session_info", id: "b2", parentId: "a2", timestamp: new Date().toISOString(), name: "Final name" },
    ];
    fs.writeFileSync(filePath, lines.map((l) => JSON.stringify(l)).join("\n") + "\n");

    await backfillSessionsFromJSONL(dbPath, sessionsDir);

    const recent = await getRecentSessions(dbPath, "/Users/test/project", "other", 10);
    assert.equal(recent.length, 1);
    assert.equal(recent[0].name, "Final name", "Should use the last session_info entry");
  });

  it("is idempotent — running twice does not duplicate rows", async () => {
    await initDb(dbPath);

    const sessionsDir = path.join(tmpDir, "sessions");
    const projectDir = path.join(sessionsDir, "--project--");
    fs.mkdirSync(projectDir, { recursive: true });

    writeSessionFile(projectDir, "s.jsonl", {
      sessionId: "uuid-1",
      cwd: "/project",
      firstPrompt: "Do something",
      name: "Topic - sub",
    });

    await backfillSessionsFromJSONL(dbPath, sessionsDir);
    await backfillSessionsFromJSONL(dbPath, sessionsDir);

    const recent = await getRecentSessions(dbPath, "/project", "other", 10);
    assert.equal(recent.length, 1);
  });
});
