/**
 * Standalone backfill script — names existing unnamed sessions using the LLM.
 * Run with: node --import tsx/esm scripts/run-backfill.ts
 */
import * as os from "node:os";
import * as path from "node:path";

import { initDb, upsertSession, updateSessionName, findSimilarSessions, getBackfillCandidates, appendSessionInfoToJSONL } from "../db.js";
import { embedText, nameSession } from "../vertex.js";
import type { Config } from "../types.js";

const CONFIG: Config = {
  gcpProject: process.env.GOOGLE_CLOUD_PROJECT ?? "",
  region: process.env.PI_MEMORY_REGION || "global",
  embeddingModel: process.env.PI_MEMORY_EMBED_MODEL || "gemini-embedding-001",
  summarizeModel: process.env.PI_MEMORY_SUMMARIZE_MODEL || "claude-haiku-4-5@20251001",
  summarizeProvider: (process.env.PI_MEMORY_SUMMARIZE_PROVIDER === "anthropic" ? "anthropic" : "vertex") as "vertex" | "anthropic",
  embeddingDims: Number(process.env.PI_MEMORY_EMBED_DIMS) || 768,
  dbPath: process.env.PI_MEMORY_DB_PATH || path.join(os.homedir(), ".pi", "agent", "memory", "memory.db"),
};

if (!CONFIG.gcpProject) {
  console.error("GOOGLE_CLOUD_PROJECT env var not set");
  process.exit(1);
}

const sessionsDir = path.join(os.homedir(), ".pi", "agent", "sessions");

await initDb(CONFIG.dbPath);

const candidates = await getBackfillCandidates(CONFIG.dbPath, sessionsDir, 5);

if (candidates.length === 0) {
  console.log("No sessions to backfill.");
  process.exit(0);
}

console.log(`Found ${candidates.length} session(s) to name:\n`);
for (const c of candidates) {
  console.log(`  ${c.sessionId} (${c.userTurnCount} turns) — ${c.cwd}`);
}
console.log();

for (const candidate of candidates) {
  process.stdout.write(`Naming ${candidate.sessionId}... `);
  try {
    await upsertSession(CONFIG.dbPath, {
      id: candidate.sessionId,
      cwd: candidate.cwd,
      sessionFile: candidate.filePath,
      name: null,
      mainTopic: null,
      subTopic: null,
      timestamp: candidate.timestamp,
      namedAt: null,
    });

    const { mainTopic, subTopic } = await nameSession(candidate.conversationText, CONFIG);
    if (!mainTopic) { console.log("skipped (no topic extracted)"); continue; }

    const embedding = await embedText(mainTopic, CONFIG, "RETRIEVAL_DOCUMENT");
    const similar = await findSimilarSessions(CONFIG.dbPath, candidate.cwd, candidate.sessionId, embedding, 0.85);
    const sameTopicCount = similar.filter(
      (s) => s.mainTopic?.toLowerCase() === mainTopic.toLowerCase(),
    ).length;

    const name = sameTopicCount > 0
      ? `${mainTopic} - ${subTopic} (cont. ${sameTopicCount + 1})`
      : subTopic ? `${mainTopic} - ${subTopic}` : mainTopic;

    await updateSessionName(CONFIG.dbPath, candidate.sessionId, mainTopic, subTopic, name, embedding);
    await appendSessionInfoToJSONL(candidate.filePath, name);

    console.log(`"${name}"`);
  } catch (err) {
    console.log(`ERROR: ${err}`);
  }
}

console.log("\nDone.");
