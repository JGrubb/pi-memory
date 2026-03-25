import * as path from "node:path";
import { getRecentForCwd, getRecentSessions, getRecentSessionsCrossProject } from "./db.js";
import type { SearchResult, SessionRecord } from "./types.js";

/**
 * Build the context string injected at the start of a new coding session.
 *
 * Structure:
 *   1. Recent sessions in this directory — named sessions as chapter headings,
 *      with their memories listed beneath. Falls back to a date heading for
 *      sessions not yet named.
 *   2. Recent named sessions in other projects — one line per project.
 */
export async function buildSessionContext(
  dbPath: string,
  cwd: string,
  currentSessionId: string | null,
): Promise<string | null> {
  const excludeId = currentSessionId ?? "";

  let memories: SearchResult[];
  let recentSessions: SessionRecord[];
  let crossProjectSessions: SessionRecord[];

  try {
    [memories, recentSessions, crossProjectSessions] = await Promise.all([
      getRecentForCwd(dbPath, cwd, currentSessionId, 25),
      getRecentSessions(dbPath, cwd, excludeId, 5),
      getRecentSessionsCrossProject(dbPath, cwd, 8),
    ]);
  } catch (err) {
    console.error("[memory] Failed to query context:", err);
    return null;
  }

  if (memories.length === 0 && recentSessions.length === 0 && crossProjectSessions.length === 0) {
    return null;
  }

  const parts: string[] = [];

  // -------------------------------------------------------------------------
  // Same-cwd: session chapter headings
  // -------------------------------------------------------------------------
  if (memories.length > 0 || recentSessions.length > 0) {
    parts.push("## Memory: Recent work in this directory");

    // Build a lookup of named sessions by id
    const sessionMap = new Map<string, SessionRecord>();
    for (const s of recentSessions) sessionMap.set(s.id, s);

    // Group memories by session_id
    const bySession = groupMemoriesBySession(memories);

    // Render: named sessions first (in recency order), then any unnamed groups
    const renderedSessionIds = new Set<string>();
    let sessionsRendered = 0;

    // Walk named sessions in order (most recent first) — up to 3
    for (const session of recentSessions) {
      if (sessionsRendered >= 3) break;
      const sessionMemories = bySession.get(session.id) ?? [];
      renderedSessionIds.add(session.id);
      parts.push(renderSessionSection(session.name!, session.timestamp, sessionMemories, cwd));
      sessionsRendered++;
    }

    // Render any memory groups whose session_id wasn't in the named sessions table
    for (const [sessionId, groupMemories] of bySession) {
      if (sessionsRendered >= 3) break;
      if (renderedSessionIds.has(sessionId)) continue;
      // Fall back to date heading
      const heading = formatDate(groupMemories[0].timestamp);
      parts.push(renderSessionSection(heading, groupMemories[0].timestamp, groupMemories, cwd));
      sessionsRendered++;
    }
  }

  // -------------------------------------------------------------------------
  // Cross-project: one named session per project
  // -------------------------------------------------------------------------
  if (crossProjectSessions.length > 0) {
    parts.push("");
    parts.push("## Memory: Recent work in other projects");
    for (const session of crossProjectSessions) {
      const shortPath = shortenPath(session.cwd);
      const date = formatDate(session.timestamp);
      parts.push(`- **${shortPath}** (${date}): ${session.name}`);
    }
  }

  return parts.length > 0 ? parts.join("\n") : null;
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

function renderSessionSection(
  heading: string,
  timestamp: number,
  memories: SearchResult[],
  projectCwd: string,
): string {
  const lines: string[] = [`\n**${heading}**`];

  const sessionFiles = new Set<string>();
  for (const m of memories) {
    const tags = m.topics.length > 0 ? ` [${m.topics.join(", ")}]` : "";
    lines.push(`- ${m.summary}${tags}`);
    for (const f of m.filesTouched) {
      if (isProjectFile(f, projectCwd)) sessionFiles.add(f);
    }
  }

  if (sessionFiles.size > 0) {
    const fileList = Array.from(sessionFiles).slice(0, 10).join(", ");
    lines.push(`  *Files in scope: ${fileList}*`);
  }

  return lines.join("\n");
}

/**
 * Group memories by their session_id. Memories without a session_id are
 * grouped by time proximity (2-hour window) as a fallback.
 * Returns a Map ordered by most-recent session first.
 */
function groupMemoriesBySession(
  memories: SearchResult[],
): Map<string, SearchResult[]> {
  const map = new Map<string, SearchResult[]>();

  for (const m of memories) {
    if (m.sessionId) {
      const group = map.get(m.sessionId) ?? [];
      group.push(m);
      map.set(m.sessionId, group);
    } else {
      // Legacy: no session_id — group by time proximity
      const legacyKey = findOrCreateLegacyGroup(map, m.timestamp);
      map.get(legacyKey)!.push(m);
    }
  }

  return map;
}

/** Find an existing legacy group whose last timestamp is within 2h, or create one. */
function findOrCreateLegacyGroup(
  map: Map<string, SearchResult[]>,
  timestamp: number,
): string {
  const twoHours = 2 * 60 * 60 * 1000;
  for (const [key, group] of map) {
    if (!key.startsWith("__legacy__")) continue;
    const lastTs = group[group.length - 1].timestamp;
    if (Math.abs(timestamp - lastTs) < twoHours) return key;
  }
  const key = `__legacy__${timestamp}`;
  map.set(key, []);
  return key;
}

/**
 * Returns true if the file path is within the project's cwd.
 * Excludes temp files, downloaded content, and home directory config paths.
 */
function isProjectFile(filePath: string, projectCwd: string): boolean {
  const resolved = path.resolve(filePath);
  return resolved.startsWith(projectCwd);
}

// ---------------------------------------------------------------------------
// Format search results for the memory_search tool
// ---------------------------------------------------------------------------

export function formatSearchResults(results: SearchResult[]): string {
  if (results.length === 0) return "No matching memories found.";

  const lines: string[] = [`Found ${results.length} matching memories:\n`];
  for (const r of results) {
    const date = formatDate(r.timestamp);
    const project = shortenPath(r.cwd);
    const similarity = ((1 - r.distance) * 100).toFixed(0);
    const tags = r.topics.length > 0 ? ` [${r.topics.join(", ")}]` : "";
    const label = r.type === "artifact" ? "📌 " : "";
    lines.push(`**${label}${project}** (${date}, ${similarity}% match)${tags}`);
    lines.push(`  ${r.summary}`);
    if (r.type === "artifact" && r.content) {
      lines.push("  ```");
      lines.push(r.content.split("\n").map((l) => `  ${l}`).join("\n"));
      lines.push("  ```");
    }
    if (r.filesTouched.length > 0) {
      lines.push(`  Files: ${r.filesTouched.slice(0, 5).join(", ")}`);
    }
    lines.push("");
  }
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function shortenPath(fullPath: string): string {
  const parts = fullPath.split("/").filter(Boolean);
  if (parts.length <= 2) return fullPath;
  return parts.slice(-2).join("/");
}

function formatDate(timestamp: number): string {
  const d = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) {
    return `today ${d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })}`;
  } else if (diffDays === 1) {
    return "yesterday";
  } else if (diffDays < 7) {
    return `${diffDays} days ago`;
  } else {
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  }
}
