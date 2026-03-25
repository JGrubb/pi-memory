import { getRecentSessions, getRecentSessionsCrossProject } from "./db.js";
import type { SearchResult, SessionRecord } from "./types.js";

/**
 * Build the context string injected at the start of a new coding session.
 *
 * Structure:
 *   1. Recent sessions in this directory — one entry per session using the
 *      LLM-generated description (prose summary) + files in scope.
 *      Falls back to memory bullets for sessions without a description yet.
 *   2. Recent named sessions in other projects — one line per project.
 */
export async function buildSessionContext(
  dbPath: string,
  cwd: string,
  currentSessionId: string | null,
): Promise<string | null> {
  const excludeId = currentSessionId ?? "";

  let recentSessions: SessionRecord[];
  let crossProjectSessions: SessionRecord[];

  try {
    [recentSessions, crossProjectSessions] = await Promise.all([
      getRecentSessions(dbPath, cwd, excludeId, 5),
      getRecentSessionsCrossProject(dbPath, cwd, 8),
    ]);
  } catch (err) {
    console.error("[memory] Failed to query context:", err);
    return null;
  }

  if (recentSessions.length === 0 && crossProjectSessions.length === 0) {
    return null;
  }

  const parts: string[] = [];

  // -------------------------------------------------------------------------
  // Same-cwd: one entry per recent session
  // -------------------------------------------------------------------------
  if (recentSessions.length > 0) {
    parts.push("## Memory: Recent work in this directory");

    for (const session of recentSessions.slice(0, 5)) {
      const date = formatDate(session.timestamp);
      const heading = session.name ?? date;
      parts.push(`\n**${heading}** (${date})`);

      if (session.description) {
        parts.push(session.description);
      }

      if (session.filesTouched.length > 0) {
        const files = session.filesTouched.slice(0, 8).join(", ");
        parts.push(`*Files: ${files}*`);
      }
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
