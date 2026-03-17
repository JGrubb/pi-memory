import { getRecentForCwd, getRecentCrossProject } from "./db.js";
import type { SearchResult } from "./types.js";

/**
 * Build the context string injected into the system prompt at the start
 * of a new coding session. Contains:
 *   1. Recent memories from the same working directory (previous sessions)
 *   2. A brief overview of recent work in other projects
 */
export async function buildSessionContext(
  dbPath: string,
  cwd: string,
  currentSessionId: string | null,
): Promise<string | null> {
  let sameCwd: SearchResult[];
  let crossProject: SearchResult[];

  try {
    sameCwd = await getRecentForCwd(dbPath, cwd, currentSessionId, 15);
    crossProject = await getRecentCrossProject(dbPath, cwd, 8);
  } catch (err) {
    console.error("[memory] Failed to query context:", err);
    return null;
  }

  if (sameCwd.length === 0 && crossProject.length === 0) {
    return null;
  }

  const parts: string[] = [];

  if (sameCwd.length > 0) {
    parts.push("## Memory: Recent work in this directory");
    // Group by session to show session boundaries
    const bySession = groupBySession(sameCwd);
    let sessionCount = 0;
    for (const [, memories] of bySession) {
      if (sessionCount >= 3) break; // At most 3 previous sessions
      const sessionDate = formatDate(memories[0].timestamp);
      parts.push(`\n**Session — ${sessionDate}**`);
      for (const m of memories.slice(0, 5)) {
        const tags = m.topics.length > 0 ? ` [${m.topics.join(", ")}]` : "";
        parts.push(`- ${m.summary}${tags}`);
      }
      sessionCount++;
    }
  }

  if (crossProject.length > 0) {
    parts.push("");
    parts.push("## Memory: Recent work in other projects");
    for (const m of crossProject) {
      const shortPath = shortenPath(m.cwd);
      const date = formatDate(m.timestamp);
      const tags = m.topics.length > 0 ? ` [${m.topics.join(", ")}]` : "";
      parts.push(`- **${shortPath}** (${date}): ${m.summary}${tags}`);
    }
  }

  return parts.join("\n");
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
    lines.push(`**${project}** (${date}, ${similarity}% match)${tags}`);
    lines.push(`  ${r.summary}`);
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

function groupBySession(memories: SearchResult[]): Map<string, SearchResult[]> {
  const map = new Map<string, SearchResult[]>();
  // Since we don't have session_id in SearchResult, group by time proximity.
  // Memories within 2 hours of each other are considered same session.
  let currentGroup: SearchResult[] = [];
  let groupKey = 0;

  for (const m of memories) {
    if (currentGroup.length === 0) {
      currentGroup.push(m);
    } else {
      const lastTs = currentGroup[currentGroup.length - 1].timestamp;
      if (Math.abs(m.timestamp - lastTs) < 2 * 60 * 60 * 1000) {
        currentGroup.push(m);
      } else {
        map.set(String(groupKey++), currentGroup);
        currentGroup = [m];
      }
    }
  }
  if (currentGroup.length > 0) {
    map.set(String(groupKey), currentGroup);
  }
  return map;
}

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
