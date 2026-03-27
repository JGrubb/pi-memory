export type MemoryStatus = "complete" | "pending" | "pending_embed";
export type MemoryType = "memory" | "artifact";

export type ResourceType = "url" | "confluence" | "jira" | "metabase" | "grafana" | "bigquery" | "other";

export interface Resource {
  type: ResourceType;
  uri: string;    // canonical identifier — URL, issue key, card ID, table name, etc.
  label?: string; // human-readable title, e.g. "INFRA-1234: Fix provisioner"
}

export interface MemoryRecord {
  id: string;
  sessionId: string;
  timestamp: number;
  cwd: string;
  summary: string;
  topics: string[];
  filesTouched: string[];
  resources: Resource[];
  toolsUsed: string[];
  userPrompt: string;
  responseSnippet: string;
  status: MemoryStatus;
  rawText: string;
  type: MemoryType;
  content: string | null; // verbatim content for artifacts, null for memories
}

export interface SearchResult {
  id: string;
  sessionId: string | null;
  summary: string;
  cwd: string;
  timestamp: number;
  topics: string[];
  filesTouched: string[];
  resources: Resource[];
  userPrompt: string;
  distance: number;
  type: MemoryType;
  content: string | null;
}

export interface SessionRecord {
  id: string;
  cwd: string;
  sessionFile: string | null;
  name: string | null;
  mainTopic: string | null;
  subTopic: string | null;
  description: string | null;
  filesTouched: string[];
  resources: Resource[];
  timestamp: number;
  namedAt: number | null;
}

export interface ExtractedContent {
  userPrompt: string;
  assistantResponse: string;
  filesTouched: string[];
  resources: Resource[];
  toolsUsed: string[];
}

export interface Config {
  // Shared Vertex config (used when either provider targets Vertex)
  gcpProject: string;
  region: string;

  // Embedding
  embedProvider: "vertex" | "ollama";
  embedModel: string;
  embedDims: number;
  ollamaUrl: string;

  // Summarization
  summarizeProvider: "vertex-anthropic" | "vertex-google" | "anthropic";
  summarizeModel: string;

  dbPath: string;
}
