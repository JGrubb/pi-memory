export type MemoryStatus = "complete" | "pending" | "pending_embed";
export type MemoryType = "memory" | "artifact";

export interface MemoryRecord {
  id: string;
  sessionId: string;
  timestamp: number;
  cwd: string;
  summary: string;
  topics: string[];
  filesTouched: string[];
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
  timestamp: number;
  namedAt: number | null;
}

export interface ExtractedContent {
  userPrompt: string;
  assistantResponse: string;
  filesTouched: string[];
  toolsUsed: string[];
}

export interface Config {
  gcpProject: string;
  region: string;
  embeddingModel: string;
  summarizeModel: string;
  summarizeProvider: "vertex" | "anthropic";
  embeddingDims: number;
  dbPath: string;
}
