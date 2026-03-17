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
}

export interface SearchResult {
  id: string;
  summary: string;
  cwd: string;
  timestamp: number;
  topics: string[];
  filesTouched: string[];
  userPrompt: string;
  distance: number;
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
  haikuModel: string;
  embeddingDims: number;
  dbPath: string;
}
