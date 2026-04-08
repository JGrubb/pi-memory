import { execSync } from "node:child_process";
import type { Config } from "./types.js";

// ---------------------------------------------------------------------------
// Auth — GCP (Vertex AI) or Anthropic API
// ---------------------------------------------------------------------------

let cachedToken: { token: string; expiresAt: number } | null = null;

function isReauthError(error: any): boolean {
  const stderr = error?.stderr?.toString?.() || error?.message || "";
  return stderr.includes("Reauthentication") || stderr.includes("invalid_rapt");
}

function refreshADC(): void {
  console.log("[memory] ADC expired — launching gcloud auth application-default login...");
  execSync("gcloud auth application-default login --quiet", {
    encoding: "utf-8",
    timeout: 120_000, // 2 minutes for user to complete browser auth
    stdio: "inherit",
  });
}

function fetchToken(): string {
  return execSync("gcloud auth application-default print-access-token", {
    encoding: "utf-8",
    timeout: 10_000,
  }).trim();
}

export function getAccessToken(): string {
  if (cachedToken && Date.now() < cachedToken.expiresAt) {
    return cachedToken.token;
  }
  try {
    const token = fetchToken();
    cachedToken = { token, expiresAt: Date.now() + 45 * 60 * 1000 };
    return token;
  } catch (error: any) {
    if (!isReauthError(error)) throw error;
    // Reauth needed — fire off browser-based login flow, then retry
    try {
      refreshADC();
    } catch {
      throw new Error(
        "Automatic re-authentication failed. Please run manually:\n" +
        "  gcloud auth application-default login",
      );
    }
    const token = fetchToken();
    cachedToken = { token, expiresAt: Date.now() + 45 * 60 * 1000 };
    return token;
  }
}

export function clearTokenCache(): void {
  cachedToken = null;
}

function getAnthropicApiKey(): string {
  const key = process.env.ANTHROPIC_API_KEY;
  if (!key) {
    throw new Error("ANTHROPIC_API_KEY env var not set");
  }
  return key;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function vertexUrl(config: Config, publisher: string, model: string, method: string): string {
  const host =
    config.region === "global"
      ? "aiplatform.googleapis.com"
      : `${config.region}-aiplatform.googleapis.com`;
  return `https://${host}/v1/projects/${config.gcpProject}/locations/${config.region}/publishers/${publisher}/models/${model}:${method}`;
}

async function vertexFetch(url: string, body: object, isEmbed: boolean = false): Promise<any> {
  const token = getAccessToken();
  // console.log(`Fetching URL: ${url}`);
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    // If auth expired, clear cache so next call refreshes
    if (res.status === 401) clearTokenCache();
    throw new Error(`Vertex AI ${res.status}: ${text}`);
  }
  const data = await res.json();
  if (isEmbed) return data;
  return Array.isArray(data) ? data[0] : data;
}

async function anthropicFetch(body: object): Promise<any> {
  const apiKey = getAnthropicApiKey();
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Anthropic API ${res.status}: ${text}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Embeddings — Vertex AI (gemini-embedding-001) or Ollama (local)
// ---------------------------------------------------------------------------

async function embedViaVertex(
  text: string,
  config: Config,
  taskType: "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY",
): Promise<Float32Array> {
  const url = vertexUrl(config, "google", config.embedModel, "predict");
  const data = await vertexFetch(url, {
    instances: [{ content: text, task_type: taskType }],
    parameters: { outputDimensionality: config.embedDims },
  }, true);
  const values: number[] = data.predictions[0].embeddings.values;
  return new Float32Array(values);
}

async function embedViaOllama(text: string, config: Config): Promise<Float32Array> {
  const res = await fetch(`${config.ollamaUrl}/api/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: config.embedModel, prompt: text }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Ollama embeddings ${res.status}: ${body}`);
  }
  const data = await res.json();
  return new Float32Array(data.embedding);
}

export async function embedText(
  text: string,
  config: Config,
  taskType: "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY" = "RETRIEVAL_DOCUMENT",
): Promise<Float32Array> {
  if (config.embedProvider === "ollama") {
    return embedViaOllama(text, config);
  }
  return embedViaVertex(text, config, taskType);
}

// ---------------------------------------------------------------------------
// Summarization — shared dispatcher
// ---------------------------------------------------------------------------

async function callLLM(prompt: string, maxTokens: number, config: Config): Promise<string> {
  switch (config.summarizeProvider) {
    case "anthropic": {
      const data = await anthropicFetch({
        model: config.summarizeModel,
        max_tokens: maxTokens,
        messages: [{ role: "user", content: prompt }],
      });
      return data.content[0].text;
    }
    case "vertex-anthropic": {
      const url = vertexUrl(config, "anthropic", config.summarizeModel, "rawPredict");
      const data = await vertexFetch(url, {
        anthropic_version: "vertex-2023-10-16",
        max_tokens: maxTokens,
        messages: [{ role: "user", content: prompt }],
      });
      return data.content[0].text;
    }
    case "vertex-google": {
      const url = vertexUrl(config, "google", config.summarizeModel, "streamGenerateContent");
      const data = await vertexFetch(url, {
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: { maxOutputTokens: maxTokens, temperature: 0.1 },
      });
      return data[0].candidates[0].content.parts[0].text;
    }
  }
}

// ---------------------------------------------------------------------------
// Summarization — Claude Haiku 4.5 on Vertex AI
// ---------------------------------------------------------------------------

const NAMING_PROMPT = `Given this coding session conversation, extract a two-part name.

Return a JSON object with exactly these fields:
- "main_topic": 2-4 words identifying the domain or system being worked on (e.g. "GCP Billing", "Memory Extension", "Pub/Sub Pipeline", "Metabase")
- "sub_topic": 3-6 words describing the specific mission in this session (e.g. "Pub/Sub backlog investigation", "artifact storage implementation", "Cloud Run blue-green upgrade")

Be specific and concrete. Prefer technical terms over vague descriptions.

<conversation>
{{CONVERSATION}}
</conversation>

Respond with ONLY the JSON object. No markdown fencing, no explanation.`;

export async function nameSession(
  conversationText: string,
  config: Config,
): Promise<{ mainTopic: string; subTopic: string }> {
  const prompt = NAMING_PROMPT.replace("{{CONVERSATION}}", conversationText);
  const raw = await callLLM(prompt, 128, config);
  const text = raw.replace(/^```(?:json)?\s*\n?/i, "").replace(/\n?```\s*$/i, "").trim();

  try {
    const parsed = JSON.parse(text);
    return {
      mainTopic: (parsed.main_topic ?? "").trim(),
      subTopic: (parsed.sub_topic ?? "").trim(),
    };
  } catch {
    // If parsing fails, use the whole text as sub_topic with a generic main_topic
    return { mainTopic: "Session", subTopic: text.slice(0, 60) };
  }
}

const SESSION_SUMMARY_PROMPT = `You are summarizing a coding session from a list of memory entries.

Write a 2-3 sentence narrative paragraph describing:
- What the overall goal or problem was
- What was built, changed, or decided
- The current state / outcome

Be specific: mention key technologies, file names, and decisions. Write in past tense. Do NOT use bullet points — write flowing prose.

<memories>
{{MEMORIES}}
</memories>

Respond with ONLY the paragraph. No title, no markdown, no explanation.`;

export async function summarizeSession(
  memorySummaries: string[],
  config: Config,
): Promise<string> {
  const memoriesText = memorySummaries.map((s, i) => `${i + 1}. ${s}`).join("\n");
  const prompt = SESSION_SUMMARY_PROMPT.replace("{{MEMORIES}}", memoriesText);
  const raw = await callLLM(prompt, 256, config);
  return raw.trim();
}

const EXTRACTION_PROMPT = `Extract a concise memory from this coding session interaction.

Return a JSON object with exactly these fields:
- "summary": 2-3 sentences capturing what was done, key decisions, technical details, and outcomes. Be specific about file names, tools, and technologies.
- "topics": array of 3-7 short lowercase tags (e.g. "dbt", "bigquery", "ci-cd", "refactor", "terraform")

Focus on information that would be useful to recall in a future session. Skip pleasantries, meta-discussion, and tool output details.

<interaction>
{{CONVERSATION}}
</interaction>

Respond with ONLY the JSON object. No markdown fencing, no explanation.`;

export async function summarizeInteraction(
  conversationText: string,
  config: Config,
): Promise<{ summary: string; topics: string[] }> {
  const prompt = EXTRACTION_PROMPT.replace("{{CONVERSATION}}", conversationText);
  const raw = await callLLM(prompt, 512, config);

  // Strip markdown fencing if model wrapped the JSON
  const text = raw.replace(/^```(?:json)?\s*\n?/i, "").replace(/\n?```\s*$/i, "").trim();

  try {
    const parsed = JSON.parse(text);
    return {
      summary: parsed.summary ?? text,
      topics: Array.isArray(parsed.topics) ? parsed.topics : [],
    };
  } catch {
    // If model didn't return valid JSON, use the raw text as summary
    return { summary: text.slice(0, 500), topics: [] };
  }
}
