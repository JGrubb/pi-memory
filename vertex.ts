import { execSync } from "node:child_process";
import type { Config } from "./types.js";

// ---------------------------------------------------------------------------
// Auth — GCP (Vertex AI) or Anthropic API
// ---------------------------------------------------------------------------

let cachedToken: { token: string; expiresAt: number } | null = null;

export function getAccessToken(): string {
  if (cachedToken && Date.now() < cachedToken.expiresAt) {
    return cachedToken.token;
  }
  const token = execSync("gcloud auth application-default print-access-token", {
    encoding: "utf-8",
    timeout: 10_000,
  }).trim();
  cachedToken = { token, expiresAt: Date.now() + 45 * 60 * 1000 };
  return token;
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
  const isGoogle = publisher === "google";
  // Anthropic models on Vertex require a specific region (not "global").
  // Fall back to us-east5 if region is left at the default "global".
  const region =
    !isGoogle && config.region === "global" ? "us-east5" : config.region;
  const host =
    region === "global"
      ? "aiplatform.googleapis.com"
      : `${region}-aiplatform.googleapis.com`;
  // For Google models (Gemini), the URL structure is the same but kept explicit
  return `https://${host}/v1/projects/${config.gcpProject}/locations/${region}/publishers/${publisher}/models/${model}:${method}`;
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
// Embeddings — Vertex AI gemini-embedding-001
// ---------------------------------------------------------------------------

export async function embedText(
  text: string,
  config: Config,
  taskType: "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY" = "RETRIEVAL_DOCUMENT",
): Promise<Float32Array> {
  const url = vertexUrl(config, "google", config.embeddingModel, "predict");
  const data = await vertexFetch(url, {
    instances: [{ content: text, task_type: taskType }],
    parameters: { outputDimensionality: config.embeddingDims },
  }, true);
  const values: number[] = data.predictions[0].embeddings.values;
  return new Float32Array(values);
}

// ---------------------------------------------------------------------------
// Summarization — Claude Haiku 4.5 on Vertex AI
// ---------------------------------------------------------------------------

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
  let raw: string;

  if (config.summarizeProvider === "anthropic") {
    // Direct Anthropic API — requires ANTHROPIC_API_KEY
    const data = await anthropicFetch({
      model: config.summarizeModel,
      max_tokens: 512,
      messages: [{ role: "user", content: prompt }],
    });
    raw = data.content[0].text;
  } else if (config.summarizeModel.startsWith("claude")) {
    // Claude served through Vertex AI (Anthropic publisher)
    const url = vertexUrl(config, "anthropic", config.summarizeModel, "rawPredict");
    const data = await vertexFetch(url, {
      anthropic_version: "vertex-2023-10-16",
      max_tokens: 512,
      messages: [{ role: "user", content: prompt }],
    });
    raw = data.content[0].text;
  } else {
    // Gemini via Vertex AI
    const url = vertexUrl(config, "google", config.summarizeModel, "streamGenerateContent");
    const data = await vertexFetch(url, {
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: {
        maxOutputTokens: 512,
        temperature: 0.1,
      },
    });
    raw = data[0].candidates[0].content.parts[0].text;
  }

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
