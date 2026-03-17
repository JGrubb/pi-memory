import { describe, it } from "node:test";
import assert from "node:assert/strict";

/**
 * Tests for the message extraction logic in index.ts.
 *
 * Since extractFromMessages and buildConversationText are not exported
 * from the extension (they're internal to the default export closure),
 * we re-implement the same logic here for isolated testing.
 *
 * If the extraction logic is ever refactored into a separate module,
 * these tests should import from there directly.
 */

// ---------------------------------------------------------------------------
// Duplicated extraction logic (mirrors index.ts internals)
// ---------------------------------------------------------------------------

interface ExtractedContent {
  userPrompt: string;
  assistantResponse: string;
  filesTouched: string[];
  toolsUsed: string[];
}

function extractTextParts(content: unknown): string[] {
  if (typeof content === "string") return [content];
  if (!Array.isArray(content)) return [];
  return content
    .filter((p: any) => p?.type === "text" && typeof p.text === "string")
    .map((p: any) => p.text);
}

function extractFromMessages(messages: any[]): ExtractedContent {
  let userPrompt = "";
  let assistantResponse = "";
  const filesTouched = new Set<string>();
  const toolsUsed = new Set<string>();

  for (const msg of messages) {
    if (!msg || !msg.role) continue;

    if (msg.role === "user") {
      const parts = extractTextParts(msg.content);
      if (parts.length > 0) userPrompt = parts.join("\n").trim();
    } else if (msg.role === "assistant") {
      const parts = extractTextParts(msg.content);
      assistantResponse += parts.join("\n").trim() + "\n";

      if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part?.type === "toolCall" && part.name) {
            toolsUsed.add(part.name);
            const args = part.arguments ?? part.input ?? {};
            if (args.path) filesTouched.add(args.path);
          }
        }
      }
    } else if (msg.role === "toolResult") {
      if (msg.toolName) toolsUsed.add(msg.toolName);
    }
  }

  return {
    userPrompt,
    assistantResponse: assistantResponse.trim().slice(0, 2000),
    filesTouched: Array.from(filesTouched),
    toolsUsed: Array.from(toolsUsed),
  };
}

function buildConversationText(messages: any[]): string {
  const parts: string[] = [];

  for (const msg of messages) {
    if (!msg?.role) continue;

    if (msg.role === "user") {
      const text = extractTextParts(msg.content).join("\n").trim();
      if (text) parts.push(`User: ${text}`);
    } else if (msg.role === "assistant") {
      const text = extractTextParts(msg.content).join("\n").trim();
      if (text) parts.push(`Assistant: ${text}`);

      if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part?.type === "toolCall" && part.name) {
            const argsStr = JSON.stringify(part.arguments ?? part.input ?? {}).slice(0, 200);
            parts.push(`[Tool: ${part.name}(${argsStr})]`);
          }
        }
      }
    }
  }

  const full = parts.join("\n\n");
  return full.length > 6000 ? full.slice(0, 6000) + "\n...[truncated]" : full;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("extractFromMessages", () => {
  it("extracts user prompt from string content", () => {
    const messages = [{ role: "user", content: "Fix the billing model" }];
    const result = extractFromMessages(messages);
    assert.equal(result.userPrompt, "Fix the billing model");
  });

  it("extracts user prompt from array content", () => {
    const messages = [
      {
        role: "user",
        content: [{ type: "text", text: "Fix the billing model" }],
      },
    ];
    const result = extractFromMessages(messages);
    assert.equal(result.userPrompt, "Fix the billing model");
  });

  it("extracts assistant text response", () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "text", text: "I'll fix the billing model now." }],
      },
    ];
    const result = extractFromMessages(messages);
    assert.ok(result.assistantResponse.includes("I'll fix the billing model now."));
  });

  it("extracts tool names from assistant tool calls", () => {
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "Let me read the file." },
          {
            type: "toolCall",
            name: "read",
            arguments: { path: "src/billing.ts" },
          },
        ],
      },
    ];
    const result = extractFromMessages(messages);
    assert.ok(result.toolsUsed.includes("read"));
    assert.ok(result.filesTouched.includes("src/billing.ts"));
  });

  it("extracts tool names from toolResult messages", () => {
    const messages = [{ role: "toolResult", toolName: "bash" }];
    const result = extractFromMessages(messages);
    assert.ok(result.toolsUsed.includes("bash"));
  });

  it("deduplicates tools and files", () => {
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "toolCall", name: "read", arguments: { path: "a.ts" } },
          { type: "toolCall", name: "read", arguments: { path: "a.ts" } },
          { type: "toolCall", name: "edit", arguments: { path: "a.ts" } },
        ],
      },
    ];
    const result = extractFromMessages(messages);
    assert.equal(result.toolsUsed.length, 2); // read, edit
    assert.equal(result.filesTouched.length, 1); // a.ts
  });

  it("handles empty and malformed messages gracefully", () => {
    const messages = [null, undefined, {}, { role: "unknown" }, { role: "user", content: null }];
    const result = extractFromMessages(messages as any[]);
    assert.equal(result.userPrompt, "");
    assert.equal(result.toolsUsed.length, 0);
  });

  it("truncates long assistant responses", () => {
    const longText = "x".repeat(5000);
    const messages = [
      { role: "assistant", content: [{ type: "text", text: longText }] },
    ];
    const result = extractFromMessages(messages);
    assert.ok(result.assistantResponse.length <= 2000);
  });

  it("handles tool calls with input instead of arguments", () => {
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "toolCall", name: "write", input: { path: "out.txt" } },
        ],
      },
    ];
    const result = extractFromMessages(messages);
    assert.ok(result.toolsUsed.includes("write"));
    assert.ok(result.filesTouched.includes("out.txt"));
  });
});

describe("buildConversationText", () => {
  it("formats a simple user/assistant exchange", () => {
    const messages = [
      { role: "user", content: "What files are in src?" },
      {
        role: "assistant",
        content: [{ type: "text", text: "Here are the files in src/..." }],
      },
    ];
    const text = buildConversationText(messages);
    assert.ok(text.includes("User: What files are in src?"));
    assert.ok(text.includes("Assistant: Here are the files in src/..."));
  });

  it("includes tool call annotations", () => {
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "Let me check." },
          { type: "toolCall", name: "bash", arguments: { command: "ls src/" } },
        ],
      },
    ];
    const text = buildConversationText(messages);
    assert.ok(text.includes("[Tool: bash("));
    assert.ok(text.includes("ls src/"));
  });

  it("truncates very long conversations", () => {
    const longMessage = "a".repeat(10000);
    const messages = [{ role: "user", content: longMessage }];
    const text = buildConversationText(messages);
    assert.ok(text.length <= 6020); // 6000 + "[truncated]" overhead
    assert.ok(text.includes("[truncated]"));
  });

  it("handles multi-turn conversations", () => {
    const messages = [
      { role: "user", content: "Step 1" },
      { role: "assistant", content: [{ type: "text", text: "Done 1" }] },
      { role: "user", content: "Step 2" },
      { role: "assistant", content: [{ type: "text", text: "Done 2" }] },
    ];
    const text = buildConversationText(messages);
    assert.ok(text.includes("User: Step 1"));
    assert.ok(text.includes("User: Step 2"));
    assert.ok(text.includes("Assistant: Done 1"));
    assert.ok(text.includes("Assistant: Done 2"));
  });
});
