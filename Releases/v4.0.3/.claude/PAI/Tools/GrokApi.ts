#!/usr/bin/env bun
/**
 * ============================================================================
 * GROK API - Direct xAI Grok inference with live X/Twitter search
 * ============================================================================
 *
 * PURPOSE:
 * Direct xAI API client for GrokResearcher and any workflow needing:
 * - Live X/Twitter data access (Grok's structural advantage)
 * - Contrarian/unbiased analysis with real-time social data
 * - Search-augmented generation via xAI Responses API + x_search tool
 *
 * USAGE:
 *   bun GrokApi.ts <prompt>                    # Chat (grok-3-mini-fast)
 *   bun GrokApi.ts --model grok-4-0709 <prompt> # Specific model
 *   bun GrokApi.ts --search <prompt>           # Live X search (auto-uses grok-4-0709 + Responses API)
 *   bun GrokApi.ts --system "prompt" <prompt>  # With system prompt
 *
 * MODELS (as of 2026-03-29):
 *   grok-3              - Full Grok 3 (best quality, no x_search)
 *   grok-3-mini-fast    - Fast mini (default for chat, no x_search)
 *   grok-4-0709         - Grok 4 (DEFAULT for --search, supports x_search)
 *   grok-4-1-fast-reasoning    - Fast reasoning (supports x_search)
 *   grok-4-1-fast-non-reasoning - Fast non-reasoning (supports x_search)
 *   grok-code-fast-1    - Code-optimized
 *
 * IMPORTANT: x_search (live X/Twitter) ONLY works with grok-4 family
 *            via the /v1/responses endpoint (NOT /v1/chat/completions)
 *
 * ENV: XAI_API_KEY (required)
 */

const XAI_BASE = "https://api.x.ai/v1";

// Preference order for model selection (best first)
const SEARCH_MODEL_PREFS = ["grok-4-0709", "grok-4-1-fast-reasoning", "grok-4-fast-reasoning", "grok-4-1-fast-non-reasoning"];
const CHAT_MODEL_PREFS = ["grok-3-mini-fast", "grok-3", "grok-4-1-fast-non-reasoning"];

/**
 * Query xAI API for available models. Self-heals against stale model names.
 */
async function discoverModels(apiKey: string): Promise<string[]> {
  try {
    const response = await fetch(`${XAI_BASE}/models`, {
      headers: { Authorization: `Bearer ${apiKey}` },
    });
    if (!response.ok) return [];
    const data = (await response.json()) as { data: Array<{ id: string }> };
    return data.data.map((m) => m.id);
  } catch {
    return [];
  }
}

/**
 * Pick the best available model from a preference list.
 * Falls back to first available if no preferences match.
 */
function pickModel(available: string[], preferences: string[], fallback: string): string {
  for (const pref of preferences) {
    if (available.includes(pref)) return pref;
  }
  // If preferences don't match, try any grok-4 for search or grok-3 for chat
  const family = preferences === SEARCH_MODEL_PREFS ? "grok-4" : "grok-3";
  const anyMatch = available.find((m) => m.startsWith(family));
  if (anyMatch) return anyMatch;
  return fallback;
}

interface GrokOptions {
  model: string;
  search: boolean;
  json: boolean;
  systemPrompt?: string;
  maxTokens: number;
  temperature: number;
}

function parseArgs(): { options: GrokOptions; prompt: string } {
  const args = process.argv.slice(2);
  const options: GrokOptions = {
    model: "", // set after parsing based on --search flag
    search: false,
    json: false,
    maxTokens: 4096,
    temperature: 0.7,
  };

  let explicitModel = false;
  const positional: string[] = [];

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;
    if (arg === "--model" && args[i + 1]) {
      options.model = args[++i]!;
      explicitModel = true;
    } else if (arg === "--search") {
      options.search = true;
    } else if (arg === "--json") {
      options.json = true;
    } else if (arg === "--max-tokens" && args[i + 1]) {
      options.maxTokens = parseInt(args[++i]!, 10);
    } else if (arg === "--temperature" && args[i + 1]) {
      options.temperature = parseFloat(args[++i]!);
    } else if (arg === "--system" && args[i + 1]) {
      options.systemPrompt = args[++i]!;
    } else {
      positional.push(arg);
    }
  }

  // Auto-select model based on mode if not explicitly set
  if (!explicitModel) {
    options.model = options.search ? SEARCH_MODEL_PREFS[0]! : CHAT_MODEL_PREFS[0]!;
  }

  // If --search with a non-grok-4 model, override to grok-4 and warn
  if (options.search && !options.model.startsWith("grok-4")) {
    process.stderr.write(
      `[GrokApi] WARNING: x_search requires grok-4 family. Overriding ${options.model} → ${SEARCH_MODEL_PREFS[0]!}\n`
    );
    options.model = SEARCH_MODEL_PREFS[0]!;
  }

  return { options, prompt: positional.join(" ") };
}

/**
 * Chat Completions API — for standard inference without X search
 */
async function callChatCompletions(prompt: string, options: GrokOptions, apiKey: string): Promise<string> {
  const messages: Array<{ role: string; content: string }> = [];
  if (options.systemPrompt) {
    messages.push({ role: "system", content: options.systemPrompt });
  }
  messages.push({ role: "user", content: prompt });

  const response = await fetch(`${XAI_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: options.model,
      messages,
      max_tokens: options.maxTokens,
      temperature: options.temperature,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`xAI API error ${response.status}: ${errorBody}`);
  }

  const data = (await response.json()) as {
    choices: Array<{ message: { content: string | null } }>;
    usage?: { total_tokens: number; cost_in_usd_ticks?: number };
  };

  const choice = data.choices[0];
  if (!choice) throw new Error("No response from Grok");

  if (data.usage) {
    const costUsd = data.usage.cost_in_usd_ticks
      ? (data.usage.cost_in_usd_ticks / 1_000_000).toFixed(4)
      : "unknown";
    process.stderr.write(`[GrokApi] ${options.model} | ${data.usage.total_tokens} tokens | $${costUsd}\n`);
  }

  return choice.message.content ?? "[no content returned]";
}

/**
 * Responses API — for x_search (live X/Twitter access)
 * Only works with grok-4 family models.
 */
async function callResponses(prompt: string, options: GrokOptions, apiKey: string): Promise<string> {
  const body: Record<string, unknown> = {
    model: options.model,
    input: prompt,
    tools: [{ type: "x_search" }],
  };

  if (options.systemPrompt) {
    body.instructions = options.systemPrompt;
  }

  const response = await fetch(`${XAI_BASE}/responses`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`xAI Responses API error ${response.status}: ${errorBody}`);
  }

  const data = (await response.json()) as {
    output: Array<{
      type: string;
      content?: Array<{ type: string; text?: string }>;
    }>;
    usage?: {
      input_tokens: number;
      output_tokens: number;
      total_tokens: number;
    };
  };

  // Extract text from output messages
  const textParts: string[] = [];
  for (const item of data.output) {
    if (item.type === "message" && item.content) {
      for (const block of item.content) {
        if (block.type === "output_text" && block.text) {
          textParts.push(block.text);
        }
      }
    }
  }

  if (data.usage) {
    process.stderr.write(
      `[GrokApi] ${options.model} + x_search | ${data.usage.total_tokens} tokens\n`
    );
  }

  return textParts.join("\n") || "[no content returned]";
}

// ── Main ──────────────────────────────────────────────────────────────────────

const { options, prompt } = parseArgs();

if (!prompt) {
  console.error("Usage: bun GrokApi.ts [--model MODEL] [--search] [--system PROMPT] <prompt>");
  console.error("\nChat models: grok-3, grok-3-mini-fast (default)");
  console.error("Search models (--search): grok-4-0709 (default), grok-4-1-fast-reasoning");
  console.error("\n--search enables live X/Twitter access (auto-selects grok-4, uses Responses API)");
  console.error("Env: XAI_API_KEY required");
  process.exit(1);
}

const apiKey = process.env["XAI_API_KEY"];
if (!apiKey) {
  console.error("[GrokApi] Error: XAI_API_KEY environment variable not set");
  process.exit(1);
}

try {
  // Self-heal: discover available models and pick the best one
  const available = await discoverModels(apiKey);
  if (available.length > 0) {
    const prefs = options.search ? SEARCH_MODEL_PREFS : CHAT_MODEL_PREFS;
    const best = pickModel(available, prefs, options.model);
    if (best !== options.model) {
      process.stderr.write(`[GrokApi] Model discovery: ${options.model} → ${best} (auto-selected from ${available.length} available)\n`);
      options.model = best;
    }
    // Validate search requires grok-4
    if (options.search && !options.model.startsWith("grok-4")) {
      const searchModel = pickModel(available, SEARCH_MODEL_PREFS, SEARCH_MODEL_PREFS[0]!);
      process.stderr.write(`[GrokApi] x_search requires grok-4 family. Overriding → ${searchModel}\n`);
      options.model = searchModel;
    }
  }

  const result = options.search
    ? await callResponses(prompt, options, apiKey)
    : await callChatCompletions(prompt, options, apiKey);
  console.log(result);
} catch (err) {
  console.error(`[GrokApi] Error: ${(err as Error).message}`);
  process.exit(1);
}
