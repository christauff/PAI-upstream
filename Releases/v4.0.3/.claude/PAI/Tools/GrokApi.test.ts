#!/usr/bin/env bun
/**
 * GrokApi Unit Tests
 *
 * Tests the xAI Grok API client tool.
 * Requires XAI_API_KEY in environment for live tests.
 * Run: XAI_API_KEY=xai-... bun test PAI/Tools/GrokApi.test.ts
 */

import { describe, test, expect } from "bun:test";
import { $ } from "bun";

const GROK_API = `${process.env["HOME"]}/.claude/PAI/Tools/GrokApi.ts`;
const HAS_KEY = !!process.env["XAI_API_KEY"];
const TIMEOUT = 30_000;

describe("GrokApi Tool", () => {

  test("prints usage when no prompt given", async () => {
    const result = await $`bun ${GROK_API} 2>&1`.nothrow().text();
    expect(result).toContain("Usage:");
  });

  test("fails gracefully without XAI_API_KEY", async () => {
    const result = await $`XAI_API_KEY= bun ${GROK_API} "hello" 2>&1`.nothrow().text();
    expect(result).toContain("XAI_API_KEY");
  });

  test("returns response from grok-3-mini-fast", async () => {
    if (!HAS_KEY) return;
    const result = await $`bun ${GROK_API} "What is 2+2? Answer with just the number."`.text();
    expect(result).toContain("4");
  }, TIMEOUT);

  test("accepts --model flag", async () => {
    if (!HAS_KEY) return;
    const result = await $`bun ${GROK_API} --model grok-3-mini-fast "Say the word hello"`.text();
    expect(result.toLowerCase()).toContain("hello");
  }, TIMEOUT);

  test("accepts --system flag for system prompt", async () => {
    if (!HAS_KEY) return;
    const result = await $`bun ${GROK_API} --system "You must respond only with the word PINEAPPLE" "What is your favorite fruit?"`.text();
    expect(result.toUpperCase()).toContain("PINEAPPLE");
  }, TIMEOUT);

  test("logs usage to stderr", async () => {
    if (!HAS_KEY) return;
    const proc = Bun.spawn(["bun", GROK_API, "say hi"], {
      env: { ...process.env },
      stdout: "pipe",
      stderr: "pipe",
    });
    const stderr = await new Response(proc.stderr).text();
    await proc.exited;
    expect(stderr).toContain("[GrokApi]");
    expect(stderr).toContain("tokens");
  }, TIMEOUT);

  test("self-heals invalid model via discovery", async () => {
    if (!HAS_KEY) return;
    const result = await $`bun ${GROK_API} --model nonexistent-model "hello" 2>&1`.nothrow().text();
    expect(result).toContain("Model discovery");
    expect(result).toContain("auto-selected");
  }, TIMEOUT);
});
