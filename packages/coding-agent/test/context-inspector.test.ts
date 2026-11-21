// Tests for the /context inspector: pure collection and formatting only.
// No TUI interaction is exercised here.
import type { AssistantMessage } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import type { ModelPriceSource } from "../src/core/cache-stats.ts";
import { type ContextReportInput, collectContextReport, formatContextReport } from "../src/core/context-inspector.ts";
import type { SessionEntry, SessionProjection } from "../src/core/session-manager.ts";

const zeroCost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 };

const models: ModelPriceSource = {
	getModel: () => ({ cost: { cacheRead: 0.3 } }),
};

const readTool = {
	name: "read",
	description: "Read a file",
	parameters: { type: "object", properties: {} } as never,
};

function projection(): SessionProjection {
	return {
		entries: [],
		messages: [
			{
				role: "system",
				content: "You are a coding assistant.",
				sections: { rules: "Be concise." },
				toolsAdded: [readTool],
				timestamp: 0,
			},
			{ role: "user", content: "hello world, please help", timestamp: 1 },
			{
				role: "assistant",
				content: [{ type: "text", text: "Sure, here it is." }],
				api: "openai-completions",
				provider: "test",
				model: "test-model",
				usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: zeroCost },
				stopReason: "stop",
				timestamp: 2,
			},
		],
		thinkingLevel: "medium",
		model: null,
	} as unknown as SessionProjection;
}

function input(overrides: Partial<ContextReportInput> = {}): ContextReportInput {
	return {
		projection: projection(),
		branchEntries: [],
		entries: [],
		contextWindow: 100_000,
		reportedTokens: null,
		compactionSettings: { enabled: true, reserveTokens: 16384, keepRecentTokens: 20000 },
		modelRuntime: models,
		...overrides,
	};
}

function assistant(options: {
	cacheRead?: number;
	cacheWrite?: number;
	cost?: Partial<typeof zeroCost>;
	model?: string;
	timestamp?: number;
}): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: "anthropic-messages",
		provider: "test",
		model: options.model ?? "test-model",
		usage: {
			input: 0,
			output: 10,
			cacheRead: options.cacheRead ?? 0,
			cacheWrite: options.cacheWrite ?? 0,
			totalTokens: 0,
			cost: { ...zeroCost, ...options.cost },
		},
		stopReason: "stop",
		timestamp: options.timestamp ?? 0,
	} as AssistantMessage;
}

function entry(message: AssistantMessage): SessionEntry {
	return { type: "message", id: "x", parentId: null, timestamp: "", message } as SessionEntry;
}

describe("collectContextReport", () => {
	it("splits system prompt, tools, and history, and the segments sum to the total", () => {
		const report = collectContextReport(input());
		expect(report.toolCount).toBe(1);
		expect(report.toolsTokens).toBe(Math.ceil(JSON.stringify([readTool]).length / 4));
		expect(report.systemPromptTokens).toBeGreaterThan(0);
		expect(report.historyTokens).toBeGreaterThan(0);
		expect(report.systemPromptTokens + report.toolsTokens + report.historyTokens).toBe(report.totalTokens);
		expect(report.estimated).toBe(true);
	});

	it("prefers reported provider usage for the total", () => {
		const report = collectContextReport(input({ reportedTokens: 1000 }));
		expect(report.totalTokens).toBe(1000);
		expect(report.estimated).toBe(false);
		expect(report.percent).toBeCloseTo(1, 5);
	});

	it("reports remaining tokens before compaction and flags when over threshold", () => {
		const small = collectContextReport(
			input({
				contextWindow: 1000,
				reportedTokens: 100,
				compactionSettings: { enabled: true, reserveTokens: 100, keepRecentTokens: 200 },
			}),
		);
		expect(small.compaction.thresholdTokens).toBe(900);
		expect(small.compaction.remainingTokens).toBe(800);
		expect(small.compaction.willCompact).toBe(false);
		const over = collectContextReport(
			input({
				contextWindow: 1000,
				reportedTokens: 950,
				compactionSettings: { enabled: true, reserveTokens: 100, keepRecentTokens: 200 },
			}),
		);
		expect(over.compaction.remainingTokens).toBe(-50);
		expect(over.compaction.willCompact).toBe(true);
		const disabled = collectContextReport(
			input({
				contextWindow: 1000,
				reportedTokens: 950,
				compactionSettings: { enabled: false, reserveTokens: 100, keepRecentTokens: 200 },
			}),
		);
		expect(disabled.compaction.willCompact).toBe(false);
	});

	it("reports the last cache miss and its reason", () => {
		const turn1 = assistant({ cacheWrite: 100_000, cost: { cacheWrite: 0.375 }, timestamp: 0 });
		const turn2 = assistant({
			cacheRead: 100_000,
			cacheWrite: 5_000,
			cost: { cacheRead: 0.03, cacheWrite: 0.019 },
			timestamp: 60_000,
		});
		const switched = assistant({
			cacheWrite: 110_000,
			cost: { cacheWrite: 0.4125 },
			model: "other-model",
			timestamp: 120_000,
		});
		const report = collectContextReport(
			input({ reportedTokens: 100, entries: [entry(turn1), entry(turn2), entry(switched)] }),
		);
		expect(report.cache.missCount).toBe(1);
		expect(report.cache.missedTokens).toBe(105_000);
		expect(report.cache.lastMiss?.reason).toBe("model changed");

		const idle = assistant({
			cacheWrite: 110_000,
			cost: { cacheWrite: 0.4125 },
			timestamp: 60_000 + 6 * 60_000,
		});
		const idleReport = collectContextReport(
			input({ reportedTokens: 100, entries: [entry(turn1), entry(turn2), entry(idle)] }),
		);
		expect(idleReport.cache.lastMiss?.reason).toBe("idle longer than cache TTL");

		const healthy = collectContextReport(input({ reportedTokens: 100, entries: [entry(turn1), entry(turn2)] }));
		expect(healthy.cache.missCount).toBe(0);
		expect(healthy.cache.lastMiss).toBeUndefined();
	});
});

describe("formatContextReport", () => {
	it("renders the three sections and never claims a first differing byte", () => {
		const text = formatContextReport(collectContextReport(input({ reportedTokens: 1000 })));
		expect(text).toContain("Context");
		expect(text).toContain("Cache");
		expect(text).toContain("Compaction");
		expect(text).toContain("System prompt:");
		expect(text).toContain("Tools:");
		expect(text).toContain("History:");
		expect(text).toContain("Remaining before compaction:");
		expect(text).toContain("First differing byte: unknown (Pi does not retain request payloads)");
		expect(text).not.toMatch(/\d{4}-\d{2}-\d{2}/);
	});

	it("says compaction is disabled instead of reporting a remaining count", () => {
		const text = formatContextReport(
			collectContextReport(
				input({
					reportedTokens: 1000,
					compactionSettings: { enabled: false, reserveTokens: 100, keepRecentTokens: 200 },
				}),
			),
		);
		expect(text).toContain("Compaction: disabled");
		expect(text).not.toContain("Remaining before compaction:");
	});
});
