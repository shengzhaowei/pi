// Read-only context inspector backing the /context command. It only reads
// session data and reports it; it never modifies the session, the agent
// loop, or the next model request.
import { getCurrentSystemMessage, getCurrentTools } from "@earendil-works/pi-ai";
import {
	CACHE_TTL_MS,
	type CacheMiss,
	collectCacheMisses,
	computeCacheWaste,
	type ModelPriceSource,
} from "./cache-stats.ts";
import {
	type CompactionSettings,
	estimateProjectedContextTokens,
	estimateTokens,
	shouldCompact,
} from "./compaction/index.ts";
import type { SessionEntry, SessionProjection } from "./session-manager.ts";

/** Why the last counted cache miss happened, derived only from CacheMiss fields. */
export type CacheMissReason = "model changed" | "idle longer than cache TTL" | "prompt prefix changed";

/** A counted cache miss on the most recent turn that paid for one. */
export interface LastCacheMiss {
	missedTokens: number;
	missedCost: number;
	idleMs: number;
	modelChanged: boolean;
	reason: CacheMissReason;
}

/** Everything /context reports, collected from existing session data. */
export interface ContextReport {
	totalTokens: number;
	/** True when no provider-reported usage was available and the total is an estimate. */
	estimated: boolean;
	contextWindow: number;
	percent: number;
	systemPromptTokens: number;
	toolsTokens: number;
	toolCount: number;
	historyTokens: number;
	cache: {
		missedTokens: number;
		missedCost: number;
		missCount: number;
		lastMiss: LastCacheMiss | undefined;
	};
	compaction: {
		enabled: boolean;
		reserveTokens: number;
		thresholdTokens: number;
		remainingTokens: number;
		willCompact: boolean;
	};
}

export interface ContextReportInput {
	projection: SessionProjection;
	branchEntries: SessionEntry[];
	entries: SessionEntry[];
	contextWindow: number;
	/** Provider-reported prompt tokens from getContextUsage, or null when unknown. */
	reportedTokens: number | null;
	compactionSettings: CompactionSettings;
	modelRuntime: ModelPriceSource;
}

/** Collect a context report from the current session projection and entries. */
export function collectContextReport(input: ContextReportInput): ContextReport {
	const estimate = estimateProjectedContextTokens(input.projection, input.branchEntries);
	const totalTokens = input.reportedTokens ?? estimate.tokens;
	const collapsed = getCurrentSystemMessage(input.projection.messages);
	const tools = getCurrentTools(input.projection.messages);
	const systemTotal = collapsed ? estimateTokens(collapsed) : 0;
	// estimateTokens charges a system message as ceil((text + tools JSON) / 4),
	// so the tool share is the JSON part on the same chars/4 basis and the
	// prompt share is the remainder. The split is off by at most one token.
	const toolsTokens = Math.ceil(JSON.stringify(tools).length / 4);
	const systemPromptTokens = Math.max(0, systemTotal - toolsTokens);
	// History is the residual so the three segments sum to the reported total.
	// The total prefers provider-reported usage when available, while the
	// segments are chars/4 estimates, so the residual absorbs that difference.
	const historyTokens = Math.max(0, totalTokens - systemPromptTokens - toolsTokens);
	const waste = computeCacheWaste(input.entries, input.modelRuntime);
	const misses = collectCacheMisses(input.entries, input.modelRuntime);
	const last = [...misses.values()].at(-1);
	const thresholdTokens = input.contextWindow - input.compactionSettings.reserveTokens;
	return {
		totalTokens,
		estimated: input.reportedTokens === null,
		contextWindow: input.contextWindow,
		percent: (totalTokens / input.contextWindow) * 100,
		systemPromptTokens,
		toolsTokens,
		toolCount: tools.length,
		historyTokens,
		cache: {
			missedTokens: waste.missedTokens,
			missedCost: waste.missedCost,
			missCount: waste.missCount,
			lastMiss:
				last === undefined
					? undefined
					: {
							missedTokens: last.missedTokens,
							missedCost: last.missedCost,
							idleMs: last.idleMs,
							modelChanged: last.modelChanged,
							reason: lastMissReason(last),
						},
		},
		compaction: {
			enabled: input.compactionSettings.enabled,
			reserveTokens: input.compactionSettings.reserveTokens,
			thresholdTokens,
			remainingTokens: thresholdTokens - totalTokens,
			willCompact: shouldCompact(totalTokens, input.contextWindow, input.compactionSettings),
		},
	};
}

function lastMissReason(miss: CacheMiss): CacheMissReason {
	if (miss.modelChanged) return "model changed";
	if (miss.idleMs > CACHE_TTL_MS) return "idle longer than cache TTL";
	return "prompt prefix changed";
}

function percentOf(part: number, total: number): string {
	if (total <= 0) return "0.0%";
	return `${((part / total) * 100).toFixed(1)}%`;
}

/** Render a collected report as plain text for ephemeral TUI display. */
export function formatContextReport(report: ContextReport): string {
	const lines: string[] = [];
	lines.push("Context");
	lines.push(
		`Total: ${report.totalTokens.toLocaleString()} tokens (${report.percent.toFixed(1)}% of ${report.contextWindow.toLocaleString()})${report.estimated ? " (estimated)" : ""}`,
	);
	lines.push(
		`System prompt: ${report.systemPromptTokens.toLocaleString()} tokens (${percentOf(report.systemPromptTokens, report.totalTokens)})`,
	);
	lines.push(
		`Tools: ${report.toolsTokens.toLocaleString()} tokens (${percentOf(report.toolsTokens, report.totalTokens)}, ${report.toolCount} tools)`,
	);
	lines.push(
		`History: ${report.historyTokens.toLocaleString()} tokens (${percentOf(report.historyTokens, report.totalTokens)})`,
	);
	lines.push("");
	lines.push("Cache");
	if (report.cache.missCount === 0) {
		lines.push("Cache waste: none counted");
	} else {
		const missLabel = report.cache.missCount === 1 ? "1 miss" : `${report.cache.missCount} misses`;
		lines.push(
			`Cache waste: ${report.cache.missedTokens.toLocaleString()} tokens re-billed ($${report.cache.missedCost.toFixed(3)}, ${missLabel})`,
		);
		if (report.cache.lastMiss !== undefined) {
			lines.push(
				`Last miss: ${report.cache.lastMiss.missedTokens.toLocaleString()} tokens re-billed, ${report.cache.lastMiss.reason}`,
			);
		}
	}
	// Pi does not retain request payloads, so the exact first differing byte
	// cannot be recovered from session entries; report that instead of guessing.
	lines.push("First differing byte: unknown (Pi does not retain request payloads)");
	lines.push("");
	lines.push("Compaction");
	if (!report.compaction.enabled) {
		lines.push("Compaction: disabled");
	} else {
		lines.push(
			`Threshold: ${report.compaction.thresholdTokens.toLocaleString()} tokens (reserve ${report.compaction.reserveTokens.toLocaleString()})`,
		);
		lines.push(`Remaining before compaction: ${report.compaction.remainingTokens.toLocaleString()} tokens`);
		if (report.compaction.willCompact) {
			lines.push("Compaction will trigger on the next run.");
		}
	}
	lines.push("Segments are chars/4 estimates.");
	return lines.join("\n");
}
