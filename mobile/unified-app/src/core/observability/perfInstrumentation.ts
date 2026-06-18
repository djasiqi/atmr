import { getPerfActiveContext } from "./perfActiveContext";
import {
  buildPerfInstrumentationReport,
  recordPerfBucket,
  resetPerfInstrumentationStoreForTests,
} from "./perfInstrumentationStore";
import {
  getPerfInstrumentationTier,
  isPerfInstrumentationActive,
  shouldEmitPerfEventPerCall,
  shouldRecordPerfMetric,
} from "./perfInstrumentationTier";
import { emitPerfKpi } from "./perfKpi";

type PerfKpiEventName =
  | "perf.realtime.notify"
  | "perf.socket.event"
  | "perf.react_query.invalidate"
  | "perf.js_long_task"
  | "perf.js_heap"
  | "perf.context_switch.phase"
  | "perf.instrumentation.aggregate";

export type SocketPerfChannel =
  | "team_chat_message"
  | "conversation_message"
  | "driver_location_batch_ack"
  | "driver_mission_event"
  | "eta_changed"
  | "other";

const AGGREGATE_FLUSH_MS = 60_000;
let aggregateTimer: ReturnType<typeof setInterval> | null = null;
let lastNotifyListenersPeak = 0;

function classifyInvalidateSubKey(queryKey: unknown): string {
  if (!Array.isArray(queryKey)) return "unknown";
  const parts = queryKey.map((p) => String(p));
  const joined = parts.join(".");
  if (joined.includes("threads")) return "threads";
  if (joined.includes("unread")) return "unread";
  if (joined.includes("messages")) return "messages";
  if (joined.includes("message-hub") || joined.includes("message_hub")) return "message_hub";
  return parts.slice(0, 3).join(".") || "unknown";
}

function emitIfVerbose(event: PerfKpiEventName, payload: Record<string, unknown>): void {
  if (!shouldEmitPerfEventPerCall()) return;
  emitPerfKpi(event, {
    source: "perf.instrumentation",
    ...getPerfActiveContext(),
    ...payload,
  });
}

export function recordRealtimeNotify(durationMs: number, listenerCount: number): void {
  if (!shouldRecordPerfMetric()) return;
  if (listenerCount > lastNotifyListenersPeak) lastNotifyListenersPeak = listenerCount;
  recordPerfBucket("notify", "lifecycle", durationMs);
  emitIfVerbose("perf.realtime.notify", {
    duration_ms: durationMs,
    listener_count: listenerCount,
  });
}

export function recordSocketEventByChannel(channel: SocketPerfChannel): void {
  if (!shouldRecordPerfMetric()) return;
  recordPerfBucket("socket_channel", channel, 0, 1);
  emitIfVerbose("perf.socket.event", { channel });
}

export async function traceInvalidateQueries<T>(
  queryKey: unknown,
  trigger: string,
  run: () => Promise<T>
): Promise<T> {
  if (!shouldRecordPerfMetric()) {
    return run();
  }
  const subKey = `${classifyInvalidateSubKey(queryKey)}:${trigger}`;
  const started = Date.now();
  try {
    return await run();
  } finally {
    const durationMs = Date.now() - started;
    recordPerfBucket("invalidate", subKey, durationMs);
    emitIfVerbose("perf.react_query.invalidate", {
      duration_ms: durationMs,
      trigger,
      query_key: JSON.stringify(queryKey).slice(0, 200),
      invalidate_sub_key: classifyInvalidateSubKey(queryKey),
    });
  }
}

export function recordHttpRequest(url: string): void {
  if (!shouldRecordPerfMetric()) return;
  const path = url.split("?")[0]?.slice(0, 120) ?? "unknown";
  recordPerfBucket("http", path, 0, 1);
}

export function recordJsLongTask(durationMs: number): void {
  if (!shouldRecordPerfMetric() || durationMs < 16) return;
  recordPerfBucket("js_long_task", "frame_budget", durationMs);
  const verboseThreshold = Number(
    process.env.EXPO_PUBLIC_PERF_LONG_TASK_EVENT_THRESHOLD_MS ?? "100"
  );
  if (durationMs >= verboseThreshold) {
    emitIfVerbose("perf.js_long_task", { duration_ms: durationMs });
  }
}

export function recordJsHeapSnapshot(usedMb: number, peakMb: number): void {
  if (!shouldRecordPerfMetric()) return;
  recordPerfBucket("heap", "used_mb", usedMb);
  recordPerfBucket("heap", "peak_mb", peakMb);
  emitIfVerbose("perf.js_heap", {
    js_heap_used_mb: usedMb,
    js_heap_peak_mb: peakMb,
  });
}

export function recordFleetEnrichCall(): void {
  if (!shouldRecordPerfMetric()) return;
  recordPerfBucket("fleet_map", "enrich_fleet_drivers", 0, 1);
}

export function recordFleetEnrichDuration(durationMs: number): void {
  if (!shouldRecordPerfMetric()) return;
  recordPerfBucket("fleet_map", "enrich_fleet_drivers_ms", durationMs);
  recordFleetEnrichCall();
}

let socketReconnectCount = 0;
let joinRoomCount = 0;

export function recordSocketReconnectCount(): void {
  socketReconnectCount += 1;
  if (!shouldRecordPerfMetric()) return;
  recordPerfBucket("context_switch", "socket_reconnect", 0, 1);
}

export function recordJoinRoomCount(): void {
  joinRoomCount += 1;
  if (!shouldRecordPerfMetric()) return;
  recordPerfBucket("context_switch", "join_room", 0, 1);
}

export function getSocketInstrumentationCountersForTests(): {
  socketReconnectCount: number;
  joinRoomCount: number;
} {
  return { socketReconnectCount, joinRoomCount };
}

export function resetSocketInstrumentationCountersForTests(): void {
  socketReconnectCount = 0;
  joinRoomCount = 0;
}

export function recordDriverMarkerRender(): void {
  if (!shouldRecordPerfMetric()) return;
  recordPerfBucket("fleet_map", "driver_marker_render", 0, 1);
}

export function recordContextSwitchPhase(
  phase: "total" | "socket" | "prefetch" | "render",
  durationMs: number,
  extra?: Record<string, unknown>
): void {
  if (!shouldRecordPerfMetric()) return;
  recordPerfBucket("context_switch", phase, durationMs);
  emitIfVerbose("perf.context_switch.phase", {
    phase,
    duration_ms: durationMs,
    ...extra,
  });
}

export function recordPageLoadWithContext(
  screen: string,
  durationMs: number,
  source: string
): void {
  if (!shouldRecordPerfMetric()) return;
  recordPerfBucket("page_load", screen, durationMs);
  emitPerfKpi("perf.page_load", {
    source,
    screen,
    duration_ms: durationMs,
    ...getPerfActiveContext(),
  });
}

export function flushPerfInstrumentationAggregates(): void {
  if (!isPerfInstrumentationActive()) return;
  const report = buildPerfInstrumentationReport(10);
  emitPerfKpi("perf.instrumentation.aggregate", {
    source: "perf.instrumentation",
    tier: getPerfInstrumentationTier(),
    notify_listeners_peak: lastNotifyListenersPeak,
    top_by_count: report.top_by_count,
    top_by_sum_ms: report.top_by_sum_ms,
    row_count: report.rows.length,
  });
}

export function startPerfInstrumentationAggregates(): void {
  if (!isPerfInstrumentationActive()) return;
  if (aggregateTimer) return;
  aggregateTimer = setInterval(() => {
    flushPerfInstrumentationAggregates();
  }, AGGREGATE_FLUSH_MS);
}

export function stopPerfInstrumentationAggregates(): void {
  if (aggregateTimer) {
    clearInterval(aggregateTimer);
    aggregateTimer = null;
  }
}

export function exportPerfInstrumentationSnapshot(): ReturnType<typeof buildPerfInstrumentationReport> {
  return buildPerfInstrumentationReport(10);
}

export function resetPerfInstrumentationForTests(): void {
  stopPerfInstrumentationAggregates();
  resetPerfInstrumentationStoreForTests();
  lastNotifyListenersPeak = 0;
}

export { classifyInvalidateSubKey };

export {
  startMessageSend,
  endMessageSend,
  recordMessageSendRetry,
  recordChatCacheMismatch,
  compareOptimisticToServer,
  countThreadOrderDrift,
} from "./perfMessageSend";
