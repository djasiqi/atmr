/**
 * KPIs performance (plan v3.2.1) — instrumentation Phase 0+.
 * Événements émis via console en dev ; extensible vers backend analytics.
 */

export type PerfKpiEventName =
  | "perf.page_load"
  | "perf.context_switch"
  | "perf.context_switch.phase"
  | "perf.react_query_refetch"
  | "perf.react_query.invalidate"
  | "perf.realtime.notify"
  | "perf.socket.event"
  | "perf.socket.connection_snapshot"
  | "perf.socket.reconnect"
  | "perf.socket.duplicate_event"
  | "perf.context.cache_snapshot"
  | "perf.message_send_ack"
  | "perf.message_send"
  | "perf.chat_cache_mismatch"
  | "perf.mission_received_to_ui"
  | "perf.motion.animation"
  | "perf.js_long_task"
  | "perf.js_heap"
  | "perf.instrumentation.aggregate"
  | "perf.thread_runtime";

export type PerfKpiPayload = {
  source: string;
  duration_ms?: number;
  screen?: string;
  trigger?: string;
  query_key?: string;
  socket_connection_count?: number;
  socket_reconnect_count?: number;
  duplicate_socket_event_count?: number;
  cached_context_count?: number;
  parked_query_count?: number;
  cached_query_count?: number;
  cache_hit?: boolean;
  context_id?: string | null;
  [key: string]: unknown;
};

type PerfKpiSink = (event: PerfKpiEventName, payload: PerfKpiPayload) => void;

const defaultSink: PerfKpiSink = (event, payload) => {
  if (typeof __DEV__ !== "undefined" && __DEV__) {
    console.info(`[perf-kpi] ${event}`, payload);
  }
};

let sink: PerfKpiSink = defaultSink;

export function setPerfKpiSink(custom: PerfKpiSink | null) {
  sink = custom ?? defaultSink;
}

export function emitPerfKpi(event: PerfKpiEventName, payload: PerfKpiPayload) {
  sink(event, {
    ...payload,
    timestamp_client: new Date().toISOString(),
  });
}

// --- Socket session gauges ---

let driverSocketConnected = 0;
let companySocketConnected = 0;
let sessionReconnectCount = 0;
let duplicateEventCount = 0;
let sessionStartedAtMs = Date.now();

export function resetPerfSocketSession() {
  driverSocketConnected = 0;
  companySocketConnected = 0;
  sessionReconnectCount = 0;
  duplicateEventCount = 0;
  sessionStartedAtMs = Date.now();
}

export function recordDriverSocketConnected(connected: boolean) {
  driverSocketConnected = connected ? 1 : 0;
  emitSocketSnapshot("driver");
}

export function recordCompanySocketConnected(connected: boolean) {
  companySocketConnected = connected ? 1 : 0;
  emitSocketSnapshot("company");
}

export function recordSocketReconnect(reason: string, surface: "driver" | "company") {
  sessionReconnectCount += 1;
  emitPerfKpi("perf.socket.reconnect", {
    source: `perf.socket.${surface}`,
    socket_reconnect_count: sessionReconnectCount,
    reason,
    session_duration_ms: Date.now() - sessionStartedAtMs,
  });
  emitSocketSnapshot(surface);
}

export function recordDuplicateSocketEvent(eventType: string, surface: string) {
  duplicateEventCount += 1;
  emitPerfKpi("perf.socket.duplicate_event", {
    source: surface,
    duplicate_socket_event_count: duplicateEventCount,
    event_type: eventType,
  });
}

export function getSocketConnectionCount(): number {
  return driverSocketConnected + companySocketConnected;
}

export function getSocketReconnectCount(): number {
  return sessionReconnectCount;
}

export function getDuplicateSocketEventCount(): number {
  return duplicateEventCount;
}

function emitSocketSnapshot(surface: string) {
  emitPerfKpi("perf.socket.connection_snapshot", {
    source: surface,
    socket_connection_count: getSocketConnectionCount(),
    socket_reconnect_count: sessionReconnectCount,
    duplicate_socket_event_count: duplicateEventCount,
    driver_connected: driverSocketConnected === 1,
    company_connected: companySocketConnected === 1,
  });
}

// --- Page load timers ---

const pageLoadStarts = new Map<string, number>();

export function startPageLoad(screen: string) {
  pageLoadStarts.set(screen, Date.now());
}

export function endPageLoad(screen: string, source: string) {
  const started = pageLoadStarts.get(screen);
  if (started == null) return;
  pageLoadStarts.delete(screen);
  const duration_ms = Date.now() - started;
  const ctx = getPageLoadPerfContext();
  emitPerfKpi("perf.page_load", {
    source,
    screen,
    duration_ms,
    ...ctx,
  });
  recordPageLoadBucketIfEnabled(screen, duration_ms);
}

function getPageLoadPerfContext(): { role?: string; screen_tag?: string } {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { getPerfActiveContext } = require("./perfActiveContext") as {
      getPerfActiveContext: () => { role: string; screen: string };
    };
    const ctx = getPerfActiveContext();
    return { role: ctx.role, screen_tag: ctx.screen };
  } catch {
    return {};
  }
}

function recordPageLoadBucketIfEnabled(screen: string, durationMs: number): void {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { shouldRecordPerfMetric } = require("./perfInstrumentationTier") as {
      shouldRecordPerfMetric: () => boolean;
    };
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { recordPerfBucket } = require("./perfInstrumentationStore") as {
      recordPerfBucket: (c: string, s: string, d: number) => void;
    };
    if (shouldRecordPerfMetric()) {
      recordPerfBucket("page_load", screen, durationMs);
    }
  } catch {
    // optional
  }
}

// --- Context switch ---

export function emitContextSwitchKpi(payload: {
  source: string;
  duration_ms: number;
  cache_hit?: boolean;
  previous_context_id?: string | null;
  next_context_id?: string | null;
}) {
  emitPerfKpi("perf.context_switch", {
    ...payload,
  });
}

// --- React Query refetch counter ---

let refetchCount = 0;

export function recordReactQueryRefetch(queryKey: unknown[], trigger: string) {
  refetchCount += 1;
  const keyStr = JSON.stringify(queryKey).slice(0, 200);
  emitPerfKpi("perf.react_query_refetch", {
    source: "perf.react_query",
    trigger,
    query_key: keyStr,
    react_query_refetch_count: refetchCount,
  });
}

export function getReactQueryRefetchCount(): number {
  return refetchCount;
}

export function resetReactQueryRefetchCountForTests() {
  refetchCount = 0;
}

// --- Context cache snapshot ---

export function emitContextCacheSnapshot(payload: {
  source: string;
  cached_context_count: number;
  parked_query_count: number;
  cached_query_count: number;
  cache_hit?: boolean;
}) {
  emitPerfKpi("perf.context.cache_snapshot", payload);
}

export function countCtxScopedQueries(queryClient: import("@tanstack/react-query").QueryClient): number {
  return queryClient
    .getQueryCache()
    .getAll()
    .filter((q) => Array.isArray(q.queryKey) && q.queryKey[0] === "ctx").length;
}

// --- Dedup guard for socket events ---

const recentEventIds = new Map<string, number>();
const DEDUP_TTL_MS = 60_000;

export function shouldAcceptSocketEvent(eventId: string | null | undefined, eventType: string): boolean {
  if (!eventId) return true;
  const key = `${eventType}:${eventId}`;
  const now = Date.now();
  const seen = recentEventIds.get(key);
  if (seen != null && now - seen < DEDUP_TTL_MS) {
    recordDuplicateSocketEvent(eventType, "socket.dedup");
    return false;
  }
  recentEventIds.set(key, now);
  if (recentEventIds.size > 500) {
    for (const [k, t] of recentEventIds) {
      if (now - t > DEDUP_TTL_MS) recentEventIds.delete(k);
    }
  }
  return true;
}
