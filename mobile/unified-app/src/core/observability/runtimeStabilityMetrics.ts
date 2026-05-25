import { emitPerfKpi } from "./perfKpi";

type CounterName =
  | "socket_connect_total"
  | "socket_disconnect_total"
  | "socket_reconnect_total"
  | "socket_reconnect_failed_total"
  | "notification_received_total"
  | "notification_deduplicated_total"
  | "notification_duplicate_dropped_total"
  | "notification_filtered_total"
  | "notification_navigation_total"
  | "context_switch_total";

const counters = new Map<CounterName, number>();

function bump(name: CounterName): number {
  const next = (counters.get(name) ?? 0) + 1;
  counters.set(name, next);
  return next;
}

function emitCounter(name: CounterName, payload: Record<string, unknown> = {}): void {
  const total = bump(name);
  emitPerfKpi(`perf.runtime.${name}`, {
    source: "perf.runtime_stability",
    total,
    ...payload,
  });
}

export function recordSocketConnectTotal(payload?: Record<string, unknown>): void {
  emitCounter("socket_connect_total", payload);
}

export function recordSocketDisconnectTotal(payload?: Record<string, unknown>): void {
  emitCounter("socket_disconnect_total", payload);
}

export function recordSocketReconnectTotal(payload?: Record<string, unknown>): void {
  emitCounter("socket_reconnect_total", payload);
}

export function recordSocketReconnectFailedTotal(payload?: Record<string, unknown>): void {
  emitCounter("socket_reconnect_failed_total", payload);
}

export function recordNotificationReceivedTotal(payload?: Record<string, unknown>): void {
  emitCounter("notification_received_total", payload);
}

export function recordNotificationDeduplicatedTotal(payload?: Record<string, unknown>): void {
  emitCounter("notification_deduplicated_total", payload);
}

export function recordNotificationDuplicateDroppedTotal(payload?: Record<string, unknown>): void {
  emitCounter("notification_duplicate_dropped_total", payload);
}

export function recordNotificationFilteredTotal(payload?: Record<string, unknown>): void {
  emitCounter("notification_filtered_total", payload);
}

export function recordNotificationNavigationTotal(payload?: Record<string, unknown>): void {
  emitCounter("notification_navigation_total", payload);
}

export function recordContextSwitchTotal(payload?: Record<string, unknown>): void {
  emitCounter("context_switch_total", payload);
}

export function recordContextSwitchDurationMs(durationMs: number, payload?: Record<string, unknown>): void {
  emitPerfKpi("perf.runtime.context_switch_duration_ms", {
    source: "perf.runtime_stability",
    duration_ms: durationMs,
    ...payload,
  });
}

export function recordNotificationProcessingMs(durationMs: number, payload?: Record<string, unknown>): void {
  emitPerfKpi("perf.runtime.notification_processing_ms", {
    source: "perf.runtime_stability",
    duration_ms: durationMs,
    ...payload,
  });
}

export function getRuntimeStabilityCounter(name: CounterName): number {
  return counters.get(name) ?? 0;
}

export function resetRuntimeStabilityMetricsForTests(): void {
  counters.clear();
}
