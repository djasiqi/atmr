/**
 * Métriques locales realtime (mobile) — debug via __DEV__ ou EXPO_PUBLIC_REALTIME_METRICS_DEBUG=1
 */

type RealtimeMetricsSnapshot = {
  reconnectAttemptCount: number;
  gpsBatchFlushCount: number;
  gpsBatchDroppedOrCoalescedCount: number;
  mapSetStatePerSecond: number;
  samplingRate: number;
};

const metrics = {
  reconnectAttemptCount: 0,
  gpsBatchFlushCount: 0,
  gpsBatchDroppedOrCoalescedCount: 0,
  mapSetStateTimestamps: [] as number[],
  gpsEventSampleCounter: 0,
  gpsEventSampleEvery: 1,
};
const MAP_SET_STATE_WINDOW_MS = 1_000;
const MAX_MAP_SET_STATE_TIMESTAMPS = 100;

export function isRealtimeMetricsDebugEnabled(): boolean {
  if (typeof __DEV__ !== "undefined" && __DEV__) return true;
  return process.env.EXPO_PUBLIC_REALTIME_METRICS_DEBUG === "1";
}

export function recordReconnectAttempt(): void {
  metrics.reconnectAttemptCount += 1;
}

export function recordGpsBatchFlush(): void {
  metrics.gpsBatchFlushCount += 1;
}

export function recordGpsBatchCoalesced(count = 1): void {
  metrics.gpsBatchDroppedOrCoalescedCount += count;
}

export function recordMapSetState(): void {
  const now = Date.now();
  metrics.mapSetStateTimestamps.push(now);
  const windowStart = now - MAP_SET_STATE_WINDOW_MS;
  metrics.mapSetStateTimestamps = metrics.mapSetStateTimestamps.filter((t) => t >= windowStart);
  if (metrics.mapSetStateTimestamps.length > MAX_MAP_SET_STATE_TIMESTAMPS) {
    metrics.mapSetStateTimestamps.splice(
      0,
      metrics.mapSetStateTimestamps.length - MAX_MAP_SET_STATE_TIMESTAMPS
    );
  }
}

export function configureGpsMetricsSampling(every: number): void {
  metrics.gpsEventSampleEvery = Math.max(1, Math.floor(every));
}

/** Retourne true si l'événement GPS doit être compté (sampling optionnel). */
export function shouldSampleGpsMetricEvent(): boolean {
  if (metrics.gpsEventSampleEvery <= 1) return true;
  metrics.gpsEventSampleCounter += 1;
  return metrics.gpsEventSampleCounter % metrics.gpsEventSampleEvery === 0;
}

export function getRealtimeMetricsSnapshot(): RealtimeMetricsSnapshot {
  return {
    reconnectAttemptCount: metrics.reconnectAttemptCount,
    gpsBatchFlushCount: metrics.gpsBatchFlushCount,
    gpsBatchDroppedOrCoalescedCount: metrics.gpsBatchDroppedOrCoalescedCount,
    mapSetStatePerSecond: metrics.mapSetStateTimestamps.length,
    samplingRate: metrics.gpsEventSampleEvery <= 1 ? 1 : 1 / metrics.gpsEventSampleEvery,
  };
}
