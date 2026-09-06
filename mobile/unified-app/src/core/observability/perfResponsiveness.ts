/**
 * PERF-01 — ressenti utilisateur (Cockpit + Courses).
 * Ne change pas l’UX, le GPS, ni les règles métier : mesures uniquement.
 */

import { getPerfActiveContext } from "./perfActiveContext";
import { recordPerfBucket } from "./perfInstrumentationStore";
import {
  shouldEmitPerfEventPerCall,
  shouldRecordPerfMetric,
} from "./perfInstrumentationTier";
import { emitPerfKpi } from "./perfKpi";

export type ResponsivenessTapAction =
  | "tab.dashboard"
  | "tab.rides"
  | "tab.messages"
  | "tab.create"
  | "rides.expand"
  | "rides.details"
  | "rides.edit"
  | "rides.date"
  | "cockpit.live"
  | "cockpit.date"
  | "cockpit.inbox"
  | "cockpit.stat"
  | "other";

type TapSession = {
  id: number;
  action: ResponsivenessTapAction;
  screen: string;
  startedAt: number;
  feedbackAt: number | null;
};

let nextTapId = 1;
const taps = new Map<number, TapSession>();

function emitTap(
  phase: "feedback" | "local" | "navigation",
  session: TapSession,
  durationMs: number
): void {
  recordPerfBucket("tap", `${session.action}.${phase}`, durationMs);
  if (!shouldEmitPerfEventPerCall()) return;
  emitPerfKpi("perf.tap", {
    source: "perf.responsiveness",
    screen: session.screen,
    action: session.action,
    phase,
    duration_ms: durationMs,
    ...getPerfActiveContext(),
  });
}

export function startResponsivenessTap(
  action: ResponsivenessTapAction,
  screen?: string
): number {
  if (!shouldRecordPerfMetric()) return 0;
  const id = nextTapId++;
  taps.set(id, {
    id,
    action,
    screen: screen ?? getPerfActiveContext().screen,
    startedAt: Date.now(),
    feedbackAt: null,
  });
  return id;
}

/** À appeler sur `onPressIn` : démarre le tap et note le premier feedback visuel. */
export function beginTapFeedback(action: ResponsivenessTapAction, screen?: string): number {
  const id = startResponsivenessTap(action, screen);
  markTapVisualFeedback(id);
  return id;
}

export function markTapVisualFeedback(tapId: number): void {
  if (!shouldRecordPerfMetric() || tapId <= 0) return;
  const session = taps.get(tapId);
  if (!session || session.feedbackAt != null) return;
  session.feedbackAt = Date.now();
  emitTap("feedback", session, session.feedbackAt - session.startedAt);
}

export function endTapLocal(tapId: number): void {
  if (!shouldRecordPerfMetric() || tapId <= 0) return;
  const session = taps.get(tapId);
  if (!session) return;
  emitTap("local", session, Date.now() - session.startedAt);
  taps.delete(tapId);
}

export function endTapNavigation(tapId: number): void {
  if (!shouldRecordPerfMetric() || tapId <= 0) return;
  const session = taps.get(tapId);
  if (!session) return;
  emitTap("navigation", session, Date.now() - session.startedAt);
  taps.delete(tapId);
}

export function recordScreenRender(screen: string): void {
  if (!shouldRecordPerfMetric()) return;
  recordPerfBucket("screen_render", screen, 0, 1);
}

export function recordQueryCacheAccess(query: string, cacheHit: boolean): void {
  if (!shouldRecordPerfMetric()) return;
  recordPerfBucket("query_cache", `${query}.${cacheHit ? "hit" : "miss"}`, 0, 1);
  if (!shouldEmitPerfEventPerCall()) return;
  emitPerfKpi("perf.query.cache", {
    source: "perf.responsiveness",
    query_key: query,
    cache_hit: cacheHit,
    ...getPerfActiveContext(),
  });
}

export function recordApiRoundtrip(url: string, durationMs: number, status?: number): void {
  if (!shouldRecordPerfMetric()) return;
  const path = url.split("?")[0]?.slice(0, 120) ?? "unknown";
  recordPerfBucket("api_roundtrip", path, durationMs);
  if (!shouldEmitPerfEventPerCall()) return;
  emitPerfKpi("perf.api.roundtrip", {
    source: "perf.responsiveness",
    query_key: path,
    duration_ms: durationMs,
    status,
    ...getPerfActiveContext(),
  });
}

export type MissionDetailsPhase =
  | "tap"
  | "navigation"
  | "snapshot_render"
  | "cache_hit"
  | "http_complete"
  | "server_reconciled";

export function recordMissionDetailsPhase(
  phase: MissionDetailsPhase,
  durationMs = 0,
  extra?: Record<string, unknown>
): void {
  if (!shouldRecordPerfMetric()) return;
  recordPerfBucket("mission_details", phase, durationMs);
  if (!shouldEmitPerfEventPerCall()) return;
  emitPerfKpi("perf.mission_details", {
    source: "perf.responsiveness",
    screen: "company.ride-details",
    action: `mission_details.${phase}`,
    phase,
    duration_ms: durationMs,
    ...getPerfActiveContext(),
    ...extra,
  });
}

export function resetResponsivenessTapsForTests(): void {
  taps.clear();
  nextTapId = 1;
}
