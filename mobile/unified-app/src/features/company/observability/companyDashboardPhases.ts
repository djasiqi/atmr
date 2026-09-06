/**
 * COMPANY-DASHBOARD-PERF-01 — mesures des phases dashboard.
 * Ne change ni le GPS, ni le merge, ni le rendu : chrono uniquement.
 */

import { getPerfActiveContext } from "../../../core/observability/perfActiveContext";
import { recordPerfBucket } from "../../../core/observability/perfInstrumentationStore";
import {
  shouldEmitPerfEventPerCall,
  shouldRecordPerfMetric,
} from "../../../core/observability/perfInstrumentationTier";
import { emitPerfKpi } from "../../../core/observability/perfKpi";
import { getBootColdStartOriginMs } from "../../../core/observability/bootMilestones";

export type CompanyDashboardPhase =
  | "snapshot"
  | "markers"
  | "overlays"
  | "realtime_fusion"
  | "presentation"
  | "view_model"
  | "react_commit";

export function recordCompanyDashboardPhase(
  phase: CompanyDashboardPhase,
  durationMs: number,
  extra?: Record<string, unknown>
): void {
  if (!shouldRecordPerfMetric()) return;
  recordPerfBucket("company_dashboard_phase", phase, durationMs);
  if (!shouldEmitPerfEventPerCall() && durationMs < 50) return;
  emitPerfKpi("perf.company.dashboard.phase", {
    source: "company.dashboard.phases",
    screen: "company.dashboard",
    phase,
    duration_ms: durationMs,
    ...getPerfActiveContext(),
    ...extra,
  });
}

export function measureCompanyDashboardPhase<T>(
  phase: CompanyDashboardPhase,
  run: () => T,
  extra?: Record<string, unknown>
): T {
  if (!shouldRecordPerfMetric()) return run();
  const started = Date.now();
  try {
    return run();
  } finally {
    recordCompanyDashboardPhase(phase, Date.now() - started, extra);
  }
}

export function markCompanyScreenUsable(
  screen: string,
  extra?: Record<string, unknown>
): void {
  if (!shouldRecordPerfMetric()) return;
  const since_cold_start_ms = Date.now() - getBootColdStartOriginMs();
  recordPerfBucket("company_screen_usable", screen, since_cold_start_ms);
  if (!shouldEmitPerfEventPerCall()) return;
  emitPerfKpi("perf.company.screen.usable", {
    source: "company.dashboard.phases",
    screen,
    duration_ms: since_cold_start_ms,
    since_cold_start_ms,
    ...getPerfActiveContext(),
    ...extra,
  });
}
