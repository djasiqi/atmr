import { describe, expect, it, beforeEach } from "@jest/globals";
import { setPerfActiveContext } from "./perfActiveContext";
import { resetPerfInstrumentationForTests } from "./perfInstrumentation";
import { buildPerfInstrumentationReport } from "./perfInstrumentationStore";
import { resetPerfInstrumentationTierForTests } from "./perfInstrumentationTier";
import {
  beginTapFeedback,
  endTapLocal,
  endTapNavigation,
  markTapVisualFeedback,
  recordApiRoundtrip,
  recordMissionDetailsPhase,
  recordQueryCacheAccess,
  recordScreenRender,
  resetResponsivenessTapsForTests,
  startResponsivenessTap,
} from "./perfResponsiveness";

describe("perfResponsiveness", () => {
  beforeEach(() => {
    resetPerfInstrumentationForTests();
    resetPerfInstrumentationTierForTests();
    resetResponsivenessTapsForTests();
    process.env.EXPO_PUBLIC_PERF_INSTRUMENTATION_TIER = "dev";
    setPerfActiveContext({ role: "company", screen: "company.rides" });
  });

  it("mesure tap → feedback via beginTapFeedback puis navigation", () => {
    const tapId = beginTapFeedback("tab.rides", "company.tabs");
    endTapNavigation(tapId);
    const report = buildPerfInstrumentationReport(10);
    expect(report.rows.find((row) => row.sub_key === "tab.rides.feedback")?.count).toBe(1);
    expect(report.rows.find((row) => row.sub_key === "tab.rides.navigation")?.count).toBe(1);
  });

  it("mesure tap → feedback puis local", () => {
    const tapId = startResponsivenessTap("rides.expand", "company.rides");
    markTapVisualFeedback(tapId);
    endTapLocal(tapId);
    const report = buildPerfInstrumentationReport(10);
    const feedback = report.rows.find((row) => row.sub_key === "rides.expand.feedback");
    const local = report.rows.find((row) => row.sub_key === "rides.expand.local");
    expect(feedback?.count).toBe(1);
    expect(local?.count).toBe(1);
    expect(feedback?.max_ms).toBeLessThan(50);
  });

  it("compte les rerenders et le cache", () => {
    recordScreenRender("company.dashboard");
    recordScreenRender("company.dashboard");
    recordQueryCacheAccess("company.missions", true);
    recordApiRoundtrip("/company_mobile/dispatch/v1/rides", 42, 200);
    const report = buildPerfInstrumentationReport(10);
    expect(report.rows.find((row) => row.category === "screen_render")?.count).toBe(2);
    expect(report.rows.find((row) => row.sub_key === "company.missions.hit")?.count).toBe(1);
    expect(report.rows.find((row) => row.category === "api_roundtrip")?.max_ms).toBe(42);
  });

  it("mesure TAP → snapshot visible pour Détails", () => {
    recordMissionDetailsPhase("tap");
    recordMissionDetailsPhase("navigation", 80);
    recordMissionDetailsPhase("snapshot_render", 140);
    const report = buildPerfInstrumentationReport(10);
    expect(report.rows.find((row) => row.sub_key === "snapshot_render")?.max_ms).toBe(140);
    expect(report.rows.find((row) => row.sub_key === "tap")?.count).toBe(1);
  });
});
