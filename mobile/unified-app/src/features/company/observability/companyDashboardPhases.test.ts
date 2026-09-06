import { beforeEach, describe, expect, it } from "@jest/globals";
import { setPerfActiveContext } from "../../../core/observability/perfActiveContext";
import { resetPerfInstrumentationForTests } from "../../../core/observability/perfInstrumentation";
import { buildPerfInstrumentationReport } from "../../../core/observability/perfInstrumentationStore";
import { resetPerfInstrumentationTierForTests } from "../../../core/observability/perfInstrumentationTier";
import {
  measureCompanyDashboardPhase,
  recordCompanyDashboardPhase,
} from "./companyDashboardPhases";

describe("companyDashboardPhases", () => {
  beforeEach(() => {
    resetPerfInstrumentationForTests();
    resetPerfInstrumentationTierForTests();
    process.env.EXPO_PUBLIC_PERF_INSTRUMENTATION_TIER = "dev";
    setPerfActiveContext({ role: "company", screen: "company.dashboard" });
  });

  it("mesure une phase synchrone", () => {
    const value = measureCompanyDashboardPhase("markers", () => 42);
    expect(value).toBe(42);
    const report = buildPerfInstrumentationReport(10);
    expect(report.rows.find((row) => row.sub_key === "markers")?.count).toBe(1);
  });

  it("enregistre react_commit", () => {
    recordCompanyDashboardPhase("react_commit", 83);
    const report = buildPerfInstrumentationReport(10);
    expect(report.rows.find((row) => row.sub_key === "react_commit")?.max_ms).toBe(83);
  });

  it("mesure presentation et view_model", () => {
    expect(measureCompanyDashboardPhase("presentation", () => "ok")).toBe("ok");
    expect(measureCompanyDashboardPhase("view_model", () => 1)).toBe(1);
    const report = buildPerfInstrumentationReport(10);
    expect(report.rows.find((row) => row.sub_key === "presentation")?.count).toBe(1);
    expect(report.rows.find((row) => row.sub_key === "view_model")?.count).toBe(1);
  });
});
