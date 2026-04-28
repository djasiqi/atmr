import { describe, expect, it } from "@jest/globals";
import {
  canTransitionCompanyRealtime,
  getCompanyRealtimePollingIntervalMs,
  reduceCompanyRealtimeStatus,
} from "./companyRealtimeState";

describe("company realtime state machine", () => {
  it("supports canonical state transitions", () => {
    expect(canTransitionCompanyRealtime("idle", "connecting")).toBe(true);
    expect(canTransitionCompanyRealtime("healthy", "degraded")).toBe(true);
    expect(canTransitionCompanyRealtime("healthy", "failed")).toBe(true);
    expect(canTransitionCompanyRealtime("idle", "healthy")).toBe(false);
    expect(reduceCompanyRealtimeStatus("idle", "healthy")).toBe("idle");
    expect(reduceCompanyRealtimeStatus("connecting", "healthy")).toBe("healthy");
  });

  it("returns fallback polling policy by state", () => {
    expect(getCompanyRealtimePollingIntervalMs("healthy")).toBeNull();
    expect(getCompanyRealtimePollingIntervalMs("connecting")).toBe(30_000);
    expect(getCompanyRealtimePollingIntervalMs("reconnecting")).toBe(15_000);
    expect(getCompanyRealtimePollingIntervalMs("degraded")).toBe(10_000);
    expect(getCompanyRealtimePollingIntervalMs("failed")).toBe(10_000);
  });
});
