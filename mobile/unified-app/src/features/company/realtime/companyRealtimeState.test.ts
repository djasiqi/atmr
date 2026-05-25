import { describe, expect, it } from "@jest/globals";
import {
  canTransitionCompanyTransport,
  computeCompanyDataFreshness,
  getCompanyRealtimePollingIntervalMs,
  reduceCompanyTransportStatus,
} from "./companyRealtimeState";

describe("company realtime state machine", () => {
  it("supports canonical transport transitions", () => {
    expect(canTransitionCompanyTransport("idle", "connecting")).toBe(true);
    expect(canTransitionCompanyTransport("healthy", "reconnecting")).toBe(true);
    expect(canTransitionCompanyTransport("healthy", "failed")).toBe(true);
    expect(canTransitionCompanyTransport("idle", "healthy")).toBe(false);
    expect(reduceCompanyTransportStatus("idle", "healthy")).toBe("idle");
    expect(reduceCompanyTransportStatus("connecting", "healthy")).toBe("healthy");
  });

  it("computes data freshness independently from transport", () => {
    const now = Date.now();
    const recent = new Date(now - 30_000).toISOString();
    const calm = new Date(now - 90_000).toISOString();
    const old = new Date(now - 6 * 60_000).toISOString();
    expect(computeCompanyDataFreshness(recent, now)).toBe("fresh");
    expect(computeCompanyDataFreshness(calm, now)).toBe("idle");
    expect(computeCompanyDataFreshness(old, now)).toBe("stale");
  });

  it("returns fallback polling policy by transport and freshness", () => {
    expect(getCompanyRealtimePollingIntervalMs("healthy", "fresh")).toBeNull();
    expect(getCompanyRealtimePollingIntervalMs("healthy", "stale")).toBe(10_000);
    expect(getCompanyRealtimePollingIntervalMs("connecting")).toBe(30_000);
    expect(getCompanyRealtimePollingIntervalMs("reconnecting")).toBe(15_000);
    expect(getCompanyRealtimePollingIntervalMs("failed")).toBe(10_000);
  });
});
