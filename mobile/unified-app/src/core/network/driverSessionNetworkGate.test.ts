import { beforeEach, describe, expect, it } from "@jest/globals";
import {
  isAuthOnlyRequestUrl,
  isDriverProtectedRequestUrl,
  isDriverSessionNetworkReady,
  resetDriverSessionNetworkGateForTests,
  setDriverSessionNetworkReady,
  shouldBlockDriverRequestUntilSessionReady,
} from "./driverSessionNetworkGate";

describe("driverSessionNetworkGate", () => {
  beforeEach(() => {
    resetDriverSessionNetworkGateForTests();
  });

  it("bloque le réseau chauffeur tant que SESSION_READY n’est pas ouvert", () => {
    expect(isDriverSessionNetworkReady()).toBe(false);
    expect(shouldBlockDriverRequestUntilSessionReady("/driver/me/bookings")).toBe(true);
    expect(shouldBlockDriverRequestUntilSessionReady("/messages/1/hub/threads")).toBe(true);
    expect(shouldBlockDriverRequestUntilSessionReady("/conversations/inbox")).toBe(true);
    expect(shouldBlockDriverRequestUntilSessionReady("/driver/me/telemetry/push")).toBe(true);
    expect(shouldBlockDriverRequestUntilSessionReady("/driver/me/device-health")).toBe(true);
    expect(shouldBlockDriverRequestUntilSessionReady("/auth/refresh-token")).toBe(false);
    expect(shouldBlockDriverRequestUntilSessionReady("/auth/bootstrap")).toBe(false);
    expect(
      shouldBlockDriverRequestUntilSessionReady("/geocode/geocode", "driver:4")
    ).toBe(true);
    expect(
      shouldBlockDriverRequestUntilSessionReady("/geocode/geocode", "company:1")
    ).toBe(false);
  });

  it("ouvre uniquement après setDriverSessionNetworkReady(true)", () => {
    setDriverSessionNetworkReady(true);
    expect(isDriverSessionNetworkReady()).toBe(true);
    expect(shouldBlockDriverRequestUntilSessionReady("/driver/me/bookings")).toBe(false);
    expect(shouldBlockDriverRequestUntilSessionReady("/conversations/inbox")).toBe(false);
  });

  it("classe auth vs chauffeur", () => {
    expect(isAuthOnlyRequestUrl("/auth/login")).toBe(true);
    expect(isDriverProtectedRequestUrl("/driver/me/bookings/since")).toBe(true);
    expect(isDriverProtectedRequestUrl("/companies/me/dashboard")).toBe(false);
  });
});
