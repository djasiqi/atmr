import { describe, expect, it, beforeEach } from "@jest/globals";
import {
  getSocketConnectionCount,
  getSocketReconnectCount,
  getDuplicateSocketEventCount,
  recordCompanySocketConnected,
  recordDriverSocketConnected,
  recordDuplicateSocketEvent,
  resetPerfSocketSession,
  shouldAcceptSocketEvent,
} from "./perfKpi";

describe("perfKpi socket gauges", () => {
  beforeEach(() => {
    resetPerfSocketSession();
  });

  it("tracks at most one driver and one company socket", () => {
    recordDriverSocketConnected(true);
    expect(getSocketConnectionCount()).toBe(1);
    recordCompanySocketConnected(true);
    expect(getSocketConnectionCount()).toBe(2);
    recordDriverSocketConnected(false);
    expect(getSocketConnectionCount()).toBe(1);
  });

  it("deduplicates socket events by id", () => {
    expect(shouldAcceptSocketEvent("evt-1", "booking_updated")).toBe(true);
    expect(shouldAcceptSocketEvent("evt-1", "booking_updated")).toBe(false);
    expect(getDuplicateSocketEventCount()).toBe(1);
  });

  it("starts reconnect count at zero after reset", () => {
    expect(getSocketReconnectCount()).toBe(0);
  });
});
