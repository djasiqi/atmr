jest.mock("@react-native-async-storage/async-storage", () => ({
  setItem: jest.fn(),
  getItem: jest.fn(),
  removeItem: jest.fn(),
}));

jest.mock("expo-crypto", () => ({
  randomUUID: () => "mock-uuid-kpi",
}));

jest.mock("../logContext", () => ({
  getLogContextSnapshot: () => ({
    platform: "ios",
    app_version: "test",
  }),
}));

jest.mock("../networkState", () => ({
  getNetworkStateSnapshot: () => null,
}));

import {
  getAuthKpiSnapshot,
  logAuthEvent,
  resetAuthKpiSnapshot,
} from "../authLogging";

describe("authLogging KPI snapshot", () => {
  beforeEach(() => {
    resetAuthKpiSnapshot();
  });

  it("agrège soft/hard/logout et unknown_refresh_401", () => {
    logAuthEvent("AUTH_REFRESH_FAIL_SOFT", {
      route: "driver",
      reason: "unknown_refresh_401",
      status: 401,
      outcome: "retry_later",
    });
    logAuthEvent("AUTH_REFRESH_SUCCESS", {
      route: "driver",
      status: 200,
      outcome: "ok",
    });
    logAuthEvent("AUTH_REFRESH_FAIL_HARD", {
      route: "enterprise",
      reason: "refresh_invalid",
      status: 401,
      outcome: "logout",
    });
    logAuthEvent("LOGOUT_TRANSITION", {
      route: "driver",
      reason: "refresh_invalid",
      tenant_id: "42",
    });

    const snapshot = getAuthKpiSnapshot();

    expect(snapshot.unknown_refresh_401_count).toBe(1);
    expect(snapshot.refresh_fail_soft_by_route.driver).toBe(1);
    expect(snapshot.refresh_fail_hard_by_route.enterprise).toBe(1);
    expect(snapshot.forced_logout_by_reason.refresh_invalid).toBe(1);
    expect(snapshot.forced_logout_by_reason_tenant["refresh_invalid|42"]).toBe(1);
    expect(snapshot.median_recovery_delay_ms_by_route.driver).toBeGreaterThanOrEqual(0);
  });
});
