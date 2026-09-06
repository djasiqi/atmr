import { beforeEach, describe, expect, it } from "@jest/globals";
import { resetDriverSessionNetworkGateForTests, setDriverSessionNetworkReady } from "./driverSessionNetworkGate";
import {
  isCompanyProtectedRequestUrl,
  isCompanySessionNetworkReady,
  shouldBlockCompanyRequestUntilSessionReady,
} from "./companySessionNetworkGate";

describe("companySessionNetworkGate", () => {
  beforeEach(() => {
    resetDriverSessionNetworkGateForTests();
  });

  it("bloque les GET protégés tant que SESSION_READY n’est pas ouvert", () => {
    expect(isCompanySessionNetworkReady()).toBe(false);
    expect(
      shouldBlockCompanyRequestUntilSessionReady("/company_mobile/dispatch/v1/rides")
    ).toBe(true);
    expect(
      shouldBlockCompanyRequestUntilSessionReady("/company_dispatch/delays/live")
    ).toBe(true);
    expect(shouldBlockCompanyRequestUntilSessionReady("/companies/notifications")).toBe(true);
    expect(shouldBlockCompanyRequestUntilSessionReady("/company/request-offers")).toBe(true);
    expect(
      shouldBlockCompanyRequestUntilSessionReady("/companies/me/drivers/locations/live")
    ).toBe(true);
    expect(shouldBlockCompanyRequestUntilSessionReady("/auth/refresh-token")).toBe(false);
    expect(shouldBlockCompanyRequestUntilSessionReady("/auth/bootstrap")).toBe(false);
    expect(
      shouldBlockCompanyRequestUntilSessionReady("/geocode/autocomplete", "company:1")
    ).toBe(true);
    expect(
      shouldBlockCompanyRequestUntilSessionReady("/geocode/autocomplete", "driver:4")
    ).toBe(false);
  });

  it("ouvre uniquement après SESSION_READY", () => {
    setDriverSessionNetworkReady(true);
    expect(isCompanySessionNetworkReady()).toBe(true);
    expect(
      shouldBlockCompanyRequestUntilSessionReady("/company_mobile/dispatch/v1/rides")
    ).toBe(false);
    expect(shouldBlockCompanyRequestUntilSessionReady("/companies/notifications")).toBe(false);
  });

  it("classe les URLs entreprise", () => {
    expect(isCompanyProtectedRequestUrl("/companies/notifications")).toBe(true);
    expect(isCompanyProtectedRequestUrl("/company/request-offers")).toBe(true);
    expect(isCompanyProtectedRequestUrl("/company_dispatch/delays")).toBe(true);
    expect(isCompanyProtectedRequestUrl("/dispatch/v1/rides")).toBe(true);
    expect(isCompanyProtectedRequestUrl("/driver/me/bookings")).toBe(false);
    expect(isCompanyProtectedRequestUrl("/auth/refresh-token")).toBe(false);
  });
});
