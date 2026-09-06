import { describe, expect, it, jest } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";
import type { AuthContext } from "../contracts/auth";
import { prefetchContextTarget } from "./prefetchContextTarget";

jest.mock("../../features/company/api/companyApi", () => ({
  getCompanyDispatchDelays: jest.fn(),
  getDispatchMissions: jest.fn(),
  getDriversLocationsSnapshot: jest.fn(),
  getOptimizerStatus: jest.fn(),
  getRealtimeDashboard: jest.fn(),
}));

jest.mock("../../features/driver/api/driverHttp", () => ({
  getDriverMissions: jest.fn(),
}));

jest.mock("../../features/driver/messages/api", () => ({
  fetchHubUnreadCount: jest.fn(),
}));

describe("prefetchContextTarget", () => {
  it("ne précharge que dashboard / missions J / drivers live (OPT-07A)", () => {
    const queryClient = new QueryClient();
    const spy = jest
      .spyOn(queryClient, "prefetchQuery")
      .mockResolvedValue(undefined as never);
    const target = {
      context_type: "company",
      context_id: "company:42",
      label: "Test",
      permissions: [],
      is_default: true,
      company_id: 42,
    } as AuthContext;

    prefetchContextTarget(queryClient, target);

    const keys = spy.mock.calls.map((call) => {
      const arg = call[0] as { queryKey?: unknown[] };
      return JSON.stringify(arg.queryKey ?? []);
    });
    expect(keys.some((key) => key.includes("dashboard"))).toBe(true);
    expect(keys.some((key) => key.includes("missions"))).toBe(true);
    expect(keys.some((key) => key.includes("locations"))).toBe(true);
    expect(keys.some((key) => key.includes("optimizer"))).toBe(false);
    expect(keys.some((key) => key.includes("dispatch-delays"))).toBe(false);
    expect(keys.some((key) => key.includes("unread"))).toBe(false);
    expect(spy).toHaveBeenCalledTimes(3);
  });
});
