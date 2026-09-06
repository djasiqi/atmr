import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";
import {
  driverQueryKeys,
  invalidateDriverMissionScope,
  resetDriverMissionScopeInvalidationForTests,
} from "./queryKeys";

describe("invalidateDriverMissionScope", () => {
  beforeEach(() => {
    resetDriverMissionScopeInvalidationForTests();
  });

  it("coalesce les invalidations missions / company-bookings dans la même fenêtre", () => {
    const queryClient = new QueryClient();
    const invalidateSpy = jest
      .spyOn(queryClient, "invalidateQueries")
      .mockResolvedValue(undefined as never);

    invalidateDriverMissionScope(queryClient, "driver:4");
    invalidateDriverMissionScope(queryClient, "driver:4");
    invalidateDriverMissionScope(queryClient, "driver:4", 45711);

    const missionInvalidations = invalidateSpy.mock.calls.filter(
      (call) =>
        JSON.stringify(call[0]?.queryKey) ===
        JSON.stringify(driverQueryKeys.missions("driver:4"))
    );
    const companyInvalidations = invalidateSpy.mock.calls.filter(
      (call) =>
        JSON.stringify(call[0]?.queryKey) ===
        JSON.stringify(driverQueryKeys.companyBookingsToday("driver:4"))
    );
    const detailInvalidations = invalidateSpy.mock.calls.filter(
      (call) =>
        JSON.stringify(call[0]?.queryKey) ===
        JSON.stringify(driverQueryKeys.missionDetail("driver:4", 45711))
    );

    expect(missionInvalidations).toHaveLength(1);
    expect(companyInvalidations).toHaveLength(1);
    expect(detailInvalidations).toHaveLength(1);
    invalidateSpy.mockRestore();
  });
});
