import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";
import type { CompanyDispatchMission, CompanyDispatchMissionListResponse } from "../api/contracts";
import { dispatchMissionsQueryKey } from "./prefetchAdjacentDispatchMissions";
import * as companyApi from "../api/companyApi";
import {
  patchMissionInCachedDays,
  patchRideDetailsIfPresent,
  reconcileAuthoritativeMission,
  refetchObservedDispatchDays,
  resetDispatchMissionCachePatchForTests,
  rideDetailsQueryKey,
} from "./dispatchMissionCachePatch";
import { resetRidesFetchReasonForTests } from "./ridesFetchReason";

function mission(id: number, extra: Partial<CompanyDispatchMission> = {}): CompanyDispatchMission {
  return { mission_id: id, status: "pending", ...extra };
}

function day(
  date: string,
  missions: CompanyDispatchMission[],
  extra: Partial<CompanyDispatchMissionListResponse> = {}
): CompanyDispatchMissionListResponse {
  return {
    context_id: "company:42",
    date,
    missions,
    refreshed_at: "2026-09-06T00:00:00.000Z",
    total: extra.total ?? 87,
    page_size: 50,
    loaded: extra.loaded ?? 87,
    is_complete: extra.is_complete ?? true,
    next_page: extra.next_page ?? 3,
    pagination_error: extra.pagination_error ?? false,
    ...extra,
  };
}

describe("dispatchMissionCachePatch (OPT-04E)", () => {
  beforeEach(() => {
    resetDispatchMissionCachePatchForTests();
    resetRidesFetchReasonForTests();
  });

  it("patche #45711 sans jeter la journée complète ni les refs des autres", () => {
    const queryClient = new QueryClient();
    const a = mission(1, { client_name: "A" });
    const b = mission(2, { client_name: "B" });
    const target = mission(45711, { client_name: "Sonia", driver_id: 10 });
    const key = dispatchMissionsQueryKey("company:42", "2026-09-06");
    queryClient.setQueryData(key, day("2026-09-06", [a, b, target]));

    const result = patchMissionInCachedDays(
      queryClient,
      "company:42",
      mission(45711, { client_name: "Sonia", driver_id: 22, status: "assigned" })
    );

    const next = queryClient.getQueryData<CompanyDispatchMissionListResponse>(key);
    expect(result.patched).toBe(true);
    expect(next?.loaded).toBe(87);
    expect(next?.total).toBe(87);
    expect(next?.is_complete).toBe(true);
    expect(next?.pagination_error).toBe(false);
    expect(next?.next_page).toBe(3);
    expect(next?.missions[0]).toBe(a);
    expect(next?.missions[1]).toBe(b);
    expect(next?.missions[2]?.driver_id).toBe(22);
    expect(next?.missions[2]?.status).toBe("assigned");
  });

  it("ne touche pas J-1 / J+1 quand la mission n’y est pas", () => {
    const queryClient = new QueryClient();
    const neighbor = mission(99, { client_name: "Autre jour" });
    const current = mission(45711, { driver_id: 1 });
    const jKey = dispatchMissionsQueryKey("company:42", "2026-09-06");
    const prevKey = dispatchMissionsQueryKey("company:42", "2026-09-05");
    const nextKey = dispatchMissionsQueryKey("company:42", "2026-09-07");
    const prevDay = day("2026-09-05", [neighbor], { total: 1, loaded: 1 });
    const nextDay = day("2026-09-07", [neighbor], { total: 1, loaded: 1 });
    queryClient.setQueryData(jKey, day("2026-09-06", [current]));
    queryClient.setQueryData(prevKey, prevDay);
    queryClient.setQueryData(nextKey, nextDay);

    patchMissionInCachedDays(
      queryClient,
      "company:42",
      mission(45711, { driver_id: 8, status: "assigned" })
    );

    expect(queryClient.getQueryData(prevKey)).toBe(prevDay);
    expect(queryClient.getQueryData(nextKey)).toBe(nextDay);
    expect(queryClient.getQueryState(prevKey)?.isInvalidated).toBeFalsy();
    expect(queryClient.getQueryState(nextKey)?.isInvalidated).toBeFalsy();
  });

  it("conserve une pagination déjà partielle (erreur page N)", () => {
    const queryClient = new QueryClient();
    const first = mission(1);
    const key = dispatchMissionsQueryKey("company:42", "2026-09-06");
    queryClient.setQueryData(
      key,
      day("2026-09-06", [first], {
        total: 87,
        loaded: 50,
        is_complete: false,
        pagination_error: true,
        next_page: 2,
      })
    );

    patchMissionInCachedDays(queryClient, "company:42", mission(1, { status: "assigned" }));

    const next = queryClient.getQueryData<CompanyDispatchMissionListResponse>(key);
    expect(next?.loaded).toBe(50);
    expect(next?.is_complete).toBe(false);
    expect(next?.pagination_error).toBe(true);
    expect(next?.next_page).toBe(2);
  });

  it("réconcilie ride-details seulement s’il est déjà en cache", () => {
    const queryClient = new QueryClient();
    const detailKey = rideDetailsQueryKey("company:42", 45711);
    queryClient.setQueryData(detailKey, { mission_id: 45711, driver_id: 1 });

    expect(
      patchRideDetailsIfPresent(queryClient, "company:42", 45711, {
        mission_id: 45711,
        driver_id: 9,
      })
    ).toBe(true);
    expect(queryClient.getQueryData(detailKey)).toEqual({ mission_id: 45711, driver_id: 9 });

    expect(
      patchRideDetailsIfPresent(queryClient, "company:42", 12, { mission_id: 12 })
    ).toBe(false);
  });

  it("refetchObservedDispatchDays ne relance pas un J+1 sans observer", async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const refetchSpy = jest.spyOn(queryClient, "refetchQueries");
    const nextKey = dispatchMissionsQueryKey("company:42", "2026-09-07");
    queryClient.setQueryData(nextKey, day("2026-09-07", [mission(3)], { total: 1, loaded: 1 }));

    const count = await refetchObservedDispatchDays(queryClient, "company:42", "recovery");
    expect(count).toBe(0);
    expect(refetchSpy).not.toHaveBeenCalled();
    expect(queryClient.getQueryState(nextKey)?.isInvalidated).toBeFalsy();
  });

  it("reconcileAuthoritativeMission patche puis ignore le doublon mutation/recovery", async () => {
    const detailSpy = jest.spyOn(companyApi, "getCompanyRideDetail").mockResolvedValue({
      mission_id: 45711,
      booking_id: 45711,
      status: "assigned",
      driver: { id: 9, name: "B" },
    });
    const queryClient = new QueryClient();
    const key = dispatchMissionsQueryKey("company:42", "2026-09-06");
    const detailKey = rideDetailsQueryKey("company:42", 45711);
    queryClient.setQueryData(key, day("2026-09-06", [mission(45711, { driver_id: 1 })]));
    queryClient.setQueryData(detailKey, { mission_id: 45711, driver_id: 1 });

    const first = await reconcileAuthoritativeMission(
      queryClient,
      "company:42",
      45711,
      "mutation"
    );
    const second = await reconcileAuthoritativeMission(
      queryClient,
      "company:42",
      45711,
      "recovery"
    );

    expect(first).toBe("patched");
    expect(second).toBe("skipped");
    expect(detailSpy).toHaveBeenCalledTimes(1);
    expect(queryClient.getQueryData<CompanyDispatchMissionListResponse>(key)?.loaded).toBe(87);
    expect(queryClient.getQueryData<CompanyDispatchMissionListResponse>(key)?.is_complete).toBe(
      true
    );
    detailSpy.mockRestore();
  });
});
