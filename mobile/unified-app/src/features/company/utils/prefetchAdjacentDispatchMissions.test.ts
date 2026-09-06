import { describe, expect, it, jest } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";

jest.mock("../api/companyApi", () => ({
  getDispatchMissions: jest.fn(async () => ({ missions: [], context_id: "company:42", refreshed_at: "" })),
}));

import {
  adjacentIsoDates,
  dispatchMissionsQueryKey,
  prefetchAdjacentDispatchMissions,
  shiftIsoDate,
} from "./prefetchAdjacentDispatchMissions";

describe("prefetchAdjacentDispatchMissions", () => {
  it("calcule J-1 et J+1 sans ambiguïté de fuseau", () => {
    expect(shiftIsoDate("2026-09-05", -1)).toBe("2026-09-04");
    expect(shiftIsoDate("2026-09-05", 1)).toBe("2026-09-06");
    expect(adjacentIsoDates("2026-01-01")).toEqual(["2025-12-31", "2026-01-02"]);
  });

  it("précharge uniquement les voisins, avec la même clé que la journée courante", () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const spy = jest.spyOn(queryClient, "prefetchQuery").mockResolvedValue(undefined as never);
    prefetchAdjacentDispatchMissions(queryClient, "company:42", "2026-09-05");
    expect(spy).toHaveBeenCalledTimes(2);
    const keys = spy.mock.calls.map((call) => (call[0] as { queryKey: unknown }).queryKey);
    expect(keys).toEqual([
      dispatchMissionsQueryKey("company:42", "2026-09-04"),
      dispatchMissionsQueryKey("company:42", "2026-09-06"),
    ]);
    expect(keys.some((key) => JSON.stringify(key).includes("2026-09-05"))).toBe(false);
    expect(spy.mock.calls.every((call) => (call[0] as { staleTime?: number }).staleTime === 10 * 60_000)).toBe(
      true
    );
    expect(spy.mock.calls.every((call) => (call[0] as { refetchOnWindowFocus?: boolean }).refetchOnWindowFocus === false)).toBe(
      true
    );
  });
});
