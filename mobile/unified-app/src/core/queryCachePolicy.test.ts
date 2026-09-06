import { describe, expect, it } from "@jest/globals";
import {
  QUERY_CACHE_POLICY,
  classifyDispatchDay,
  dispatchDayCacheOptions,
  isoDateDayDelta,
  queryCacheOptions,
} from "./queryCachePolicy";

describe("OPT-05 query cache policy", () => {
  it("ne recycle pas une seule durée pour toutes les familles", () => {
    const staleTimes = new Set(
      (Object.keys(QUERY_CACHE_POLICY) as (keyof typeof QUERY_CACHE_POLICY)[]).map(
        (family) => QUERY_CACHE_POLICY[family].staleTime
      )
    );
    expect(staleTimes.size).toBeGreaterThanOrEqual(4);
    expect(QUERY_CACHE_POLICY.realtime.staleTime).toBeLessThan(
      QUERY_CACHE_POLICY.operational.staleTime
    );
    expect(QUERY_CACHE_POLICY.operational.staleTime).toBeLessThan(
      QUERY_CACHE_POLICY.adjacent.staleTime
    );
    expect(QUERY_CACHE_POLICY.adjacent.gcTime).toBe(QUERY_CACHE_POLICY.operational.gcTime);
    expect(QUERY_CACHE_POLICY.operational.gcTime).toBeGreaterThan(
      QUERY_CACHE_POLICY.operational.staleTime
    );
  });

  it("classe J / J±1 / hors voisinage", () => {
    expect(classifyDispatchDay("2026-09-06", { today: "2026-09-06" })).toBe("operational");
    expect(classifyDispatchDay("2026-09-05", { today: "2026-09-06" })).toBe("adjacent");
    expect(classifyDispatchDay("2026-09-07", { today: "2026-09-06" })).toBe("adjacent");
    expect(classifyDispatchDay("2026-09-01", { today: "2026-09-06" })).toBe("historical");
    expect(classifyDispatchDay("2026-09-07", { completeDay: true, today: "2026-09-06" })).toBe(
      "operational"
    );
  });

  it("J±1 ne refetch pas au focus ; J ouvert peut", () => {
    const adjacent = dispatchDayCacheOptions("2026-09-07", { today: "2026-09-06" });
    const open = dispatchDayCacheOptions("2026-09-07", {
      completeDay: true,
      today: "2026-09-06",
    });
    expect(adjacent.refetchOnWindowFocus).toBe(false);
    expect(adjacent.staleTime).toBe(QUERY_CACHE_POLICY.adjacent.staleTime);
    expect(open.staleTime).toBe(QUERY_CACHE_POLICY.operational.staleTime);
    expect(open.gcTime).toBe(30 * 60_000);
  });

  it("les référentiels et le détail ont des politiques distinctes", () => {
    expect(queryCacheOptions("referential").refetchOnWindowFocus).toBe(false);
    expect(queryCacheOptions("detail").staleTime).toBe(30_000);
    expect(queryCacheOptions("realtime").staleTime).toBe(10_000);
    expect(isoDateDayDelta("2026-01-02", "2026-01-01")).toBe(1);
  });
});
