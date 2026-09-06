import { describe, expect, it } from "@jest/globals";
import type { CompanyDispatchMission, CompanyDispatchMissionListResponse } from "../api/contracts";
import {
  applyDayPage,
  DISPATCH_DAY_PAGE_SIZE,
  markDayPaginationError,
  remainingUnloadedCount,
  resolveDispatchSearchPresentation,
  shouldFetchNextDayPage,
} from "./dispatchDayPagination";

function mission(
  partial: Partial<CompanyDispatchMission> & { mission_id: number }
): CompanyDispatchMission {
  return {
    status: "assigned",
    client_name: `Client ${partial.mission_id}`,
    ...partial,
  };
}

function pagePayload(
  page: number,
  ids: number[],
  total: number,
  extras?: Partial<CompanyDispatchMissionListResponse>
): CompanyDispatchMissionListResponse {
  const missions = ids.map((id) => mission({ mission_id: id }));
  return {
    context_id: "company:1",
    date: "2026-09-06",
    missions,
    refreshed_at: `t${page}`,
    total,
    page_size: DISPATCH_DAY_PAGE_SIZE,
    loaded: missions.length,
    page,
    next_page: page + 1,
    is_complete: false,
    pagination_error: false,
    ...extras,
  };
}

describe("applyDayPage", () => {
  it("affiche la page 1 immédiatement sans prétendre J complète", () => {
    const ids = Array.from({ length: 50 }, (_, index) => index + 1);
    const incoming = pagePayload(1, ids, 87);
    const day = applyDayPage(undefined, incoming);
    expect(day.loaded).toBe(50);
    expect(day.total).toBe(87);
    expect(day.is_complete).toBe(false);
    expect(day.next_page).toBe(2);
    expect(day.missions).toHaveLength(50);
    expect(day.missions[0]).toBe(incoming.missions[0]);
  });

  it("fusionne la page 2 en conservant les refs de la page 1", () => {
    const page1Ids = Array.from({ length: 50 }, (_, index) => index + 1);
    const page1 = applyDayPage(undefined, pagePayload(1, page1Ids, 87));
    const page1Refs = page1.missions.slice();
    const page2Ids = Array.from({ length: 37 }, (_, index) => 51 + index);
    const merged = applyDayPage(page1, pagePayload(2, page2Ids, 87));
    expect(merged.loaded).toBe(87);
    expect(merged.is_complete).toBe(true);
    expect(merged.missions).toHaveLength(87);
    expect(merged.missions.slice(0, 50)).toEqual(page1Refs);
    for (let index = 0; index < 50; index += 1) {
      expect(merged.missions[index]).toBe(page1Refs[index]);
    }
    expect(merged.missions.map((item) => item.mission_id)).toEqual(
      Array.from({ length: 87 }, (_, index) => index + 1)
    );
  });

  it("déduplique par mission_id et garde la première occurrence", () => {
    const first = applyDayPage(undefined, pagePayload(1, [1, 2, 3], 3));
    const again = applyDayPage(first, pagePayload(1, [1, 2, 3], 3));
    expect(again.missions).toHaveLength(3);
    expect(again.missions[0]).toBe(first.missions[0]);
  });

  it("refuse de fusionner une page d’une autre date", () => {
    const current = applyDayPage(undefined, pagePayload(1, [1, 2], 2));
    const late = pagePayload(2, [99], 2, { date: "2026-09-05" });
    const merged = applyDayPage(current, late);
    expect(merged).toBe(current);
    expect(merged.date).toBe("2026-09-06");
    expect(merged.missions.map((item) => item.mission_id)).toEqual([1, 2]);
  });

  it("une page 2 vide alors que total > loaded pose pagination_error sans perdre la page 1", () => {
    const page1 = applyDayPage(
      undefined,
      pagePayload(1, Array.from({ length: 50 }, (_, index) => index + 1), 87)
    );
    const failed = applyDayPage(page1, pagePayload(2, [], 87));
    expect(failed.missions).toHaveLength(50);
    expect(failed.loaded).toBe(50);
    expect(failed.total).toBe(87);
    expect(failed.is_complete).toBe(false);
    expect(failed.pagination_error).toBe(true);
    expect(failed.next_page).toBe(2);
  });

  it("un refetch page 1 conserve la queue déjà chargée", () => {
    const page1Ids = Array.from({ length: 50 }, (_, index) => index + 1);
    const page2Ids = Array.from({ length: 37 }, (_, index) => 51 + index);
    const full = applyDayPage(
      applyDayPage(undefined, pagePayload(1, page1Ids, 87)),
      pagePayload(2, page2Ids, 87)
    );
    const tailRef = full.missions[50];
    const refreshedPage1 = pagePayload(1, page1Ids, 87, {
      missions: page1Ids.map((id) => mission({ mission_id: id, client_name: `Client ${id}` })),
    });
    const after = applyDayPage(full, refreshedPage1);
    expect(after.missions).toHaveLength(87);
    expect(after.missions[50]).toBe(tailRef);
    expect(after.is_complete).toBe(true);
  });
});

describe("shouldFetchNextDayPage / search presentation", () => {
  it("ne complète pas J±1 (completeDay=false)", () => {
    const day = applyDayPage(
      undefined,
      pagePayload(1, Array.from({ length: 50 }, (_, index) => index + 1), 87)
    );
    expect(shouldFetchNextDayPage(day, false)).toBe(false);
    expect(shouldFetchNextDayPage(day, true)).toBe(true);
  });

  it("s’arrête si erreur pagination — retry reprend la même page", () => {
    const page1 = applyDayPage(
      undefined,
      pagePayload(1, Array.from({ length: 50 }, (_, index) => index + 1), 87)
    );
    const errored = markDayPaginationError(page1);
    expect(errored.loaded).toBe(50);
    expect(errored.is_complete).toBe(false);
    expect(errored.pagination_error).toBe(true);
    expect(shouldFetchNextDayPage(errored, true)).toBe(false);
    expect(errored.next_page).toBe(2);
  });

  it("n’affiche « aucun résultat » que si J est complète", () => {
    expect(
      resolveDispatchSearchPresentation({
        query: "Sonia",
        hitCount: 1,
        loaded: 50,
        total: 87,
        isComplete: false,
      })
    ).toEqual({ kind: "hits" });
    expect(
      resolveDispatchSearchPresentation({
        query: "Sonia",
        hitCount: 0,
        loaded: 50,
        total: 87,
        isComplete: false,
      })
    ).toEqual({ kind: "pending", remaining: 37 });
    expect(
      resolveDispatchSearchPresentation({
        query: "Sonia",
        hitCount: 0,
        loaded: 87,
        total: 87,
        isComplete: true,
      })
    ).toEqual({ kind: "none" });
    expect(
      resolveDispatchSearchPresentation({
        query: "",
        hitCount: 0,
        loaded: 50,
        total: 87,
        isComplete: false,
      })
    ).toEqual({ kind: "idle" });
  });

  it("remainingUnloadedCount ne descend pas sous 0", () => {
    expect(remainingUnloadedCount({ loaded: 87, total: 87 })).toBe(0);
    expect(remainingUnloadedCount({ loaded: 90, total: 87 })).toBe(0);
    expect(remainingUnloadedCount({ loaded: 50, total: 87 })).toBe(37);
  });
});
