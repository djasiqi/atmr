import { describe, expect, it } from "@jest/globals";
import {
  rematerializeLiveDrivers,
  reuseLiveDriverListIfUnchanged,
} from "./liveDriverListMaterialize";

function driver(
  id: number,
  recordedAt: string,
  extras?: { lastSeen?: number; lat?: number; lon?: number; status?: "live" | "recent" | "stale" }
) {
  return {
    driver_id: id,
    recorded_at: recordedAt,
    last_seen_seconds: extras?.lastSeen ?? 0,
    location_status: extras?.status ?? ("live" as const),
    latitude: extras?.lat ?? 46.5,
    longitude: extras?.lon ?? 6.6,
  };
}

function rematerialize(
  sources: ReturnType<typeof driver>[],
  previous: ReturnType<typeof rematerializeLiveDrivers<ReturnType<typeof driver>>> | null,
  nowMs: number,
  refreshAge: boolean
) {
  return rematerializeLiveDrivers({
    sources,
    previousById: previous?.nextById ?? new Map(),
    nowMs,
    refreshAgeForUnchangedSources: refreshAge,
  });
}

describe("rematerializeLiveDrivers", () => {
  const now = Date.parse("2026-09-06T10:00:00.000Z");
  const t0 = new Date(now - 5_000).toISOString();
  const t1 = new Date(now - 2_000).toISOString();

  it("ne rematérialise que le chauffeur dont last_seen_at a changé", () => {
    const ss = driver(1, t0);
    const other = driver(2, t0);
    const first = rematerialize([ss, other], null, now, false);

    const ssMoved = { ...ss, recorded_at: t1 };
    const second = rematerialize([ssMoved, other], first, now, false);

    const otherApplied = first.drivers.find((d) => d.driver_id === 2);
    const otherNext = second.drivers.find((d) => d.driver_id === 2);
    const ssNext = second.drivers.find((d) => d.driver_id === 1);
    expect(otherNext).toBe(otherApplied);
    expect(ssNext).not.toBe(first.drivers.find((d) => d.driver_id === 1));
    expect(ssNext?.recorded_at).toBe(t1);
    expect(second.reused).toBe(1);
    expect(second.replaced).toBe(1);
  });

  it("trois ticks immobiles préservent les 287 refs et la liste", () => {
    const sources = Array.from({ length: 287 }, (_, index) => driver(index + 1, t0, { lastSeen: 4 }));
    let current = rematerialize(sources, null, now, false);
    const firstList = current.drivers;
    const firstObjects = [...current.drivers];

    for (const offset of [5_000, 10_000, 15_000]) {
      current = rematerialize(sources, current, now + offset, true);
      expect(current.replaced).toBe(0);
      expect(current.reused).toBe(287);
      expect(reuseLiveDriverListIfUnchanged(firstList, current.drivers)).toBe(firstList);
      current.drivers.forEach((item, index) => {
        expect(item).toBe(firstObjects[index]);
        expect(item.last_seen_seconds).toBe(4);
        expect(item.recorded_at).toBe(t0);
      });
    }
  });

  it("un refetch live sans mouvement spatial réutilise les objets", () => {
    const sources = [driver(1, t0, { lastSeen: 4 }), driver(2, t0, { lastSeen: 4 })];
    const first = rematerialize(sources, null, now, false);
    const refetched = sources.map((item) => ({ ...item }));
    const afterRefetch = rematerialize(refetched, first, now + 2_000, false);
    expect(afterRefetch.replaced).toBe(0);
    expect(afterRefetch.reused).toBe(2);
    expect(afterRefetch.drivers[0]).toBe(first.drivers[0]);
    expect(afterRefetch.drivers[1]).toBe(first.drivers[1]);
    expect(
      reuseLiveDriverListIfUnchanged(first.drivers, afterRefetch.drivers)
    ).toBe(first.drivers);
  });

  it("un vrai changement spatial ne remplace que le chauffeur déplacé", () => {
    const parked = driver(1, t0);
    const moving = driver(2, t0);
    const first = rematerialize([parked, moving], null, now, false);
    const moved = { ...moving, latitude: 46.51, longitude: 6.61, recorded_at: t1 };
    const second = rematerialize([parked, moved], first, now, false);
    expect(second.reused).toBe(1);
    expect(second.replaced).toBe(1);
    expect(second.drivers.find((d) => d.driver_id === 1)).toBe(
      first.drivers.find((d) => d.driver_id === 1)
    );
    expect(second.drivers.find((d) => d.driver_id === 2)).not.toBe(
      first.drivers.find((d) => d.driver_id === 2)
    );
  });

  it("un franchissement live → recent ne remplace que l’objet concerné", () => {
    const crossingAt = new Date(now - 5_000).toISOString();
    const stillFreshAt = new Date(now - 1_000).toISOString();
    const sources = [
      driver(1, crossingAt, { lastSeen: 4 }),
      ...Array.from({ length: 286 }, (_, index) =>
        driver(index + 2, stillFreshAt, { lastSeen: 1, lat: 46.5 + index * 0.0001 })
      ),
    ];
    const first = rematerialize(sources, null, now, false);
    const crossed = rematerialize(sources, first, now + 26_000, true);
    expect(crossed.replaced).toBe(1);
    expect(crossed.reused).toBe(286);
    expect(crossed.drivers[0]?.driver_id).toBe(1);
    expect(crossed.drivers[0]).not.toBe(first.drivers[0]);
    expect(crossed.drivers[0]?.location_status).toBe("recent");
    expect(crossed.drivers[0]?.latitude).toBe(first.drivers[0]?.latitude);
    expect(crossed.drivers[0]?.longitude).toBe(first.drivers[0]?.longitude);
    expect(crossed.drivers[0]?.recorded_at).toBe(crossingAt);
    expect(crossed.drivers[0]?.last_seen_seconds).toBe(4);
    crossed.drivers.slice(1).forEach((item, index) => {
      expect(item).toBe(first.drivers[index + 1]);
    });
    expect(reuseLiveDriverListIfUnchanged(first.drivers, crossed.drivers)).not.toBe(
      first.drivers
    );
  });

  it("conserve la liste précédente si les références sont identiques", () => {
    const list = [driver(1, t0)];
    expect(reuseLiveDriverListIfUnchanged(list, list)).toBe(list);
    expect(reuseLiveDriverListIfUnchanged(list, [{ ...list[0] }])).not.toBe(list);
  });
});
