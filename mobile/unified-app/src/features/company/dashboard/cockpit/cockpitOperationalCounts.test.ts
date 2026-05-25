import { computeCockpitOperationalCounts } from "./cockpitOperationalCounts";

describe("computeCockpitOperationalCounts", () => {
  const baseOpsFeed = {
    stats: [
      { key: "delayed" as const, label: "Retards", value: "3", accentColor: "#f00" },
    ],
    delayedMission: null,
  };

  it("does not alias urgentCount to delayedCount", () => {
    const result = computeCockpitOperationalCounts({
      missions: [],
      drivers: [
        {
          driver_id: 1,
          latitude: 46.2,
          longitude: 6.1,
          location_status: "online",
        },
      ],
      opsFeed: baseOpsFeed,
    });
    expect(result.delayedCount).toBe(3);
    expect(result.urgentCount).toBe(0);
  });

  it("counts incident drivers as urgent", () => {
    const result = computeCockpitOperationalCounts({
      missions: [
        {
          mission_id: 10,
          status: "in_progress",
          driver_id: 1,
          assignment_pickup_delay_minutes: 20,
        },
      ],
      drivers: [
        {
          driver_id: 1,
          latitude: 46.2,
          longitude: 6.1,
          location_status: "online",
        },
      ],
      opsFeed: { stats: [], delayedMission: null },
    });
    expect(result.urgentCount).toBeGreaterThanOrEqual(1);
  });
});
