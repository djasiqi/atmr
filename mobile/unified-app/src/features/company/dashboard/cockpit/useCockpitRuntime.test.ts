import { computeCockpitOperationalCounts } from "./cockpitOperationalCounts";

describe("useCockpitRuntime integration signals", () => {
  it("operational counts feed distinct urgent vs delayed", () => {
    const counts = computeCockpitOperationalCounts({
      missions: [],
      drivers: [],
      opsFeed: {
        stats: [{ key: "delayed", label: "Retards", value: "3", accentColor: "#f00" }],
        delayedMission: null,
      },
    });
    expect(counts.delayedCount).toBe(3);
    expect(counts.urgentCount).toBe(0);
  });
});
