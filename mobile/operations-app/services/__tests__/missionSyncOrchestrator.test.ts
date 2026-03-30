import {
  __resetMissionSyncOrchestratorForTests,
  requestMissionSync,
  requestMissionSyncWithMeta,
} from "../missionSyncOrchestrator";

const mockGetAssignedTrips = jest.fn();

jest.mock("../api", () => ({
  getAssignedTrips: (...args: unknown[]) => mockGetAssignedTrips(...args),
}));

describe("missionSyncOrchestrator", () => {
  beforeEach(() => {
    __resetMissionSyncOrchestratorForTests();
    mockGetAssignedTrips.mockReset();
    mockGetAssignedTrips.mockResolvedValue([{ id: 1 } as any]);
  });

  it("coalesce 2 appels socket_connect rapprochés en 1 GET (même clé in-flight)", async () => {
    let resolveFirst!: (v: any) => void;
    const first = new Promise((r) => {
      resolveFirst = r;
    });
    mockGetAssignedTrips.mockReturnValueOnce(first).mockResolvedValue([{ id: 2 } as any]);

    const p1 = requestMissionSync("socket_connect", {});
    const p2 = requestMissionSync("socket_connect", {});
    expect(mockGetAssignedTrips).toHaveBeenCalledTimes(1);
    resolveFirst!([{ id: 99 } as any]);
    const [a, b] = await Promise.all([p1, p2]);
    expect(a).toEqual([{ id: 99 }]);
    expect(b).toEqual([{ id: 99 }]);
    expect(mockGetAssignedTrips).toHaveBeenCalledTimes(1);
  });

  it("socket_connect et manual_screen en parallèle => 1 GET (même since)", async () => {
    let resolveFirst!: (v: any) => void;
    const first = new Promise((r) => {
      resolveFirst = r;
    });
    mockGetAssignedTrips.mockReturnValueOnce(first).mockResolvedValue([{ id: 2 } as any]);

    const p1 = requestMissionSync("socket_connect", {});
    const p2 = requestMissionSync("manual_screen", {});
    expect(mockGetAssignedTrips).toHaveBeenCalledTimes(1);
    resolveFirst!([{ id: 99 } as any]);
    const [a, b] = await Promise.all([p1, p2]);
    expect(a).toEqual([{ id: 99 }]);
    expect(b).toEqual([{ id: 99 }]);
    expect(mockGetAssignedTrips).toHaveBeenCalledTimes(1);
  });

  it("2 triggers différents séquentiels après fenêtre de coalescing => 2 GET", async () => {
    await requestMissionSync("manual_screen");
    await new Promise((r) => setTimeout(r, 500));
    await requestMissionSync("hydrate_empty");
    expect(mockGetAssignedTrips).toHaveBeenCalledTimes(2);
  });

  it("passe syncTrigger au transport", async () => {
    await requestMissionSync("reconcile_now");
    expect(mockGetAssignedTrips).toHaveBeenCalledWith({
      since: undefined,
      syncTrigger: "reconcile_now",
    });
  });

  it("requestMissionSyncWithMeta expose debounce_cache en rafale séquentielle", async () => {
    const r1 = await requestMissionSyncWithMeta("manual_screen", {});
    expect(r1.outcome).toBe("network");
    const r2 = await requestMissionSyncWithMeta("hydrate_empty", {});
    expect(r2.outcome).toBe("debounce_cache");
    expect(mockGetAssignedTrips).toHaveBeenCalledTimes(1);
  });
});
