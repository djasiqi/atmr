import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import AsyncStorage from "@react-native-async-storage/async-storage";

const mockUpdateDriverMissionStatus = jest.fn<(...args: any[]) => any>();

jest.mock("./api", () => ({
  updateDriverMissionStatus: (...args: unknown[]) => mockUpdateDriverMissionStatus(...args),
}));

jest.mock("../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: jest.fn(),
}));

jest.mock("../../core/featureFlags/registry", () => ({
  isFeatureEnabled: jest.fn(() => false),
}));

jest.mock("./services/driverTrackingQueue", () => ({
  driverTrackingQueue: {
    getSnapshot: jest.fn(async () => ({ oldestItemAgeMs: 0, queueDepth: 0 })),
  },
}));

// Import APRÈS les mocks (singleton module).
import { driverOfflineQueue } from "./offlineQueue";

describe("driverOfflineQueue (P1 file contextualisée)", () => {
  beforeEach(async () => {
    jest.clearAllMocks();
    await AsyncStorage.clear();
    // Réinitialise le contexte actif entre les tests.
    await driverOfflineQueue.setActiveContext(null);
    await driverOfflineQueue.purgeMission(1);
    await driverOfflineQueue.purgeMission(2);
  });

  it("retire l'action après succès direct (removeAction) — plus aucun replay", async () => {
    await driverOfflineQueue.setActiveContext("driver:1");
    const queued = await driverOfflineQueue.enqueue(1, "EN_ROUTE", null);
    expect(await driverOfflineQueue.count()).toBe(1);
    await driverOfflineQueue.removeAction(queued.id);
    expect(await driverOfflineQueue.count()).toBe(0);
    const result = await driverOfflineQueue.flush();
    expect(result.sent).toBe(0);
    expect(mockUpdateDriverMissionStatus).not.toHaveBeenCalled();
  });

  it("erreur permanente (retryable=false) : action retirée, les suivantes continuent", async () => {
    await driverOfflineQueue.setActiveContext("driver:1");
    await driverOfflineQueue.enqueue(1, "EN_ROUTE", null);
    await driverOfflineQueue.enqueue(1, "ARRIVED", null);
    mockUpdateDriverMissionStatus
      .mockRejectedValueOnce({ retryable: false, code: "driver_transition_stale" })
      .mockResolvedValueOnce({ status: "arrived", mission_milestone: "ARRIVED" });

    const result = await driverOfflineQueue.flush();

    expect(result.dropped).toBe(1);
    expect(result.sent).toBe(1);
    expect(await driverOfflineQueue.count()).toBe(0);
    expect(mockUpdateDriverMissionStatus).toHaveBeenCalledTimes(2);
  });

  it("erreur transitoire (réseau) : l'action reste en file pour replay", async () => {
    await driverOfflineQueue.setActiveContext("driver:1");
    await driverOfflineQueue.enqueue(1, "EN_ROUTE", null);
    mockUpdateDriverMissionStatus.mockRejectedValueOnce(
      Object.assign(new Error("Network Error"), { retryable: undefined })
    );

    const result = await driverOfflineQueue.flush();

    expect(result.sent).toBe(0);
    expect(result.failed).toBe(1);
    expect(await driverOfflineQueue.count()).toBe(1);
  });

  it("C8 : changement de contexte chauffeur → actions étrangères purgées", async () => {
    await driverOfflineQueue.setActiveContext("driver:1");
    await driverOfflineQueue.enqueue(1, "EN_ROUTE", null);
    expect(await driverOfflineQueue.count()).toBe(1);

    await driverOfflineQueue.setActiveContext("driver:2");
    expect(await driverOfflineQueue.count()).toBe(0);

    const result = await driverOfflineQueue.flush();
    expect(result.sent).toBe(0);
    expect(mockUpdateDriverMissionStatus).not.toHaveBeenCalled();
  });

  it("C7 : replay FIFO par queuedAt — aucune inversion d'ordre", async () => {
    await driverOfflineQueue.setActiveContext("driver:1");
    await driverOfflineQueue.enqueue(1, "EN_ROUTE", null);
    await driverOfflineQueue.enqueue(1, "ARRIVED", null);
    await driverOfflineQueue.enqueue(1, "IN_PROGRESS", null);
    mockUpdateDriverMissionStatus.mockResolvedValue({ status: "ok" });

    const result = await driverOfflineQueue.flush();

    expect(result.sent).toBe(3);
    const sentStatuses = mockUpdateDriverMissionStatus.mock.calls.map(
      (call) => (call[0] as { targetStatus: string }).targetStatus
    );
    expect(sentStatuses).toEqual(["EN_ROUTE", "ARRIVED", "IN_PROGRESS"]);
  });
});
