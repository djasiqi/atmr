import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockHasStarted = jest.fn<() => Promise<boolean>>();
const mockStart = jest.fn<() => Promise<void>>();
const mockStop = jest.fn<() => Promise<void>>();
const mockIsTaskRegistered = jest.fn<() => Promise<boolean>>();
const mockGetFg = jest.fn<() => Promise<{ status: string; granted: boolean }>>();
const mockGetBg = jest.fn<() => Promise<{ status: string; granted: boolean }>>();
const mockRequestFg = jest.fn<() => Promise<{ granted: boolean }>>();
const mockRequestBg = jest.fn<() => Promise<{ granted: boolean }>>();
const mockEmit = jest.fn();

jest.mock("react-native", () => ({
  AppState: { currentState: "active" },
  Platform: { OS: "android" },
}));

jest.mock("expo-battery", () => ({
  getBatteryLevelAsync: jest.fn().mockResolvedValue(0.9),
}));

jest.mock("expo-location", () => ({
  hasStartedLocationUpdatesAsync: () => mockHasStarted(),
  startLocationUpdatesAsync: (...args: unknown[]) => mockStart(...args),
  stopLocationUpdatesAsync: (...args: unknown[]) => mockStop(...args),
  getForegroundPermissionsAsync: () => mockGetFg(),
  getBackgroundPermissionsAsync: () => mockGetBg(),
  requestForegroundPermissionsAsync: () => mockRequestFg(),
  requestBackgroundPermissionsAsync: () => mockRequestBg(),
  Accuracy: { Balanced: "balanced", Low: "low", High: "high" },
}));

jest.mock("@react-native-async-storage/async-storage", () => ({
  getItem: jest.fn().mockResolvedValue(null),
  setItem: jest.fn().mockResolvedValue(undefined),
  removeItem: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("../../../core/featureFlags/registry", () => ({
  isFeatureEnabled: (key: string) =>
    key === "tracking_background_enabled" || key === "tracking_presence_mode_enabled",
}));

jest.mock("../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: (...args: unknown[]) => mockEmit(...args),
}));

jest.mock("./backgroundRuntimeCompat", () => ({
  canUseBackgroundLocation: () => true,
  describeBackgroundRuntime: () => "dev_client_or_standalone",
}));

jest.mock("./driverTrackingQueue", () => ({
  driverTrackingQueue: {
    enqueue: jest.fn(),
    getSnapshot: jest.fn().mockResolvedValue({ queueDepth: 0 }),
    flush: jest.fn().mockResolvedValue({
      queueDepth: 0,
      sent: 0,
      backendAcked: 0,
      socketEmitted: 0,
      dropped: 0,
    }),
  },
}));

const mockInitAndHealthcheckHeadless = jest.fn<() => Promise<{
  durable: boolean;
  schemaReady: boolean;
  recovered: boolean;
}>>();

jest.mock("./trackingQueueStore", () => ({
  trackingQueueStore: {
    initAndHealthcheckHeadless: (...args: unknown[]) => mockInitAndHealthcheckHeadless(...args),
  },
}));

const mockReadLease = jest.fn();
jest.mock("./trackingContextLease", () => ({
  readTrackingContextLease: (...args: unknown[]) => mockReadLease(...args),
  leaseAllowsCapture: (lease: { state?: string; fromDriver?: boolean } | null) => {
    if (!lease) return false;
    if (lease.state === "driver_active") return true;
    if (lease.state === "switching" && lease.fromDriver === true) return true;
    return false;
  },
  leaseAllowsTransport: (lease: { state?: string } | null) => lease?.state === "driver_active",
}));

jest.mock("../../../core/auth/sessionAuthDecision", () => ({
  getTrackingAuthAvailability: () => ({
    kind: "SESSION_AVAILABLE",
    sessionGenerationId: 1,
    trackingIdentityId: "driver:42:company:1",
    driverId: 42,
  }),
}));

jest.mock("./trackingRuntimeRegistry", () => ({
  validateNativeOwnerForHeadless: ({
    owner,
    lease,
    authUsable,
  }: {
    owner: { trackingGenerationId?: string } | null;
    lease: { state?: string; trackingGenerationId?: string } | null;
    authUsable: boolean;
  }) => {
    if (!lease || lease.state !== "driver_active") {
      return { ok: false, reason: "lease_not_driver_active" };
    }
    if (!owner) return { ok: false, reason: "missing_native_owner" };
    if (!authUsable) return { ok: false, reason: "auth_not_usable" };
    if (owner.trackingGenerationId !== lease.trackingGenerationId) {
      return { ok: false, reason: "tracking_generation_mismatch" };
    }
    return { ok: true };
  },
  isNativeOwnerCurrent: () => true,
}));

const mockDefineTask = jest.fn();

jest.mock("expo-task-manager", () => ({
  defineTask: (...args: unknown[]) => mockDefineTask(...args),
  isTaskRegisteredAsync: () => mockIsTaskRegistered(),
}));

 
const bgTask = require("./backgroundLocationTask") as typeof import("./backgroundLocationTask");
 
const trackingRuntime = require("./trackingRuntime") as typeof import("./trackingRuntime");
 
const { driverTrackingQueue } = require("./driverTrackingQueue") as {
  driverTrackingQueue: {
    enqueue: jest.Mock;
    getSnapshot: jest.Mock;
    flush: jest.Mock;
  };
};

type TaskHandler = (args: {
  data?: { locations?: { timestamp?: number; coords: Record<string, number | null> }[] };
  error?: Error;
}) => Promise<void>;

function getDefinedTaskHandler(): TaskHandler {
  bgTask.initializeBackgroundLocationTask();
  const call = mockDefineTask.mock.calls.find(
    (c) => c[0] === bgTask.BACKGROUND_LOCATION_TASK_NAME
  );
  if (!call || typeof call[1] !== "function") {
    throw new Error("defineTask handler introuvable");
  }
  return call[1] as TaskHandler;
}

async function seedEligibleMissionContext(opts?: {
  leaseState?: "driver_active" | "switching" | "inactive" | "absent";
  fromDriver?: boolean;
  includeOwner?: boolean;
  staleOwner?: boolean;
}): Promise<void> {
  const leaseState = opts?.leaseState ?? "driver_active";
  const includeOwner = opts?.includeOwner !== false;
  const owner = includeOwner
    ? {
        trackingGenerationId: opts?.staleOwner ? "stale" : "trk-1",
        sessionGenerationId: 1,
        trackingIdentityId: "driver:42:company:1",
        missionContextVersion: 1,
        missionId: 42,
        driverId: 42,
      }
    : null;

  if (leaseState === "absent") {
    mockReadLease.mockResolvedValue(null);
  } else if (leaseState === "switching") {
    mockReadLease.mockResolvedValue({
      state: "switching",
      fromDriver: opts?.fromDriver === true,
      updatedAt: Date.now(),
    });
  } else if (leaseState === "inactive") {
    mockReadLease.mockResolvedValue({ state: "inactive", updatedAt: Date.now() });
  } else {
    mockReadLease.mockResolvedValue({
      state: "driver_active",
      contextId: "driver:42",
      driverId: 42,
      sessionGenerationId: 1,
      trackingGenerationId: "trk-1",
      trackingIdentityId: "driver:42:company:1",
      missionId: 42,
      missionContextVersion: 1,
      updatedAt: Date.now(),
    });
  }

  const asyncStorage = require("@react-native-async-storage/async-storage") as {
    getItem: jest.Mock;
  };
  asyncStorage.getItem.mockImplementation(async (key: string) => {
    if (key === "@driver:bg_tracking_context_v1") {
      return JSON.stringify({
        missionId: 42,
        missionStatus: "EN_ROUTE",
        taskMode: "mission",
        updatedAt: new Date().toISOString(),
        nativeOwner: owner,
      });
    }
    return null;
  });
}

function sampleLocation(i: number) {
  return {
    timestamp: Date.now() + i,
    coords: {
      latitude: 48.85 + i * 0.001,
      longitude: 2.35 + i * 0.001,
      accuracy: 10,
      heading: null,
      speed: null,
    },
  };
}

describe("backgroundLocationTask", () => {
  beforeEach(() => {
    trackingRuntime.__resetTrackingRuntimeForTests();
    mockHasStarted.mockReset();
    mockStart.mockReset();
    mockStop.mockReset();
    mockIsTaskRegistered.mockReset();
    mockEmit.mockReset();
    mockDefineTask.mockClear();
    mockInitAndHealthcheckHeadless.mockReset();
    mockInitAndHealthcheckHeadless.mockResolvedValue({
      durable: true,
      schemaReady: true,
      recovered: false,
    });
    mockReadLease.mockReset();
    mockReadLease.mockResolvedValue({
      state: "driver_active",
      contextId: "driver:42",
      driverId: 42,
      sessionGenerationId: 1,
      trackingGenerationId: "trk-1",
      trackingIdentityId: "driver:42:company:1",
      updatedAt: Date.now(),
    });
    driverTrackingQueue.enqueue.mockReset();
    driverTrackingQueue.enqueue.mockResolvedValue(undefined);
    driverTrackingQueue.getSnapshot.mockReset();
    driverTrackingQueue.getSnapshot.mockResolvedValue({ queueDepth: 0 });
    driverTrackingQueue.flush.mockReset();
    driverTrackingQueue.flush.mockResolvedValue({
      queueDepth: 0,
      sent: 0,
      backendAcked: 0,
      socketEmitted: 0,
      dropped: 0,
    });
    mockGetFg.mockResolvedValue({ status: "granted", granted: true });
    mockGetBg.mockResolvedValue({ status: "granted", granted: true });
    mockRequestFg.mockResolvedValue({ granted: true });
    mockRequestBg.mockResolvedValue({ granted: true });
    mockHasStarted.mockResolvedValue(false);
    mockIsTaskRegistered.mockResolvedValue(false);
    mockStart.mockResolvedValue(undefined);
    mockStop.mockResolvedValue(undefined);
     
    const asyncStorage = require("@react-native-async-storage/async-storage") as {
      getItem: jest.Mock;
      setItem: jest.Mock;
      removeItem: jest.Mock;
    };
    asyncStorage.getItem.mockReset();
    asyncStorage.getItem.mockResolvedValue(null);
    asyncStorage.setItem.mockReset();
    asyncStorage.setItem.mockResolvedValue(undefined);
    asyncStorage.removeItem.mockReset();
    asyncStorage.removeItem.mockResolvedValue(undefined);
    bgTask.__resetBackgroundLocationTaskStateForTests();
  });

  afterEach(() => {
    jest.useRealTimers();
    bgTask.__resetBackgroundLocationTaskStateForTests();
  });

  it("getNativeTaskLifecycleStatus exposes taskDefined and taskStarted", async () => {
    bgTask.initializeBackgroundLocationTask();
    mockHasStarted.mockResolvedValue(true);
    const status = await bgTask.getNativeTaskLifecycleStatus();
    expect(status.taskDefined).toBe(true);
    expect(status.taskStarted).toBe(true);
  });

  it("emits start_failed when startLocationUpdatesAsync throws", async () => {
    bgTask.initializeBackgroundLocationTask();
    mockStart.mockRejectedValueOnce(new Error("Foreground service cannot be started"));

    await bgTask.ensureNativeTrackingWhileForeground(42, "EN_ROUTE", {}, "test_start");

    expect(mockEmit).toHaveBeenCalledWith(
      "tracking.background.start_failed",
      expect.objectContaining({
        failure_reason: "start_exception",
      })
    );
    expect(trackingRuntime.getTrackingRuntimeSnapshot().lastNativeStartError).toContain("test_start");
  });

  it("passes killServiceOnDestroy false in foregroundService options", async () => {
    bgTask.initializeBackgroundLocationTask();
    await bgTask.ensureNativeTrackingWhileForeground(11, "EN_ROUTE", {}, "options_test");

    expect(mockStart).toHaveBeenCalled();
    const options = mockStart.mock.calls[0]?.[1] as { foregroundService?: { killServiceOnDestroy?: boolean } };
    expect(options.foregroundService?.killServiceOnDestroy).toBe(false);
  });

  it("restartNativeTrackingFromWake emits wake_restart telemetry for mission context", async () => {
    bgTask.initializeBackgroundLocationTask();
     
    const asyncStorage = require("@react-native-async-storage/async-storage") as {
      getItem: jest.Mock;
    };
    asyncStorage.getItem.mockResolvedValueOnce(
      JSON.stringify({
        missionId: 55,
        missionStatus: "IN_PROGRESS",
        taskMode: "mission",
        updatedAt: new Date().toISOString(),
      })
    );
    mockHasStarted.mockResolvedValue(false);
    mockStart.mockResolvedValue(undefined);

    await bgTask.restartNativeTrackingFromWake("silent_push_wake_test");

    expect(mockEmit).toHaveBeenCalledWith(
      "tracking.background.wake_restart",
      expect.objectContaining({ reason: "silent_push_wake_test", mission_id: 55 })
    );
  });

  it("stopBackgroundLocationTask skips native stop when task is not registered", async () => {
    bgTask.initializeBackgroundLocationTask();
    mockHasStarted.mockResolvedValue(true);
    mockIsTaskRegistered.mockResolvedValue(false);

    await bgTask.stopBackgroundLocationTask("test_stop_unregistered");

    expect(mockStop).not.toHaveBeenCalled();
    expect(mockEmit).toHaveBeenCalledWith(
      "tracking.background.task.stop_skipped",
      expect.objectContaining({
        reason: "task_not_registered",
      })
    );
    expect(mockEmit).toHaveBeenCalledWith(
      "tracking.background.task.stopped",
      expect.objectContaining({ reason: "test_stop_unregistered" })
    );
  });

  it("refresh mission context when native task is already started", async () => {
    bgTask.initializeBackgroundLocationTask();
    mockHasStarted.mockResolvedValue(true);
     
    const asyncStorage = require("@react-native-async-storage/async-storage") as {
      setItem: jest.Mock;
    };
    asyncStorage.setItem.mockClear();

    await bgTask.ensureNativeTrackingWhileForeground(31770, "ASSIGNED", {}, "mission_context_refresh");

    expect(mockStart).not.toHaveBeenCalled();
    expect(asyncStorage.setItem).toHaveBeenCalledWith(
      "@driver:bg_tracking_context_v1",
      expect.stringContaining('"missionId":31770')
    );
  });

  it("uses distanceInterval 0 for mission background updates", async () => {
    bgTask.initializeBackgroundLocationTask();
    await bgTask.ensureNativeTrackingWhileForeground(11, "EN_ROUTE", {}, "distance_test");

    expect(mockStart).toHaveBeenCalled();
    const options = mockStart.mock.calls[0]?.[1] as { distanceInterval?: number };
    expect(options.distanceInterval).toBe(0);
  });

  it("records startup_timeout when watchdog exhausts without task started", async () => {
    jest.useFakeTimers();
    bgTask.initializeBackgroundLocationTask();
    mockHasStarted.mockResolvedValue(false);
    mockStart.mockImplementation(() => Promise.resolve());

    void bgTask.ensureNativeTrackingWhileForeground(7, "IN_PROGRESS", {}, "watchdog_test");
    await Promise.resolve();
    await jest.advanceTimersByTimeAsync(31_000);
    await Promise.resolve();

    const snap = trackingRuntime.getTrackingRuntimeSnapshot();
    expect(snap.lastNativeStartError).toContain("startup_timeout");
    expect(mockEmit).toHaveBeenCalledWith(
      "tracking.background.start_failed",
      expect.objectContaining({ failure_reason: "startup_timeout" })
    );
  });

  it("appelle le healthcheck exactement une fois avant enqueue/flush", async () => {
    await seedEligibleMissionContext();
    const handler = getDefinedTaskHandler();
    await handler({ data: { locations: [sampleLocation(0), sampleLocation(1), sampleLocation(2)] } });

    expect(mockInitAndHealthcheckHeadless).toHaveBeenCalledTimes(1);
    expect(driverTrackingQueue.enqueue).toHaveBeenCalledTimes(3);
    expect(driverTrackingQueue.flush).toHaveBeenCalled();
  });

  it("health KO → zéro enqueue, zéro flush", async () => {
    await seedEligibleMissionContext();
    mockInitAndHealthcheckHeadless.mockResolvedValueOnce({
      durable: false,
      schemaReady: false,
      recovered: false,
    });
    const handler = getDefinedTaskHandler();
    await handler({ data: { locations: [sampleLocation(0), sampleLocation(1)] } });

    expect(mockInitAndHealthcheckHeadless).toHaveBeenCalledTimes(1);
    expect(driverTrackingQueue.enqueue).not.toHaveBeenCalled();
    expect(driverTrackingQueue.flush).not.toHaveBeenCalled();
    expect(mockEmit).toHaveBeenCalledWith(
      "sqlite_headless_init_failed",
      expect.objectContaining({
        durable: false,
        schema_ready: false,
        recovered: false,
        task_name: bgTask.BACKGROUND_LOCATION_TASK_NAME,
      })
    );
  });

  it("health OK + 3 locations → 3 enqueue, un flush", async () => {
    await seedEligibleMissionContext();
    driverTrackingQueue.flush.mockResolvedValue({
      queueDepth: 0,
      sent: 3,
      backendAcked: 3,
      socketEmitted: 0,
      dropped: 0,
    });
    const handler = getDefinedTaskHandler();
    await handler({ data: { locations: [sampleLocation(0), sampleLocation(1), sampleLocation(2)] } });

    expect(mockInitAndHealthcheckHeadless).toHaveBeenCalledTimes(1);
    expect(driverTrackingQueue.enqueue).toHaveBeenCalledTimes(3);
    expect(driverTrackingQueue.flush).toHaveBeenCalledTimes(1);
  });

  it("company cold start (lease inactive) → 0 enqueue, 0 flush", async () => {
    await seedEligibleMissionContext({ leaseState: "inactive" });
    const handler = getDefinedTaskHandler();
    await handler({ data: { locations: [sampleLocation(0)] } });
    expect(driverTrackingQueue.enqueue).not.toHaveBeenCalled();
    expect(driverTrackingQueue.flush).not.toHaveBeenCalled();
    expect(mockInitAndHealthcheckHeadless).not.toHaveBeenCalled();
  });

  it("switching fromDriver → enqueue OK, flush=0", async () => {
    await seedEligibleMissionContext({ leaseState: "switching", fromDriver: true });
    const handler = getDefinedTaskHandler();
    await handler({ data: { locations: [sampleLocation(0), sampleLocation(1)] } });
    expect(driverTrackingQueue.enqueue).toHaveBeenCalledTimes(2);
    expect(driverTrackingQueue.flush).not.toHaveBeenCalled();
  });

  it("owner génération obsolète → 0 réseau", async () => {
    await seedEligibleMissionContext({ staleOwner: true });
    const handler = getDefinedTaskHandler();
    await handler({ data: { locations: [sampleLocation(0)] } });
    expect(driverTrackingQueue.enqueue).not.toHaveBeenCalled();
    expect(driverTrackingQueue.flush).not.toHaveBeenCalled();
  });

  it("nativeOwner undefined conserve l'owner existant", async () => {
    const asyncStorage = require("@react-native-async-storage/async-storage") as {
      getItem: jest.Mock;
      setItem: jest.Mock;
    };
    const owner = {
      trackingGenerationId: "trk-keep",
      sessionGenerationId: 2,
      trackingIdentityId: "driver:1:company:1",
      missionContextVersion: 3,
      missionId: 10,
      driverId: 1,
    };
    asyncStorage.getItem.mockResolvedValue(
      JSON.stringify({
        missionId: 10,
        missionStatus: "EN_ROUTE",
        taskMode: "mission",
        updatedAt: new Date().toISOString(),
        nativeOwner: owner,
      })
    );
    asyncStorage.setItem.mockClear();
    await bgTask.setBackgroundTrackingMissionContext(10, "EN_ROUTE", "mission", null);
    const written = JSON.parse(String(asyncStorage.setItem.mock.calls[0]?.[1] ?? "{}"));
    expect(written.nativeOwner.trackingGenerationId).toBe("trk-keep");
    expect(written.nativeOwner.driverId).toBe(1);
  });

  it("nativeOwner null efface explicitement", async () => {
    const asyncStorage = require("@react-native-async-storage/async-storage") as {
      getItem: jest.Mock;
      setItem: jest.Mock;
    };
    asyncStorage.getItem.mockResolvedValue(
      JSON.stringify({
        missionId: 10,
        missionStatus: "EN_ROUTE",
        taskMode: "mission",
        updatedAt: new Date().toISOString(),
        nativeOwner: {
          trackingGenerationId: "trk",
          sessionGenerationId: 1,
          trackingIdentityId: "driver:1:company:1",
          missionContextVersion: 1,
          missionId: 10,
          driverId: 1,
        },
      })
    );
    asyncStorage.setItem.mockClear();
    await bgTask.setBackgroundTrackingMissionContext(10, "EN_ROUTE", "mission", null, null);
    const written = JSON.parse(String(asyncStorage.setItem.mock.calls[0]?.[1] ?? "{}"));
    expect(written.nativeOwner).toBeNull();
  });

  it("stopPresenceWindowIfStillCurrent : gen 41 vs 42 → NO-OP", async () => {
    const asyncStorage = require("@react-native-async-storage/async-storage") as {
      getItem: jest.Mock;
      setItem: jest.Mock;
      removeItem: jest.Mock;
    };
    asyncStorage.getItem.mockResolvedValue(
      JSON.stringify({
        missionId: 99,
        missionStatus: "EN_ROUTE",
        taskMode: "mission",
        updatedAt: new Date().toISOString(),
        nativeOwner: {
          trackingGenerationId: "gen-42",
          sessionGenerationId: 1,
          trackingIdentityId: "driver:1:company:1",
          missionContextVersion: 42,
          missionId: 99,
          driverId: 1,
        },
      })
    );
    asyncStorage.setItem.mockClear();
    const stopped = await bgTask.stopPresenceWindowIfStillCurrent({
      expectedGenerationId: "gen-41",
      expectedMissionContextVersion: 41,
      reason: "presence_window_closed",
    });
    expect(stopped).toBe(false);
    // Contexte mission intact (pas d'écriture null)
    expect(asyncStorage.setItem).not.toHaveBeenCalled();
  });

  it("stopPresenceWindowIfStillCurrent : même génération presence → stop", async () => {
    const asyncStorage = require("@react-native-async-storage/async-storage") as {
      getItem: jest.Mock;
      setItem: jest.Mock;
      removeItem: jest.Mock;
    };
    asyncStorage.getItem.mockResolvedValue(
      JSON.stringify({
        missionId: null,
        missionStatus: null,
        taskMode: "presence_window",
        updatedAt: new Date().toISOString(),
        nativeOwner: {
          trackingGenerationId: "gen-41",
          sessionGenerationId: 1,
          trackingIdentityId: "driver:1:company:1",
          missionContextVersion: 41,
          missionId: null,
          driverId: 1,
        },
      })
    );
    const stopped = await bgTask.stopPresenceWindowIfStillCurrent({
      expectedGenerationId: "gen-41",
      expectedMissionContextVersion: 41,
      reason: "presence_window_closed",
    });
    expect(stopped).toBe(true);
  });
});

describe("resolveBackgroundGpsQuality (P0-F)", () => {
  it("mission_live batterie faible : High + cadence mission (pas 60s)", () => {
    const q = bgTask.resolveBackgroundGpsQuality({
      trackingMode: "mission_live",
      isLowBattery: true,
      missionIntervalMs: 20_000,
      lowBatteryIntervalMs: 60_000,
    });
    expect(q.accuracy).toBe("high");
    expect(q.timeIntervalMs).toBe(20_000);
    expect(q.batteryDegradesGps).toBe(false);
  });

  it("mission_live batterie normale : High + cadence mission", () => {
    const q = bgTask.resolveBackgroundGpsQuality({
      trackingMode: "mission_live",
      isLowBattery: false,
      missionIntervalMs: 20_000,
    });
    expect(q.accuracy).toBe("high");
    expect(q.timeIntervalMs).toBe(20_000);
  });

  it("availability_presence batterie faible : Low + cadence allongée", () => {
    const q = bgTask.resolveBackgroundGpsQuality({
      trackingMode: "availability_presence",
      isLowBattery: true,
      missionIntervalMs: 20_000,
      presenceMinIntervalMs: 90_000,
      lowBatteryIntervalMs: 60_000,
    });
    expect(q.accuracy).toBe("low");
    expect(q.timeIntervalMs).toBe(90_000);
    expect(q.batteryDegradesGps).toBe(true);
  });

  it("availability_presence batterie normale : Balanced + ≥90s", () => {
    const q = bgTask.resolveBackgroundGpsQuality({
      trackingMode: "availability_presence",
      isLowBattery: false,
      missionIntervalMs: 20_000,
      presenceMinIntervalMs: 90_000,
    });
    expect(q.accuracy).toBe("balanced");
    expect(q.timeIntervalMs).toBe(90_000);
    expect(q.batteryDegradesGps).toBe(false);
  });
});
