import { runRoleTransition } from "@/services/sessionModeTransition";
import { getAuthSurfaceRole, setAuthSurfaceRole } from "@/services/authSurface";

jest.mock("@/services/socket", () => ({
  disconnectSocket: jest.fn(),
}));

jest.mock("@/services/locationTracker", () => ({
  ensureBackgroundTrackingStopped: jest.fn(async () => {}),
  stopAdaptiveLocationTracking: jest.fn(),
}));

const mockSyncEngineStop = jest.fn();
const mockSyncEngineInstance = { stop: mockSyncEngineStop };

jest.mock("@/services/syncEngine", () => ({
  getSyncEngine: jest.fn(() => mockSyncEngineInstance),
}));

const mockStopMission = jest.fn(async (..._args: unknown[]) => {});

jest.mock("@/services/missionState", () => ({
  MissionStateManager: {
    stopMission: (...args: unknown[]) => mockStopMission(...args),
  },
}));

import { disconnectSocket } from "@/services/socket";
import { ensureBackgroundTrackingStopped, stopAdaptiveLocationTracking } from "@/services/locationTracker";

describe("sessionModeTransition", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    setAuthSurfaceRole("enterprise");
  });

  it("driver → enterprise : teardown driver + disconnect + surface enterprise", async () => {
    setAuthSurfaceRole("driver");
    await runRoleTransition({
      fromRole: "driver",
      toRole: "enterprise",
      reason: "test_driver_to_ent",
      options: { preserveMissionState: false },
    });

    expect(stopAdaptiveLocationTracking).toHaveBeenCalled();
    expect(ensureBackgroundTrackingStopped).toHaveBeenCalled();
    expect(mockSyncEngineStop).toHaveBeenCalled();
    expect(mockStopMission).toHaveBeenCalled();
    expect(disconnectSocket).toHaveBeenCalled();
    expect(getAuthSurfaceRole()).toBe("enterprise");
  });

  it("enterprise → driver : disconnect sans teardown driver-only", async () => {
    setAuthSurfaceRole("enterprise");
    await runRoleTransition({
      fromRole: "enterprise",
      toRole: "driver",
      reason: "test_ent_to_driver",
      options: { preserveMissionState: true },
    });

    expect(stopAdaptiveLocationTracking).not.toHaveBeenCalled();
    expect(mockSyncEngineStop).not.toHaveBeenCalled();
    expect(mockStopMission).not.toHaveBeenCalled();
    expect(disconnectSocket).toHaveBeenCalled();
    expect(getAuthSurfaceRole()).toBe("driver");
  });

  it("même rôle sans option : noop infrastructure, surface mise à jour", async () => {
    setAuthSurfaceRole("driver");
    await runRoleTransition({
      fromRole: "driver",
      toRole: "driver",
      reason: "test_noop",
    });

    expect(disconnectSocket).not.toHaveBeenCalled();
    expect(mockSyncEngineStop).not.toHaveBeenCalled();
    expect(getAuthSurfaceRole()).toBe("driver");
  });
});
