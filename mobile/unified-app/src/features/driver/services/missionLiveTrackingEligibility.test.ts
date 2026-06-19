import { Platform } from "react-native";

import {
  evaluateMissionTrackingCapability,
  hasMissionTrackingCapability,
  requiresLiveTrackingPermission,
} from "./missionLiveTrackingEligibility";

jest.mock("expo-location", () => ({
  getForegroundPermissionsAsync: jest.fn(),
  getBackgroundPermissionsAsync: jest.fn(),
  hasServicesEnabledAsync: jest.fn(),
}));

jest.mock("./backgroundLocationTask", () => ({
  getNativeTaskLifecycleStatus: jest.fn(),
}));

const Location = jest.requireMock("expo-location") as {
  getForegroundPermissionsAsync: jest.Mock;
  getBackgroundPermissionsAsync: jest.Mock;
  hasServicesEnabledAsync: jest.Mock;
};

const bgTask = jest.requireMock("./backgroundLocationTask") as {
  getNativeTaskLifecycleStatus: jest.Mock;
};

describe("missionLiveTrackingEligibility", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    Platform.OS = "ios";
    Location.getForegroundPermissionsAsync.mockResolvedValue({ granted: true });
    Location.getBackgroundPermissionsAsync.mockResolvedValue({ granted: true });
    Location.hasServicesEnabledAsync.mockResolvedValue(true);
    bgTask.getNativeTaskLifecycleStatus.mockResolvedValue({
      taskDefined: true,
      taskStarted: true,
    });
  });

  it("requiresLiveTrackingPermission — EN_ROUTE et IN_PROGRESS uniquement", () => {
    expect(requiresLiveTrackingPermission("EN_ROUTE")).toBe(true);
    expect(requiresLiveTrackingPermission("IN_PROGRESS")).toBe(true);
    expect(requiresLiveTrackingPermission("COMPLETED")).toBe(false);
    expect(requiresLiveTrackingPermission("ARRIVED")).toBe(false);
  });

  it("hasMissionTrackingCapability iOS — sans bg → false", () => {
    expect(
      hasMissionTrackingCapability({
        fgGranted: true,
        bgGranted: false,
        gpsEnabled: true,
        foregroundServiceRunning: false,
        platform: "ios",
        constraintReason: "permission_bg_denied",
      })
    ).toBe(false);
  });

  it("hasMissionTrackingCapability Android — bg OK + FGS arrêté → false", () => {
    expect(
      hasMissionTrackingCapability({
        fgGranted: true,
        bgGranted: true,
        gpsEnabled: true,
        foregroundServiceRunning: false,
        platform: "android",
        constraintReason: "fgs_not_running",
      })
    ).toBe(false);
  });

  it("hasMissionTrackingCapability Android — transition gate ignore FGS", () => {
    expect(
      hasMissionTrackingCapability(
        {
          fgGranted: true,
          bgGranted: true,
          gpsEnabled: true,
          foregroundServiceRunning: false,
          platform: "android",
          constraintReason: "fgs_not_running",
        },
        { requireForegroundService: false }
      )
    ).toBe(true);
  });

  it("evaluateMissionTrackingCapability — forLiveTransition capable sans FGS Android", async () => {
    Platform.OS = "android";
    bgTask.getNativeTaskLifecycleStatus.mockResolvedValue({
      taskDefined: true,
      taskStarted: false,
    });

    const result = await evaluateMissionTrackingCapability({ forLiveTransition: true });
    expect(result.capable).toBe(true);
    expect(result.foregroundServiceRunning).toBe(false);
  });

  it("evaluateMissionTrackingCapability — status granted sans boolean granted", async () => {
    Platform.OS = "android";
    Location.getForegroundPermissionsAsync.mockResolvedValue({ status: "granted" });
    Location.getBackgroundPermissionsAsync.mockResolvedValue({ status: "granted" });
    bgTask.getNativeTaskLifecycleStatus.mockResolvedValue({
      taskDefined: true,
      taskStarted: true,
    });

    const result = await evaluateMissionTrackingCapability({ forLiveTransition: false });
    expect(result.fgGranted).toBe(true);
    expect(result.bgGranted).toBe(true);
    expect(result.capable).toBe(true);
  });

  it("evaluateMissionTrackingCapability — en mission Android sans FGS → non capable", async () => {
    Platform.OS = "android";
    bgTask.getNativeTaskLifecycleStatus.mockResolvedValue({
      taskDefined: true,
      taskStarted: false,
    });

    const result = await evaluateMissionTrackingCapability({ forLiveTransition: false });
    expect(result.capable).toBe(false);
    expect(result.constraintReason).toBe("fgs_not_running");
  });
});
