import React from "react";
import { act, create } from "react-test-renderer";

import { useMissionLiveTrackingGuard } from "./useMissionLiveTrackingGuard";

jest.mock("../../../core/featureFlags/registry", () => ({
  isFeatureEnabled: (key: string) =>
    key === "driver_mission_live_tracking_guard_enabled" ||
    key === "tracking_background_enabled",
}));

jest.mock("../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: jest.fn(),
}));

jest.mock("../services/missionLiveTrackingEligibility", () => ({
  requiresLiveTrackingPermission: (target: string) =>
    target === "EN_ROUTE" || target === "IN_PROGRESS",
  evaluateMissionTrackingCapability: jest.fn(),
}));

jest.mock("../services/trackingReadinessPersistence", () => ({
  markTrackingOnboarded: jest.fn(),
  setTrackingNeedsAttention: jest.fn(),
}));

jest.mock("expo-location", () => ({
  requestForegroundPermissionsAsync: jest.fn(),
  requestBackgroundPermissionsAsync: jest.fn(),
}));

const eligibility = jest.requireMock("../services/missionLiveTrackingEligibility") as {
  evaluateMissionTrackingCapability: jest.Mock;
};

function Probe(props: { onReady: (api: ReturnType<typeof useMissionLiveTrackingGuard>) => void }) {
  const api = useMissionLiveTrackingGuard();
  React.useEffect(() => {
    props.onReady(api);
  }, [api, props]);
  return null;
}

describe("useMissionLiveTrackingGuard", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    eligibility.evaluateMissionTrackingCapability.mockResolvedValue({
      capable: false,
      constraintReason: "permission_bg_denied",
      fgGranted: true,
      bgGranted: false,
      gpsEnabled: true,
      foregroundServiceRunning: false,
      platform: "ios",
    });
  });

  it("ouvre la modale pour EN_ROUTE si non capable", async () => {
    let api: ReturnType<typeof useMissionLiveTrackingGuard> | null = null;
    const onProceed = jest.fn();

    act(() => {
      create(<Probe onReady={(value) => { api = value; }} />);
    });

    await act(async () => {
      api!.guardTransition({
        missionId: 42,
        target: "EN_ROUTE",
        onProceed,
      });
      await Promise.resolve();
    });

    expect(api!.disclosureVisible).toBe(true);
    expect(onProceed).not.toHaveBeenCalled();
  });

  it("appelle onProceed directement pour ARRIVED", async () => {
    eligibility.evaluateMissionTrackingCapability.mockResolvedValue({
      capable: true,
      constraintReason: null,
    });

    let api: ReturnType<typeof useMissionLiveTrackingGuard> | null = null;
    const onProceed = jest.fn();

    act(() => {
      create(<Probe onReady={(value) => { api = value; }} />);
    });

    await act(async () => {
      api!.guardTransition({
        missionId: 42,
        target: "ARRIVED",
        onProceed,
      });
      await Promise.resolve();
    });

    expect(onProceed).toHaveBeenCalled();
    expect(api!.disclosureVisible).toBe(false);
  });
});
