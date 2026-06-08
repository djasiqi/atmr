import { useCallback, useRef, useState } from "react";
import { Linking, Platform } from "react-native";
import * as Location from "expo-location";

import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import type { DriverTransitionStatus } from "../types";
import {
  evaluateMissionTrackingCapability,
  requiresLiveTrackingPermission,
} from "../services/missionLiveTrackingEligibility";
import {
  markLiveTrackingDisclosureAccepted,
} from "../services/liveTrackingDisclosureSession";
import {
  markTrackingOnboarded,
  setTrackingNeedsAttention,
} from "../services/trackingReadinessPersistence";

export type MissionLiveTrackingGuardState = {
  disclosureVisible: boolean;
  disclosurePending: boolean;
  showOpenSettings: boolean;
  pendingMissionId: number | null;
  pendingTarget: DriverTransitionStatus | null;
};

const INITIAL: MissionLiveTrackingGuardState = {
  disclosureVisible: false,
  disclosurePending: false,
  showOpenSettings: false,
  pendingMissionId: null,
  pendingTarget: null,
};

async function requestMissionTrackingPermissions(): Promise<{
  fgGranted: boolean;
  bgGranted: boolean;
}> {
  const fg = await Location.requestForegroundPermissionsAsync().catch(() => ({
    granted: false,
  }));
  if (!fg.granted) {
    return { fgGranted: false, bgGranted: false };
  }
  if (
    !isFeatureEnabled("tracking_background_enabled") ||
    typeof Location.requestBackgroundPermissionsAsync !== "function"
  ) {
    return { fgGranted: true, bgGranted: false };
  }
  const bg = await Location.requestBackgroundPermissionsAsync().catch(() => ({
    granted: false,
  }));
  return { fgGranted: true, bgGranted: Boolean(bg.granted) };
}

export function useMissionLiveTrackingGuard() {
  const [state, setState] = useState<MissionLiveTrackingGuardState>(INITIAL);
  const permissionRequestedThisAttemptRef = useRef(false);
  const proceedRef = useRef<(() => void) | null>(null);

  const closeDisclosure = useCallback(() => {
    setState(INITIAL);
    permissionRequestedThisAttemptRef.current = false;
    proceedRef.current = null;
  }, []);

  const runAfterCapability = useCallback(
    async (missionId: number, target: DriverTransitionStatus, onProceed: () => void) => {
      const capability = await evaluateMissionTrackingCapability({ forLiveTransition: true });
      if (capability.capable) {
        await setTrackingNeedsAttention(false);
        await markTrackingOnboarded().catch(() => undefined);
        onProceed();
        closeDisclosure();
        return;
      }

      emitDriverTelemetry("tracking.transition_blocked_permission", {
        source: "driver.mission_live_tracking_guard",
        mission_id: missionId,
        target_status: target,
        constraint_reason: capability.constraintReason,
      });

      await setTrackingNeedsAttention(true);

      setState((prev) => ({
        ...prev,
        disclosurePending: false,
        showOpenSettings: true,
      }));
    },
    [closeDisclosure]
  );

  const guardTransition = useCallback(
    (params: {
      missionId: number;
      target: DriverTransitionStatus;
      onProceed: () => void;
    }) => {
      const { missionId, target, onProceed } = params;

      if (
        !isFeatureEnabled("driver_mission_live_tracking_guard_enabled") ||
        !isFeatureEnabled("tracking_background_enabled")
      ) {
        onProceed();
        return;
      }

      if (!requiresLiveTrackingPermission(target)) {
        onProceed();
        return;
      }

      void (async () => {
        const capability = await evaluateMissionTrackingCapability({ forLiveTransition: true });
        if (capability.capable) {
          await setTrackingNeedsAttention(false);
          onProceed();
          return;
        }

        proceedRef.current = onProceed;
        permissionRequestedThisAttemptRef.current = false;

        emitDriverTelemetry("tracking.mission_live_guard.disclosure_shown", {
          source: "driver.mission_live_tracking_guard",
          mission_id: missionId,
          target_status: target,
          platform: Platform.OS,
        });

        setState({
          disclosureVisible: true,
          disclosurePending: false,
          showOpenSettings: false,
          pendingMissionId: missionId,
          pendingTarget: target,
        });
      })();
    },
    []
  );

  const onDisclosureContinue = useCallback(() => {
    const missionId = state.pendingMissionId;
    const target = state.pendingTarget;
    const onProceed = proceedRef.current;
    if (missionId == null || target == null || !onProceed) return;

    if (permissionRequestedThisAttemptRef.current) {
      void runAfterCapability(missionId, target, onProceed);
      return;
    }

    setState((prev) => ({ ...prev, disclosurePending: true }));
    markLiveTrackingDisclosureAccepted();

    void (async () => {
      const perms = await requestMissionTrackingPermissions();
      permissionRequestedThisAttemptRef.current = true;

      emitDriverTelemetry("tracking.mission_live_guard.permission_requested", {
        source: "driver.mission_live_tracking_guard",
        mission_id: missionId,
        target_status: target,
        platform: Platform.OS,
        fg_granted: perms.fgGranted,
        bg_granted: perms.bgGranted,
      });

      await runAfterCapability(missionId, target, onProceed);
    })();
  }, [runAfterCapability, state.pendingMissionId, state.pendingTarget]);

  const onDisclosureOpenSettings = useCallback(() => {
    if (Platform.OS === "ios") {
      void Linking.openURL("app-settings:");
    } else {
      void Linking.openSettings();
    }
  }, []);

  return {
    guardTransition,
    disclosureVisible: state.disclosureVisible,
    disclosurePending: state.disclosurePending,
    showOpenSettings: state.showOpenSettings,
    onDisclosureCancel: closeDisclosure,
    onDisclosureContinue,
    onDisclosureOpenSettings,
  };
}
