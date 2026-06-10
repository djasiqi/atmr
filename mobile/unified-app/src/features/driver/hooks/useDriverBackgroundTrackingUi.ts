import { useCallback, useEffect, useState } from "react";
import { AppState } from "react-native";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { describeBackgroundRuntime } from "../services/backgroundRuntimeCompat";
import { getNativeTaskLifecycleStatus } from "../services/backgroundLocationTask";
import { evaluateMissionTrackingCapability } from "../services/missionLiveTrackingEligibility";
import {
  getTrackingRuntimeSnapshot,
  subscribeTrackingRuntime,
} from "../services/trackingRuntime";
import { getTrackingSnapshot, subscribeTrackingSnapshot } from "../tracking";
import { isTrackingActiveStatus } from "../domain/status";

export type DriverBackgroundTrackingUiState = {
  showBanner: boolean;
  bannerKind: "permission_required" | "background_unavailable" | null;
  taskDefined: boolean;
  taskStarted: boolean;
  bgFlagEnabled: boolean;
  runtime: string;
  lastNativeStartError: string | null;
  lastTaskInvokedAt: number | null;
  pendingFgsStart: boolean;
  nativeStartPhase: string | null;
  nativeStartError: string | null;
  nativeTaskDefined: boolean | null;
  nativeStartedBefore: boolean | null;
  nativeStartedAfter: boolean | null;
};

const EMPTY: DriverBackgroundTrackingUiState = {
  showBanner: false,
  bannerKind: null,
  taskDefined: false,
  taskStarted: false,
  bgFlagEnabled: false,
  runtime: describeBackgroundRuntime(),
  lastNativeStartError: null,
  lastTaskInvokedAt: null,
  pendingFgsStart: false,
  nativeStartPhase: null,
  nativeStartError: null,
  nativeTaskDefined: null,
  nativeStartedBefore: null,
  nativeStartedAfter: null,
};

export function isTrackingQaPanelEnabled(): boolean {
  return (
    process.env.EXPO_PUBLIC_TRACKING_QA_PANEL === "1" ||
    process.env.EXPO_PUBLIC_TRACKING_QA_PANEL === "true"
  );
}

async function loadDiagnostics(): Promise<DriverBackgroundTrackingUiState> {
  const tracking = getTrackingSnapshot();
  const runtime = getTrackingRuntimeSnapshot();
  const lifecycle = await getNativeTaskLifecycleStatus();
  const missionActive =
    tracking.missionId != null &&
    tracking.missionStatus != null &&
    isTrackingActiveStatus(tracking.missionStatus);

  const capability = missionActive
    ? await evaluateMissionTrackingCapability({ forLiveTransition: false })
    : null;

  let showBanner = false;
  let bannerKind: DriverBackgroundTrackingUiState["bannerKind"] = null;

  if (missionActive && capability && !capability.capable) {
    showBanner = true;
    if (
      capability.constraintReason === "permission_bg_denied" ||
      capability.constraintReason === "permission_fg_denied"
    ) {
      bannerKind = "permission_required";
    } else {
      bannerKind = "background_unavailable";
    }
  }

  return {
    showBanner,
    bannerKind,
    taskDefined: lifecycle.taskDefined,
    taskStarted: lifecycle.taskStarted,
    bgFlagEnabled: isFeatureEnabled("tracking_background_enabled"),
    runtime: describeBackgroundRuntime(),
    lastNativeStartError: runtime.lastNativeStartError,
    lastTaskInvokedAt: runtime.lastTaskInvokedAt,
    pendingFgsStart: runtime.pendingFgsStart.active,
    nativeStartPhase: runtime.nativeStartDiagnostics.native_start_phase,
    nativeStartError:
      runtime.nativeStartDiagnostics.native_start_error ?? runtime.lastNativeStartError,
    nativeTaskDefined: runtime.nativeStartDiagnostics.native_task_defined,
    nativeStartedBefore: runtime.nativeStartDiagnostics.native_started_before,
    nativeStartedAfter: runtime.nativeStartDiagnostics.native_started_after,
  };
}

export function useDriverBackgroundTrackingUi(): DriverBackgroundTrackingUiState {
  const [ui, setUi] = useState<DriverBackgroundTrackingUiState>(EMPTY);

  const refresh = useCallback(async () => {
    const next = await loadDiagnostics();
    setUi(next);
    return next;
  }, []);

  useEffect(() => {
    void refresh();

    const unsubTracking = subscribeTrackingSnapshot(() => {
      void refresh();
    });
    const unsubRuntime = subscribeTrackingRuntime(() => {
      void refresh();
    });
    const appSub = AppState.addEventListener("change", () => {
      void refresh();
    });

    let interval: ReturnType<typeof setInterval> | null = null;

    const armFallback = async () => {
      const state = await loadDiagnostics();
      setUi(state);
      if (interval) {
        clearInterval(interval);
        interval = null;
      }
      if (!state.showBanner) return;
      interval = setInterval(() => {
        void refresh();
      }, 45_000);
    };

    void armFallback();

    return () => {
      unsubTracking();
      unsubRuntime();
      appSub.remove();
      if (interval) clearInterval(interval);
    };
  }, [refresh]);

  return ui;
}
