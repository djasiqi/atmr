import { useCallback, useEffect, useState } from "react";
import { AppState } from "react-native";

import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { evaluateTrackingReadiness } from "../components/DriverTrackingReadinessGate";
import {
  readTrackingNeedsAttention,
  readTrackingOnboarded,
  setTrackingNeedsAttention,
} from "../services/trackingReadinessPersistence";

export function useTrackingAttentionState() {
  const [trackingReady, setTrackingReady] = useState(false);
  const [trackingOnboarded, setTrackingOnboarded] = useState<boolean | null>(null);
  const [trackingNeedsAttention, setTrackingNeedsAttentionState] = useState(false);
  const [panelDismissed, setPanelDismissed] = useState(false);
  const trackingBackgroundEnabled = isFeatureEnabled("tracking_background_enabled");

  const refresh = useCallback(async () => {
    if (!trackingBackgroundEnabled) {
      setTrackingOnboarded(true);
      setTrackingReady(true);
      setTrackingNeedsAttentionState(false);
      return;
    }
    const [onboarded, needsAttention, readiness] = await Promise.all([
      readTrackingOnboarded(),
      readTrackingNeedsAttention(),
      evaluateTrackingReadiness(),
    ]);
    setTrackingOnboarded(onboarded);
    setTrackingNeedsAttentionState(needsAttention);
    setTrackingReady(readiness.ready);
    if (!readiness.ready) {
      await setTrackingNeedsAttention(true);
      setTrackingNeedsAttentionState(true);
    } else if (needsAttention) {
      await setTrackingNeedsAttention(false);
      setTrackingNeedsAttentionState(false);
    }
  }, [trackingBackgroundEnabled]);

  useEffect(() => {
    void refresh();
    const sub = AppState.addEventListener("change", (next) => {
      if (next === "active") void refresh();
    });
    return () => sub.remove();
  }, [refresh]);

  const showPedagogicalPanel =
    trackingBackgroundEnabled &&
    !panelDismissed &&
    ((!trackingOnboarded && !trackingReady) || trackingNeedsAttention);

  return {
    trackingReady,
    trackingOnboarded,
    trackingNeedsAttention,
    trackingBackgroundEnabled,
    showPedagogicalPanel,
    refreshTrackingAttention: refresh,
    dismissPedagogicalPanel: () => setPanelDismissed(true),
    onReadinessGateReady: (ready: boolean) => {
      setTrackingReady(ready);
      if (ready) {
        setTrackingNeedsAttentionState(false);
        void setTrackingNeedsAttention(false).catch(() => undefined);
      }
    },
  };
}
