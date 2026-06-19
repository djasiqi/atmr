import { AppState, Platform } from "react-native";
import * as Updates from "expo-updates";
import type { PropsWithChildren } from "../reactCompat";
import { useCallback, useEffect, useRef, useState } from "../reactCompat";
import { emitDriverTelemetry } from "../observability/driverTelemetry";
import {
  evaluateOtaAutoReload,
  isOtaAutoReloadFeatureEnabled,
  OTA_AUTO_RELOAD_STARTUP_DELAY_MS,
} from "./otaAutoReloadPolicy";
import { isOtaAutoReloadMissionBlocking } from "./otaAutoReloadMissionGuard";
import { reloadPendingOtaUpdate } from "./otaUpdateActions";
import { useExpoUpdatesState } from "./useExpoUpdatesState";

const SOURCE = "core.version.OtaAutoReloadProvider";

function isOtaRuntimeSupported(): boolean {
  return Platform.OS !== "web" && Updates.isEnabled && !__DEV__;
}

export function OtaAutoReloadProvider({ children }: PropsWithChildren) {
  const { isUpdatePending, isUpdateAvailable } = useExpoUpdatesState();
  const [startupReady, setStartupReady] = useState(false);
  const reloadConsumedRef = useRef(false);
  const fetchInFlightRef = useRef(false);

  useEffect(() => {
    const timer = setTimeout(() => setStartupReady(true), OTA_AUTO_RELOAD_STARTUP_DELAY_MS);
    return () => clearTimeout(timer);
  }, []);

  const tryAutoReload = useCallback(
    async (trigger: string) => {
      if (!isOtaRuntimeSupported() || !isOtaAutoReloadFeatureEnabled()) {
        return;
      }

      const evaluation = evaluateOtaAutoReload({
        updatesEnabled: Updates.isEnabled,
        isDev: __DEV__,
        appState: AppState.currentState,
        missionBlocking: isOtaAutoReloadMissionBlocking(),
        reloadConsumedThisSession: reloadConsumedRef.current,
        startupReady,
        isUpdatePending,
      });

      if (!evaluation.allowed) {
        if (evaluation.deferReason) {
          emitDriverTelemetry("ota.auto_reload.deferred", {
            source: SOURCE,
            reason: evaluation.deferReason,
            trigger,
            update_id: Updates.updateId ?? null,
          });
        }
        return;
      }

      reloadConsumedRef.current = true;
      emitDriverTelemetry("ota.auto_reload.start", {
        source: SOURCE,
        trigger,
        update_id: Updates.updateId ?? null,
      });

      const result = await reloadPendingOtaUpdate();
      if (result === "failed") {
        reloadConsumedRef.current = false;
        emitDriverTelemetry("ota.auto_reload.failed", {
          source: SOURCE,
          trigger,
          reason: "reload_async_failed",
          update_id: Updates.updateId ?? null,
        });
      }
    },
    [isUpdatePending, startupReady]
  );

  useEffect(() => {
    if (!isUpdatePending) return;
    emitDriverTelemetry("ota.auto_reload.pending_detected", {
      source: SOURCE,
      update_id: Updates.updateId ?? null,
    });
    void tryAutoReload("isUpdatePending");
  }, [isUpdatePending, tryAutoReload]);

  useEffect(() => {
    if (!isOtaRuntimeSupported() || !isOtaAutoReloadFeatureEnabled()) return;
    if (!startupReady || !isUpdateAvailable || isUpdatePending || fetchInFlightRef.current) {
      return;
    }

    fetchInFlightRef.current = true;
    void (async () => {
      try {
        await Updates.fetchUpdateAsync();
      } catch {
        emitDriverTelemetry("ota.auto_reload.failed", {
          source: SOURCE,
          trigger: "fetch_update",
          reason: "fetch_update_failed",
          update_id: Updates.updateId ?? null,
        });
      } finally {
        fetchInFlightRef.current = false;
      }
    })();
  }, [isUpdateAvailable, isUpdatePending, startupReady]);

  useEffect(() => {
    if (!isOtaRuntimeSupported() || !isOtaAutoReloadFeatureEnabled()) return;

    const subscription = AppState.addEventListener("change", (nextState) => {
      if (nextState !== "active") return;
      if (!isUpdatePending && !isUpdateAvailable) return;
      void tryAutoReload("app_foreground");
    });

    return () => subscription.remove();
  }, [isUpdateAvailable, isUpdatePending, tryAutoReload]);

  return children;
}
