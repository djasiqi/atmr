import { isFeatureEnabled } from "../featureFlags/registry";
import { isOtaAutoReloadMissionBlocking } from "./otaAutoReloadMissionGuard";

export const OTA_AUTO_RELOAD_STARTUP_DELAY_MS = 400;

export type OtaReloadDeferReason =
  | "disabled"
  | "dev"
  | "updates_unavailable"
  | "background"
  | "startup_not_ready"
  | "active_mission"
  | "already_reloaded_session";

export type OtaAutoReloadEvaluationInput = {
  updatesEnabled: boolean;
  isDev: boolean;
  featureEnabled?: boolean;
  appState: string;
  missionBlocking?: boolean;
  reloadConsumedThisSession: boolean;
  startupReady: boolean;
  isUpdatePending: boolean;
};

export type OtaAutoReloadEvaluation = {
  allowed: boolean;
  deferReason: OtaReloadDeferReason | null;
};

export function isOtaAutoReloadFeatureEnabled(): boolean {
  return isFeatureEnabled("ota_auto_reload_enabled");
}

export function evaluateOtaAutoReload(
  input: OtaAutoReloadEvaluationInput
): OtaAutoReloadEvaluation {
  const featureEnabled = input.featureEnabled ?? isOtaAutoReloadFeatureEnabled();

  if (input.isDev) {
    return { allowed: false, deferReason: "dev" };
  }
  if (!input.updatesEnabled) {
    return { allowed: false, deferReason: "updates_unavailable" };
  }
  if (!featureEnabled) {
    return { allowed: false, deferReason: "disabled" };
  }
  if (!input.isUpdatePending) {
    return { allowed: false, deferReason: null };
  }
  if (input.reloadConsumedThisSession) {
    return { allowed: false, deferReason: "already_reloaded_session" };
  }
  if (!input.startupReady) {
    return { allowed: false, deferReason: "startup_not_ready" };
  }
  if (input.appState !== "active") {
    return { allowed: false, deferReason: "background" };
  }
  const missionBlocking = input.missionBlocking ?? isOtaAutoReloadMissionBlocking();
  if (missionBlocking) {
    return { allowed: false, deferReason: "active_mission" };
  }

  return { allowed: true, deferReason: null };
}
