/**
 * État observable du suivi GPS côté UI chauffeur.
 * Plan GPS background : Suivi actif / limité / désactivé / permission requise.
 */

import { useState, useEffect, useCallback } from "react";
import {
  buildBgTrackingInputs,
  getTrackingRuntimeState,
  type BgTrackingInputs,
} from "@/services/locationTracker";
import { MissionStateManager } from "@/services/missionState";

export type TrackingDisplayState =
  | "active"
  | "limited"
  | "permission_required"
  | "disabled";

export interface TrackingState {
  /** État affiché pour le chauffeur */
  displayState: TrackingDisplayState;
  /** Suivi background actif (positions en arrière-plan) */
  isBackgroundActive: boolean;
  /** Mission active (ASSIGNED | EN_ROUTE | IN_PROGRESS) */
  hasActiveMission: boolean;
  /** Permission foreground accordée */
  fgGranted: boolean;
  /** Permission background accordée */
  bgGranted: boolean;
  /** Raison du dernier start/stop (diagnostic) */
  lastReason?: string;
}

const EMPTY_STATE: TrackingState = {
  displayState: "disabled",
  isBackgroundActive: false,
  hasActiveMission: false,
  fgGranted: false,
  bgGranted: false,
};

/**
 * Dérive l'état d'affichage depuis les inputs et le runtime.
 */
function deriveDisplayState(
  inputs: BgTrackingInputs,
  bgStarted: boolean
): TrackingDisplayState {
  if (inputs.role !== "driver" || !inputs.isAuthenticated) return "disabled";
  if (!inputs.hasActiveMission) return "disabled";

  if (inputs.fgPermission !== "granted") return "permission_required";
  if (inputs.bgPermission !== "granted") {
    // Foreground ok, background refusé → fallback foreground-only
    return "limited";
  }
  if (inputs.killSwitchEnabled) return "disabled";

  return bgStarted ? "active" : "disabled";
}

/**
 * Hook exposant l'état du suivi GPS pour l'UI chauffeur.
 * Rafraîchit à chaque changement de mission ou permissions.
 */
export function useTrackingState(params: {
  isDriverAuthenticated: boolean;
  role: "driver" | "enterprise";
}): TrackingState {
  const { isDriverAuthenticated, role } = params;
  const [state, setState] = useState<TrackingState>(EMPTY_STATE);

  const refresh = useCallback(async () => {
    if (role !== "driver" || !isDriverAuthenticated) {
      setState(EMPTY_STATE);
      return;
    }
    try {
      const inputs = await buildBgTrackingInputs({
        isAuthenticated: isDriverAuthenticated,
        role,
        hasActiveMission: MissionStateManager.isActive(),
      });
      const runtime = getTrackingRuntimeState();
      const displayState = deriveDisplayState(inputs, runtime.started);
      setState({
        displayState,
        isBackgroundActive: runtime.started,
        hasActiveMission: inputs.hasActiveMission,
        fgGranted: inputs.fgPermission === "granted",
        bgGranted: inputs.bgPermission === "granted",
        lastReason: runtime.lastReason,
      });
    } catch {
      setState(EMPTY_STATE);
    }
  }, [isDriverAuthenticated, role]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (role !== "driver") return;
    const unsub = MissionStateManager.subscribe((event) => {
      if (
        event === "mission_started" ||
        event === "mission_stopped" ||
        event === "state_changed"
      ) {
        refresh();
      }
    });
    return unsub;
  }, [role, refresh]);

  return state;
}
