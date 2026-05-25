import type { CockpitFsmState } from "./cockpitFiniteStateMachine";
import type { AttentionLevel } from "./fleetAttentionSystem";
import type { MapDensityPolicy } from "./mapDensityGovernor";

/** Niveaux L0–L5 de hiérarchie cartographique. */
export type CartographicLevel = 0 | 1 | 2 | 3 | 4 | 5;

export type CartographicRenderPlan = {
  showBaseMapMuted: boolean;
  showImminentDepartures: boolean;
  showPassiveDrivers: boolean;
  showActiveDriverCapsule: boolean;
  showActiveRoute: boolean;
  showCriticalIncidentPulse: boolean;
  vectorModeOnly: boolean;
  maxCartographicLevel: CartographicLevel;
};

export function resolveCartographicPlan(input: {
  fsmState: CockpitFsmState;
  attentionLevel: AttentionLevel;
  density: MapDensityPolicy;
  driverSelected: boolean;
}): CartographicRenderPlan {
  const globalMode =
    input.fsmState === "GLOBAL_IDLE" ||
    input.fsmState === "GLOBAL_TRACKING" ||
    input.fsmState === "SEARCH_ACTIVE";

  const driverFocus = input.fsmState === "DRIVER_FOCUS";

  return {
    showBaseMapMuted: true,
    showImminentDepartures: globalMode && !input.density.aggregateMode,
    showPassiveDrivers: !input.density.aggregateMode || input.driverSelected,
    showActiveDriverCapsule: input.driverSelected || driverFocus,
    /** Itinéraire uniquement après sélection explicite d’un chauffeur (pas au chargement / urgence). */
    showActiveRoute: input.driverSelected,
    showCriticalIncidentPulse: input.attentionLevel === "CRITICAL",
    vectorModeOnly: globalMode,
    maxCartographicLevel: input.attentionLevel === "CRITICAL" ? 5 : driverFocus ? 4 : 3,
  };
}
