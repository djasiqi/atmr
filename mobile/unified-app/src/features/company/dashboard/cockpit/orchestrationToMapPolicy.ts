import type { CockpitMapPolicy } from "../../components/maps/fleetMapTypes";
import type { CockpitOrchestrationDecision } from "./cockpitOrchestrator";

/** Map orchestration decision → map render policy (no logic). */
export function orchestrationToMapPolicy(
  orchestration: CockpitOrchestrationDecision
): CockpitMapPolicy {
  return {
    maxVisibleRoutes: orchestration.maxVisibleRoutes,
    globalVectorMode: orchestration.globalVectorMode,
    showImminentDepartures: orchestration.cartographic.showImminentDepartures,
    showPassiveDrivers: orchestration.cartographic.showPassiveDrivers,
    showActiveRoute: orchestration.cartographic.showActiveRoute,
    routeFadeMs: orchestration.routeFadeMs,
    allowDecorativeGlow: orchestration.allowDecorativeGlow,
    simplifyMarkers: orchestration.density.simplifyMarkers,
    cameraPolicy: orchestration.cameraPolicy,
  };
}
