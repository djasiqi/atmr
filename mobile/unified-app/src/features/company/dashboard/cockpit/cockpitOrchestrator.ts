import { resolveCameraPolicy, type CameraPolicy } from "./cameraPolicyManager";
import { resolveCartographicPlan, type CartographicRenderPlan } from "./cartographicHierarchy";
import { arbitrateCockpitFrame } from "./cockpitFrameArbitration";
import type { CockpitEvent } from "./cockpitEventBus";
import {
  createFsmSnapshot,
  reduceCockpitFsm,
  type CockpitFsmSnapshot,
  type CockpitFsmState,
} from "./cockpitFiniteStateMachine";
import { COCKPIT_MOTION_TOKENS } from "./cockpitMotionTokens";
import {
  canApplyStabilityAction,
  createStabilityBudgetState,
  recordStabilityAction,
  type StabilityBudgetState,
} from "./cockpitStabilityBudget";
import { shouldDeferRendering } from "./criticalRenderingPaths";
import { resolveFleetAttention, type AttentionLevel } from "./fleetAttentionSystem";
import { resolveFleetHealthScore } from "./healthScoreModel";
import { buildImminentDepartures, type ImminentDeparturesResult } from "./imminentDepartures";
import {
  computeMapDensityPolicy,
  resolveDensityLevel,
  type DensityLevel,
  type MapDensityPolicy,
} from "./mapDensityGovernor";
import { resolveMapIntent, type MapIntent } from "./mapIntentResolver";
import {
  motionAllowsDecorativeGlow,
  motionAllowsIncidentPulse,
  motionAllowsSelectionPulse,
  resolveMotionPolicy,
  type MotionPolicyState,
} from "./motionPolicy";
import {
  resolveReducedComplexityEffects,
  shouldEnableReducedComplexity,
  type ReducedComplexityEffects,
} from "./reducedComplexityMode";
import { capVisibleRouteCount } from "./semanticRouteSystem";
import type { CockpitGovernanceInput } from "./cockpitGovernance";
import { computeCockpitUiState } from "./cockpitGovernance";
import type { CockpitUiState } from "./cockpitTypes";
import { recordCockpitMetric } from "./cockpitMetrics";
import type { CompanyDispatchMission } from "../../api/contracts";

export type CockpitOrchestratorInput = CockpitGovernanceInput & {
  driverCount: number;
  unassignedCount?: number;
  criticalEtaCount?: number;
  zoomLatitudeDelta?: number | null;
  lowPowerMode?: boolean;
  fpsBelowThreshold?: boolean;
  manualRecenterOnly?: boolean;
  userGestureActive?: boolean;
  missions?: CompanyDispatchMission[];
  frameEvents?: CockpitEvent[];
  fsmSnapshot?: CockpitFsmSnapshot;
  stabilityState?: StabilityBudgetState;
  nowMs?: number;
};

export type CockpitOrchestrationDecision = {
  uiState: CockpitUiState;
  fsm: CockpitFsmSnapshot;
  fsmState: CockpitFsmState;
  mapIntent: MapIntent;
  attentionLevel: AttentionLevel;
  density: MapDensityPolicy;
  motionPolicy: MotionPolicyState;
  cameraPolicy: CameraPolicy;
  cartographic: CartographicRenderPlan;
  reducedComplexity: boolean;
  reducedEffects: ReducedComplexityEffects;
  fleetHealthScore: number;
  maxVisibleRoutes: number;
  globalVectorMode: boolean;
  imminentDepartures: ImminentDeparturesResult;
  routeFadeMs: number;
  stabilityState: StabilityBudgetState;
  allowSelectionPulse: boolean;
  allowIncidentPulse: boolean;
  allowDecorativeGlow: boolean;
  deferSecondaryOverlays: boolean;
  lastStabilityBypass?: string;
};

function applyFrameEventsToFsm(
  snapshot: CockpitFsmSnapshot,
  events: CockpitEvent[]
): CockpitFsmSnapshot {
  let next = snapshot;
  for (const event of events) {
    if (event.type === "DRIVER_SELECTED") {
      next = reduceCockpitFsm(next, { type: "DRIVER_SELECT" });
    } else if (event.type === "DRIVER_CLEARED") {
      next = reduceCockpitFsm(next, { type: "DRIVER_CLEAR" });
    } else if (event.type === "SEARCH_OPENED") {
      next = reduceCockpitFsm(next, { type: "SEARCH_OPEN" });
    } else if (event.type === "SEARCH_CLEARED") {
      next = reduceCockpitFsm(next, { type: "SEARCH_CLOSE" });
    } else if (event.type === "INCIDENT_ACK") {
      next = reduceCockpitFsm(next, { type: "INCIDENT_ACK" });
    } else if (event.type === "FILTERS_OPENED") {
      next = reduceCockpitFsm(next, { type: "DISPATCH_OPEN" });
    } else if (event.type === "FILTERS_CLOSED") {
      next = reduceCockpitFsm(next, { type: "DISPATCH_CLOSE" });
    }
  }
  return next;
}

function driverCountForDensityLevel(level: DensityLevel): number {
  switch (level) {
    case "low":
      return 8;
    case "medium":
      return 18;
    case "high":
      return 25;
    case "extreme":
      return 55;
    case "aggregate":
      return 110;
    default:
      return 8;
  }
}

function stabilizeDensity(
  candidate: MapDensityPolicy,
  driverCount: number,
  zoomLatitudeDelta: number | null | undefined,
  stability: StabilityBudgetState,
  nowMs: number,
  canBypass: boolean
): { density: MapDensityPolicy; stability: StabilityBudgetState; bypass?: string } {
  const rawLevel = resolveDensityLevel(driverCount, zoomLatitudeDelta);
  const prevLevel = stability.lastDensityLevel;
  if (prevLevel == null || prevLevel === rawLevel) {
    return {
      density: candidate,
      stability: { ...stability, lastDensityLevel: rawLevel },
    };
  }
  if (canBypass || canApplyStabilityAction(stability, "density", nowMs, canBypass)) {
    return {
      density: candidate,
      stability: recordStabilityAction({ ...stability, lastDensityLevel: rawLevel }, "density", nowMs),
    };
  }
  const held = computeMapDensityPolicy({
    driverCount: driverCountForDensityLevel(prevLevel),
    zoomLatitudeDelta,
  });
  return {
    density: { ...held, level: prevLevel },
    stability,
    bypass: "density_flip_blocked",
  };
}

/** Source unique de vérité pour décisions cockpit — managers = calcul pur, pas de side effects. */
export function resolveCockpitOrchestration(
  input: CockpitOrchestratorInput
): CockpitOrchestrationDecision {
  const nowMs = input.nowMs ?? Date.now();
  let stability = input.stabilityState ?? createStabilityBudgetState(nowMs);
  let lastStabilityBypass: string | undefined;

  const frameResult = arbitrateCockpitFrame(input.frameEvents ?? [], () => input);
  const effectiveInput = frameResult.decision;
  const frameEvents = frameResult.events;

  const fleetHealthScore = resolveFleetHealthScore({
    delayedCount: effectiveInput.delayedCount,
    urgentCount: effectiveInput.urgentCount,
    unassignedCount: effectiveInput.unassignedCount ?? 0,
    criticalEtaCount: effectiveInput.criticalEtaCount ?? 0,
    realtimeStatus: effectiveInput.realtimeStatus,
    realtimeDataFreshness: effectiveInput.realtimeDataFreshness,
    policyFailureCount: effectiveInput.policyFailureCount,
    interactionBurstPerMinute: effectiveInput.interactionBurstPerMinute,
  });

  const attention = resolveFleetAttention({
    urgentCount: effectiveInput.urgentCount,
    delayedCount: effectiveInput.delayedCount,
    unassignedCount: effectiveInput.unassignedCount ?? 0,
    criticalEtaCount: effectiveInput.criticalEtaCount ?? 0,
    healthScore: fleetHealthScore,
  });

  const densityCandidate = computeMapDensityPolicy({
    driverCount: effectiveInput.driverCount,
    zoomLatitudeDelta: effectiveInput.zoomLatitudeDelta,
  });

  const densityStabilized = stabilizeDensity(
    densityCandidate,
    effectiveInput.driverCount,
    effectiveInput.zoomLatitudeDelta,
    stability,
    nowMs,
    attention.level === "CRITICAL"
  );
  let density = densityStabilized.density;
  stability = densityStabilized.stability;
  if (densityStabilized.bypass) lastStabilityBypass = densityStabilized.bypass;

  const baseFsm = input.fsmSnapshot ?? createFsmSnapshot();
  const fsm = applyFrameEventsToFsm(baseFsm, frameEvents);

  const mapIntent = resolveMapIntent({
    searchActive: effectiveInput.searchActive,
    filtersOpen: effectiveInput.filtersOpen,
    selectedDriverId: effectiveInput.selectedDriverId,
    driverSheetOpen: effectiveInput.driverSheetOpen,
    urgentCount: effectiveInput.urgentCount,
  });

  const reducedComplexity = shouldEnableReducedComplexity({
    densityLevel: density.level,
    healthScore: fleetHealthScore,
    lowPowerMode: effectiveInput.lowPowerMode,
    fpsBelowThreshold: effectiveInput.fpsBelowThreshold,
  });

  const motionPolicy = resolveMotionPolicy({
    densityLevel: density.level,
    attentionLevel: attention.level,
    reducedComplexity,
    safeMode: fleetHealthScore < 35,
  });

  let uiState = computeCockpitUiState({
    ...effectiveInput,
    healthScore: fleetHealthScore,
    fsmState: fsm.state,
    nowMs,
  });

  const prevMode = stability.lastUiMode;
  if (
    prevMode != null &&
    uiState.mode !== prevMode &&
    !canApplyStabilityAction(stability, "mode", nowMs, attention.canInterruptMode)
  ) {
    uiState = computeCockpitUiState({
      ...effectiveInput,
      healthScore: fleetHealthScore,
      fsmState: fsm.state,
      nowMs,
      forcedMode: prevMode,
    });
    lastStabilityBypass = lastStabilityBypass ?? "mode_flip_blocked";
  } else if (uiState.mode !== prevMode) {
    stability = recordStabilityAction({ ...stability, lastUiMode: uiState.mode }, "mode", nowMs);
  } else if (stability.lastUiMode == null) {
    stability = { ...stability, lastUiMode: uiState.mode };
  }

  const globalVectorMode =
    fsm.state === "GLOBAL_IDLE" ||
    fsm.state === "GLOBAL_TRACKING" ||
    fsm.state === "SEARCH_ACTIVE";

  const maxVisibleRoutes = capVisibleRouteCount(density, globalVectorMode);

  const cameraPolicyCandidate = resolveCameraPolicy({
    fsmState: fsm.state,
    mapIntent,
    attentionLevel: attention.level,
    manualRecenterOnly: effectiveInput.manualRecenterOnly ?? false,
    userGestureActive: effectiveInput.userGestureActive,
  });

  let cameraPolicy = cameraPolicyCandidate;
  if (
    !canApplyStabilityAction(stability, "camera", nowMs, attention.canInterruptMode) &&
    cameraPolicyCandidate !== "user_gesture_preserve"
  ) {
    cameraPolicy = "user_gesture_preserve";
    lastStabilityBypass = lastStabilityBypass ?? "camera_blocked";
  } else if (cameraPolicyCandidate !== "user_gesture_preserve") {
    stability = recordStabilityAction(stability, "camera", nowMs);
  }

  const cartographic = resolveCartographicPlan({
    fsmState: fsm.state,
    attentionLevel: attention.level,
    density,
    driverSelected: effectiveInput.selectedDriverId != null,
  });

  const imminentDepartures = buildImminentDepartures(effectiveInput.missions ?? []);

  const underLoad = density.level === "high" || density.level === "extreme" || density.level === "aggregate";

  if (__DEV__ || process.env.EXPO_PUBLIC_COCKPIT_METRICS === "1") {
    recordCockpitMetric("density_level", density.level === "low" ? 1 : density.level === "medium" ? 2 : density.level === "high" ? 3 : density.level === "extreme" ? 4 : 5);
    recordCockpitMetric("routes_visible", maxVisibleRoutes);
  }

  return {
    uiState,
    fsm,
    fsmState: fsm.state,
    mapIntent,
    attentionLevel: attention.level,
    density,
    motionPolicy,
    cameraPolicy,
    cartographic,
    reducedComplexity,
    reducedEffects: resolveReducedComplexityEffects(reducedComplexity),
    fleetHealthScore,
    maxVisibleRoutes,
    globalVectorMode,
    imminentDepartures,
    routeFadeMs: globalVectorMode
      ? COCKPIT_MOTION_TOKENS.FAST_FADE_MS
      : COCKPIT_MOTION_TOKENS.CONTEXT_FADE_MS,
    stabilityState: stability,
    allowSelectionPulse: motionAllowsSelectionPulse(motionPolicy),
    allowIncidentPulse: motionAllowsIncidentPulse(motionPolicy),
    allowDecorativeGlow: motionAllowsDecorativeGlow(motionPolicy) && !reducedComplexity,
    deferSecondaryOverlays: shouldDeferRendering("secondary_overlays", underLoad),
    lastStabilityBypass,
  };
}
