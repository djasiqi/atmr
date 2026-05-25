import {
  COCKPIT_MODE_PROFILES,
  countMajorOverlays,
  mergeOverlayVisibility,
} from "./cockpitModeProfiles";
import type { CockpitFsmState } from "./cockpitFiniteStateMachine";
import type {
  CockpitHumanOverride,
  CockpitOverlayVisibility,
  CockpitRuntimeSignals,
  CockpitUiMode,
  CockpitUiState,
} from "./cockpitTypes";
import {
  COCKPIT_INITIAL_OVERRIDE,
  DEFAULT_OVERLAY_VISIBILITY,
} from "./cockpitTypes";
import { resolveCockpitLiveStatus } from "./cockpitLiveStatus";
import { getCockpitAdvancedFlags } from "./cockpitFeatureFlags";
import { isCompanyRealtimeSocketExpected } from "../../../../core/featureFlags/registry";
import { formatCompanyActivityHint } from "../../realtime/companyRealtimePresentation";
import type { CompanyDataFreshness } from "../../realtime/companyRealtimeState";

import {
  COCKPIT_CHAOS_DELAYED_THRESHOLD,
  COCKPIT_CHAOS_URGENT_THRESHOLD,
  COCKPIT_HEALTH_SAFE_THRESHOLD,
} from "./cockpitThresholds";

const CHAOS_URGENT_THRESHOLD = COCKPIT_CHAOS_URGENT_THRESHOLD;
const CHAOS_DELAYED_THRESHOLD = COCKPIT_CHAOS_DELAYED_THRESHOLD;
const SAFE_HEALTH_THRESHOLD = COCKPIT_HEALTH_SAFE_THRESHOLD;
const STABILIZATION_MS = 600;

export type CockpitGovernanceInput = CockpitRuntimeSignals & {
  override?: Partial<CockpitHumanOverride>;
  nowMs?: number;
  bootMs?: number;
  fsmState?: CockpitFsmState;
  forcedMode?: CockpitUiMode;
};

function resolveMode(
  signals: CockpitRuntimeSignals,
  override: CockpitHumanOverride,
  fsmState?: CockpitFsmState
): {
  mode: CockpitUiMode;
  reason: string;
} {
  if (override.killSwitch) {
    return { mode: "baseline", reason: "kill_switch" };
  }
  if (signals.healthScore < SAFE_HEALTH_THRESHOLD) {
    return { mode: "safe_mode", reason: "health_score_low" };
  }
  if (
    signals.urgentCount >= CHAOS_URGENT_THRESHOLD ||
    signals.delayedCount >= CHAOS_DELAYED_THRESHOLD
  ) {
    return { mode: "chaos_overload", reason: "event_overload" };
  }
  if (signals.navigationActive) {
    return { mode: "navigation_active", reason: "navigation_active" };
  }
  if (signals.filtersOpen || signals.layersOpen) {
    return { mode: "filters", reason: "filters_open" };
  }
  if (signals.searchActive) {
    return { mode: "search", reason: "search_active" };
  }
  if (signals.urgentCount > 0 && !override.autoFocusOff) {
    return { mode: "focus_urgent", reason: "urgent_present" };
  }
  if (signals.selectedMissionId != null && !override.autoFocusOff) {
    return { mode: "focus_mission", reason: "mission_selected" };
  }
  if (
    fsmState === "DRIVER_FOCUS" ||
    (signals.selectedDriverId != null && signals.driverSheetOpen && !override.autoFocusOff)
  ) {
    return { mode: "focus_driver", reason: "driver_focus" };
  }
  return { mode: "normal", reason: "default" };
}

function applyVisualBudget(
  overlays: CockpitOverlayVisibility,
  maxMajor: number
): CockpitOverlayVisibility {
  const next = { ...overlays };
  const majors: (keyof CockpitOverlayVisibility)[] = [
    "delayedBanner",
    "statusPills",
    "quickTools",
    "opsSheet",
  ];
  let count = countMajorOverlays(next);
  if (count <= maxMajor) return next;
  for (const key of [...majors].reverse()) {
    if (!next[key]) continue;
    next[key] = false;
    count -= 1;
    if (count <= maxMajor) break;
  }
  return next;
}

/** Gouvernance centrale — unique point de décision UI cockpit. */
export function computeCockpitUiState(input: CockpitGovernanceInput): CockpitUiState {
  const flags = getCockpitAdvancedFlags();
  const override: CockpitHumanOverride = { ...COCKPIT_INITIAL_OVERRIDE, ...input.override };
  const nowMs = input.nowMs ?? Date.now();
  const bootMs = input.bootMs ?? nowMs;
  const stabilized = nowMs - bootMs >= STABILIZATION_MS;

  const healthScore = input.healthScore;
  const signals: CockpitRuntimeSignals = { ...input, healthScore: healthScore ?? 0 };

  const { mode, reason } =
    input.forcedMode != null
      ? { mode: input.forcedMode, reason: "stability_hold" }
      : resolveMode(signals, override, input.fsmState);
  const profile = COCKPIT_MODE_PROFILES[mode];

  let overlays = mergeOverlayVisibility(mode);
  if (!stabilized) {
    overlays = {
      ...DEFAULT_OVERLAY_VISIBILITY,
      topBar: true,
      mapControls: true,
      livePill: true,
      statusPills: false,
      quickTools: false,
      delayedBanner: false,
      opsSheet: false,
      trustBanner: false,
    };
  }

  if (signals.driverSheetOpen) {
    overlays = {
      ...overlays,
      statusPills: false,
      quickTools: false,
      delayedBanner: false,
    };
  }

  if (signals.opsSheetOpen) {
    overlays = { ...overlays, opsSheet: true };
  }

  const maxMajor = profile.maxMajorOverlays;
  if (flags.adaptiveOverlays) {
    overlays = applyVisualBudget(overlays, maxMajor);
  }

  const baselineOnly = mode === "baseline" || override.killSwitch;
  const safeMode = mode === "safe_mode";

  let trustMessage: string | null = null;
  if (safeMode) trustMessage = "Mode stable — affichage simplifié";
  else if (mode === "chaos_overload") trustMessage = "Charge élevée — priorité urgences";
  else if (resolveCockpitLiveStatus(signals.realtimeStatus) === "offline") {
    trustMessage = "Temps réel indisponible";
  }

  const lowAttention =
    flags.lowAttentionMode &&
    (mode === "chaos_overload" ||
      mode === "focus_urgent" ||
      mode === "navigation_active" ||
      signals.interactionBurstPerMinute > 35);

  let animationIntensity = profile.animationIntensity;
  if (safeMode || baselineOnly) animationIntensity = 0;
  if (override.predictiveSimplificationOff) animationIntensity = Math.min(animationIntensity, 0.2);

  const realtimeSocketExpected = isCompanyRealtimeSocketExpected();
  const liveStatus = resolveCockpitLiveStatus(signals.realtimeStatus, {
    realtimeSocketExpected,
  });
  const freshness = (signals.realtimeDataFreshness ?? "idle") as CompanyDataFreshness;
  const liveActivityHint =
    liveStatus === "connected" && realtimeSocketExpected
      ? formatCompanyActivityHint(
          signals.realtimeLastEventAt ?? null,
          freshness
        )
      : null;

  return {
    mode,
    liveStatus,
    realtimeSocketExpected,
    liveActivityHint,
    overlays: {
      ...overlays,
      trustBanner: Boolean(trustMessage) && overlays.trustBanner !== false,
    },
    majorOverlayLayers: countMajorOverlays(overlays),
    animationIntensity,
    cognitiveLoadLevel: lowAttention ? "low" : profile.cognitiveLoadLevel,
    lowAttention,
    safeMode,
    baselineOnly,
    trustMessage,
    cameraVerticalBias: profile.cameraVerticalBias,
    simplifyClustering: safeMode || baselineOnly,
    reduceRealtimeFrequency: safeMode,
    override,
    lastTransitionReason: reason,
  };
}
