export type StabilityAction = "camera" | "overlay" | "mode" | "density";

export const STABILITY_WINDOWS_MS: Record<StabilityAction, number> = {
  camera: 2000,
  overlay: 1500,
  mode: 2500,
  density: 3000,
};

import type { CockpitUiMode } from "./cockpitTypes";
import type { DensityLevel } from "./mapDensityGovernor";

export type StabilityBudgetState = {
  lastCameraSwitchMs: number;
  lastOverlayResetMs: number;
  lastModeFlipMs: number;
  lastDensityFlipMs: number;
  lastDensityLevel?: DensityLevel;
  lastUiMode?: CockpitUiMode;
};

export function createStabilityBudgetState(nowMs = Date.now()): StabilityBudgetState {
  return {
    lastCameraSwitchMs: 0,
    lastOverlayResetMs: 0,
    lastModeFlipMs: 0,
    lastDensityFlipMs: 0,
  };
}

export function canApplyStabilityAction(
  state: StabilityBudgetState,
  action: StabilityAction,
  nowMs: number,
  bypassCritical = false
): boolean {
  if (bypassCritical) return true;
  const last =
    action === "camera"
      ? state.lastCameraSwitchMs
      : action === "overlay"
        ? state.lastOverlayResetMs
        : action === "density"
          ? state.lastDensityFlipMs
          : state.lastModeFlipMs;
  return nowMs - last >= STABILITY_WINDOWS_MS[action];
}

export function recordStabilityAction(
  state: StabilityBudgetState,
  action: StabilityAction,
  nowMs: number
): StabilityBudgetState {
  if (action === "camera") {
    return { ...state, lastCameraSwitchMs: nowMs };
  }
  if (action === "overlay") {
    return { ...state, lastOverlayResetMs: nowMs };
  }
  if (action === "density") {
    return { ...state, lastDensityFlipMs: nowMs };
  }
  return { ...state, lastModeFlipMs: nowMs };
}
