import type { CockpitOverlayVisibility, CockpitUiMode } from "./cockpitTypes";
import { DEFAULT_OVERLAY_VISIBILITY } from "./cockpitTypes";

export type CockpitModeProfile = {
  overlays: Partial<CockpitOverlayVisibility>;
  animationIntensity: number;
  cognitiveLoadLevel: "low" | "medium" | "high";
  cameraVerticalBias: number;
  maxMajorOverlays: number;
};

const FOCUS_REDUCED: Partial<CockpitOverlayVisibility> = {
  statusPills: false,
  quickTools: false,
  delayedBanner: false,
};

export const COCKPIT_MODE_PROFILES: Record<CockpitUiMode, CockpitModeProfile> = {
  normal: {
    overlays: {},
    animationIntensity: 0.35,
    cognitiveLoadLevel: "medium",
    cameraVerticalBias: 0,
    maxMajorOverlays: 2,
  },
  focus_driver: {
    overlays: { ...FOCUS_REDUCED, quickTools: false },
    animationIntensity: 0.25,
    cognitiveLoadLevel: "low",
    cameraVerticalBias: 0.4,
    maxMajorOverlays: 1,
  },
  focus_mission: {
    overlays: { ...FOCUS_REDUCED },
    animationIntensity: 0.3,
    cognitiveLoadLevel: "low",
    cameraVerticalBias: 0.38,
    maxMajorOverlays: 1,
  },
  focus_urgent: {
    overlays: { statusPills: false, quickTools: false, livePill: true, delayedBanner: true },
    animationIntensity: 0.4,
    cognitiveLoadLevel: "low",
    cameraVerticalBias: 0.35,
    maxMajorOverlays: 2,
  },
  search: {
    overlays: { statusPills: false, delayedBanner: false, quickTools: true },
    animationIntensity: 0.2,
    cognitiveLoadLevel: "low",
    cameraVerticalBias: 0,
    maxMajorOverlays: 1,
  },
  filters: {
    overlays: { statusPills: false, delayedBanner: false },
    animationIntensity: 0.2,
    cognitiveLoadLevel: "low",
    cameraVerticalBias: 0,
    maxMajorOverlays: 1,
  },
  navigation_active: {
    overlays: { statusPills: false, quickTools: false, delayedBanner: false, livePill: true },
    animationIntensity: 0.15,
    cognitiveLoadLevel: "low",
    cameraVerticalBias: 0.42,
    maxMajorOverlays: 1,
  },
  chaos_overload: {
    overlays: { quickTools: false, statusPills: true, delayedBanner: true, livePill: true },
    animationIntensity: 0.2,
    cognitiveLoadLevel: "low",
    cameraVerticalBias: 0.36,
    maxMajorOverlays: 2,
  },
  safe_mode: {
    overlays: {
      statusPills: true,
      quickTools: false,
      delayedBanner: true,
      livePill: true,
      mapControls: true,
    },
    animationIntensity: 0,
    cognitiveLoadLevel: "low",
    cameraVerticalBias: 0,
    maxMajorOverlays: 2,
  },
  baseline: {
    overlays: {
      statusPills: true,
      quickTools: false,
      delayedBanner: false,
      livePill: true,
      mapControls: true,
      trustBanner: false,
    },
    animationIntensity: 0,
    cognitiveLoadLevel: "low",
    cameraVerticalBias: 0,
    maxMajorOverlays: 1,
  },
};

export function mergeOverlayVisibility(
  mode: CockpitUiMode,
  base: CockpitOverlayVisibility = DEFAULT_OVERLAY_VISIBILITY
): CockpitOverlayVisibility {
  const patch = COCKPIT_MODE_PROFILES[mode].overlays;
  return { ...base, ...patch };
}

export function countMajorOverlays(overlays: CockpitOverlayVisibility): number {
  const majors: (keyof CockpitOverlayVisibility)[] = [
    "statusPills",
    "quickTools",
    "delayedBanner",
    "opsSheet",
  ];
  return majors.filter((k) => overlays[k]).length;
}
