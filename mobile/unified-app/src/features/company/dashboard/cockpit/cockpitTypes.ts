/** Modes cockpit — source de vérité UI. */
export type CockpitUiMode =
  | "normal"
  | "focus_driver"
  | "focus_mission"
  | "focus_urgent"
  | "search"
  | "filters"
  | "navigation_active"
  | "chaos_overload"
  | "safe_mode"
  | "baseline";

export type CockpitLiveStatus =
  | "connected"
  | "reconnecting"
  | "offline"
  | "initializing"
  | "http_fallback";

export type CockpitOverlayKey =
  | "topBar"
  | "mapControls"
  | "statusPills"
  | "quickTools"
  | "delayedBanner"
  | "livePill"
  | "opsSheet"
  | "trustBanner";

export type CockpitDriverSheetSnap = "collapsed" | "medium" | "expanded";

export type CockpitHumanOverride = {
  killSwitch: boolean;
  autoFocusOff: boolean;
  predictiveSimplificationOff: boolean;
  manualRecenterOnly: boolean;
};

export type CockpitRuntimeSignals = {
  realtimeStatus: string;
  realtimeDataFreshness?: string;
  realtimeLastEventAt?: string | null;
  driverSheetOpen: boolean;
  driverSheetSnap: CockpitDriverSheetSnap;
  opsSheetOpen: boolean;
  filtersOpen: boolean;
  layersOpen: boolean;
  searchActive: boolean;
  navigationActive: boolean;
  selectedDriverId: number | null;
  selectedMissionId: number | null;
  urgentCount: number;
  delayedCount: number;
  activeDriverCount: number;
  inProgressMissionCount: number;
  policyFailureCount: number;
  interactionBurstPerMinute: number;
  healthScore: number;
  stabilized: boolean;
};

export type CockpitOverlayVisibility = Record<CockpitOverlayKey, boolean>;

export type CockpitUiState = {
  mode: CockpitUiMode;
  liveStatus: CockpitLiveStatus;
  /** false = repli HTTP sans WebSocket (évite faux « Connexion lente »). */
  realtimeSocketExpected: boolean;
  /** Sous-titre optionnel sous la pill LIVE (activité calme / données anciennes). */
  liveActivityHint: string | null;
  overlays: CockpitOverlayVisibility;
  majorOverlayLayers: number;
  animationIntensity: number;
  cognitiveLoadLevel: "low" | "medium" | "high";
  lowAttention: boolean;
  safeMode: boolean;
  baselineOnly: boolean;
  trustMessage: string | null;
  cameraVerticalBias: number;
  simplifyClustering: boolean;
  reduceRealtimeFrequency: boolean;
  override: CockpitHumanOverride;
  lastTransitionReason: string | null;
};

export const COCKPIT_INITIAL_OVERRIDE: CockpitHumanOverride = {
  killSwitch: false,
  autoFocusOff: false,
  predictiveSimplificationOff: false,
  manualRecenterOnly: false,
};

export const DEFAULT_OVERLAY_VISIBILITY: CockpitOverlayVisibility = {
  topBar: true,
  mapControls: true,
  statusPills: true,
  quickTools: true,
  delayedBanner: true,
  livePill: true,
  opsSheet: false,
  trustBanner: false,
};
