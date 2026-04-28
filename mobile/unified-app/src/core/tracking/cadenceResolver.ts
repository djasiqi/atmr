import { TrackingMode } from "./trackingManager";

export type TrackingNetworkProfile = "offline" | "poor" | "normal";

export type CadenceResolverInput = {
  mode: TrackingMode;
  appState: "active" | "background" | "inactive";
  queueDepth: number;
  socketReady: boolean;
  consecutiveFailures: number;
  previousProfile: TrackingNetworkProfile | null;
  profileSinceMs: number;
  nowMs: number;
  networkModeHint?: "offline" | "degraded" | "normal" | null;
};

export type TrackingCadence = {
  networkProfile: TrackingNetworkProfile;
  foregroundIntervalMs: number;
  backgroundIntervalMs: number;
  ackStaleMs: number;
};

const HYSTERESIS_MS = Number(process.env.EXPO_PUBLIC_DRIVER_CADENCE_HYSTERESIS_MS ?? "60000");
const ACK_STALE_POOR_MS = Number(process.env.EXPO_PUBLIC_DRIVER_ACK_STALE_POOR_MS ?? "25000");
const ACK_STALE_MEDIUM_MS = Number(process.env.EXPO_PUBLIC_DRIVER_ACK_STALE_MEDIUM_MS ?? "45000");
const ACK_STALE_NORMAL_MS = Number(process.env.EXPO_PUBLIC_DRIVER_ACK_STALE_NORMAL_MS ?? "75000");

function resolveProfile(input: CadenceResolverInput): TrackingNetworkProfile {
  if (input.networkModeHint === "offline") return "offline";
  if (input.networkModeHint === "degraded") return "poor";
  if (!input.socketReady && input.queueDepth >= 200) return "offline";
  if (!input.socketReady || input.consecutiveFailures >= 3 || input.queueDepth >= 60) return "poor";
  return "normal";
}

function applyHysteresis(input: CadenceResolverInput, nextProfile: TrackingNetworkProfile) {
  if (!input.previousProfile || input.previousProfile === nextProfile) {
    return nextProfile;
  }
  const spentInPreviousProfile = input.nowMs - input.profileSinceMs;
  if (spentInPreviousProfile < HYSTERESIS_MS) {
    return input.previousProfile;
  }
  return nextProfile;
}

function modeIntervals(mode: TrackingMode, profile: TrackingNetworkProfile) {
  if (mode === "availability_presence") {
    if (profile === "normal") return { foregroundIntervalMs: 45_000, backgroundIntervalMs: 120_000 };
    if (profile === "poor") return { foregroundIntervalMs: 60_000, backgroundIntervalMs: 150_000 };
    return { foregroundIntervalMs: 60_000, backgroundIntervalMs: 180_000 };
  }
  if (profile === "normal") return { foregroundIntervalMs: 8_000, backgroundIntervalMs: 20_000 };
  if (profile === "poor") return { foregroundIntervalMs: 15_000, backgroundIntervalMs: 20_000 };
  return { foregroundIntervalMs: 20_000, backgroundIntervalMs: 30_000 };
}

function ackStaleMsForProfile(profile: TrackingNetworkProfile) {
  if (profile === "offline") return ACK_STALE_POOR_MS;
  if (profile === "poor") return ACK_STALE_MEDIUM_MS;
  return ACK_STALE_NORMAL_MS;
}

export function resolveTrackingCadence(input: CadenceResolverInput): TrackingCadence {
  const candidateProfile = resolveProfile(input);
  const networkProfile = applyHysteresis(input, candidateProfile);
  const intervals = modeIntervals(input.mode, networkProfile);
  return {
    networkProfile,
    foregroundIntervalMs: intervals.foregroundIntervalMs,
    backgroundIntervalMs: intervals.backgroundIntervalMs,
    ackStaleMs: ackStaleMsForProfile(networkProfile),
  };
}
