import { NetworkSnapshot } from "./networkState";

export type ConnectivityMode = "offline" | "degraded" | "normal";

export type ConnectivityDecision = {
  mode: ConnectivityMode;
  allowSocket: boolean;
  allowGpsFlush: boolean;
  recommendedSyncIntervalMs: number;
};

export function evaluateConnectivityPolicy(snapshot: NetworkSnapshot): ConnectivityDecision {
  const normalIntervalMs = Number(process.env.EXPO_PUBLIC_NETWORK_POLICY_NORMAL_INTERVAL_MS ?? "10000");
  const poorIntervalMs = Number(process.env.EXPO_PUBLIC_NETWORK_POLICY_POOR_INTERVAL_MS ?? "20000");
  const offlineIntervalMs = Number(process.env.EXPO_PUBLIC_NETWORK_POLICY_OFFLINE_INTERVAL_MS ?? "30000");
  if (!snapshot.connected || !snapshot.internetReachable) {
    return {
      mode: "offline",
      allowSocket: false,
      allowGpsFlush: false,
      recommendedSyncIntervalMs: offlineIntervalMs,
    };
  }
  const generation = String(snapshot.cellularGeneration ?? "").toLowerCase();
  const poor = generation === "2g" || generation === "3g";
  if (poor) {
    return {
      mode: "degraded",
      allowSocket: true,
      allowGpsFlush: true,
      recommendedSyncIntervalMs: poorIntervalMs,
    };
  }
  return {
    mode: "normal",
    allowSocket: true,
    allowGpsFlush: true,
    recommendedSyncIntervalMs: normalIntervalMs,
  };
}
