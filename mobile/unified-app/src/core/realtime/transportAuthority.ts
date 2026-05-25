import type { RealtimeState } from "./realtimeManager";

/** Autorité effective pour les missions : socket > polling > http. */
export type MissionTransportAuthority = "socket" | "polling" | "http";

export function resolveMissionTransportAuthority(
  state: Pick<
    RealtimeState,
    "connected" | "degradedMode" | "transportAuthority" | "actualTransport"
  >
): MissionTransportAuthority {
  if (state.connected && !state.degradedMode && state.transportAuthority === "socket") {
    return "socket";
  }
  if (state.degradedMode || state.transportAuthority === "polling") {
    return "polling";
  }
  return "http";
}

export function shouldSkipMissionPolling(
  state: Pick<
    RealtimeState,
    "connected" | "degradedMode" | "transportAuthority" | "actualTransport"
  >
): boolean {
  return resolveMissionTransportAuthority(state) === "socket";
}
