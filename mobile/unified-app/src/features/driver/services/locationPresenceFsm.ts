export type PresenceState =
  | "offline"
  | "idle"
  | "available"
  | "assigned"
  | "en_route"
  | "on_trip"
  | "paused";

export type PresenceEvent =
  | "session_ready"
  | "mission_assigned"
  | "mission_started"
  | "mission_completed"
  | "go_offline"
  | "pause"
  | "resume";

const transitions: Record<PresenceState, Partial<Record<PresenceEvent, PresenceState>>> = {
  offline: { session_ready: "idle" },
  idle: { mission_assigned: "assigned", go_offline: "offline", pause: "paused" },
  available: { mission_assigned: "assigned", go_offline: "offline", pause: "paused" },
  assigned: { mission_started: "on_trip", go_offline: "offline" },
  en_route: { mission_started: "on_trip", go_offline: "offline" },
  on_trip: { mission_completed: "available", go_offline: "offline" },
  paused: { resume: "available", go_offline: "offline" },
};

let currentState: PresenceState = "offline";

export function getPresenceState(): PresenceState {
  return currentState;
}

export function applyPresenceEvent(event: PresenceEvent): PresenceState {
  const next = transitions[currentState]?.[event];
  if (next) {
    currentState = next;
  }
  return currentState;
}

export function resetPresenceState(): void {
  currentState = "offline";
}
