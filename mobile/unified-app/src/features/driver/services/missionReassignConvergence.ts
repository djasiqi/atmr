import type { DriverMission, DriverSocketEvent } from "../types";

type Resolution = {
  shouldForceResync: boolean;
  reason: "none" | "reassigned_status" | "mission_reassigned_event";
};

export function resolveMissionReassignConvergence(
  localMission: DriverMission | undefined,
  event: DriverSocketEvent
): Resolution {
  const eventType = String(event.event_type ?? "").toLowerCase();
  const payload = (event.payload ?? {}) as Record<string, unknown>;
  const payloadStatus = String(payload.status ?? localMission?.status ?? "").toUpperCase();
  const missionStatus = String(localMission?.status ?? "").toUpperCase();

  if (eventType === "mission_reassigned") {
    return { shouldForceResync: true, reason: "mission_reassigned_event" };
  }

  const reassigned = payloadStatus === "REASSIGNED" || missionStatus === "REASSIGNED";
  if (reassigned) {
    return { shouldForceResync: true, reason: "reassigned_status" };
  }

  return { shouldForceResync: false, reason: "none" };
}
