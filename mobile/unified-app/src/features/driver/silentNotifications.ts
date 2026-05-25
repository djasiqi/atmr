import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";

export type SilentPushPayload = {
  type?: string;
  mission_id?: number | string;
  missionId?: number | string;
  booking_id?: number | string;
  silent?: boolean | number | string;
  silent_push?: boolean | number | string;
  background?: boolean | number | string;
  content_available?: boolean | number | string;
};

function parseMissionId(input: SilentPushPayload): number | null {
  const raw = input.mission_id ?? input.missionId ?? input.booking_id;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

export function isSilentPayload(input: unknown): input is SilentPushPayload {
  if (!input || typeof input !== "object") return false;
  const value = input as SilentPushPayload;
  const rawType = typeof value.type === "string" ? value.type.toLowerCase() : "";
  if (rawType === "mission_refresh") return true;
  const silent = value.silent ?? value.silent_push ?? value.background ?? value.content_available;
  if (typeof silent === "boolean") return silent;
  if (typeof silent === "number") return silent === 1;
  if (typeof silent === "string") {
    const normalized = silent.toLowerCase();
    return normalized === "1" || normalized === "true" || normalized === "silent";
  }
  return false;
}

export async function handleSilentPushPayload(
  payload: unknown,
  onResync: (missionId: number | null) => Promise<void>
): Promise<void> {
  if (!isSilentPayload(payload)) return;
  const missionId = parseMissionId(payload as SilentPushPayload);
  emitDriverTelemetry("push.notification.silent_sync", {
    source: "driver.silent_notifications",
    mission_id: missionId,
  });
  await onResync(missionId);
}
