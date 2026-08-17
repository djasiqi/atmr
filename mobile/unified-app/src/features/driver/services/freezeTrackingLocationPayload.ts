/**
 * Immutabilité event_id → payload (P0-E BG_FRESHNESS / D4-B).
 *
 * À l'enqueue uniquement : figer timestamp/recorded_at/sent_at + identité session/seq.
 * Les retries HTTP doivent rejouer EXACTEMENT ces champs (jamais Date.now() au flush).
 */
import type { DriverLocationPayload } from "../types";

export type FrozenTrackingLocationPayload = DriverLocationPayload & {
  timestamp: string;
  recordedAt: string;
  sentAt: string;
  trackingEventId: string;
  trackingSessionId: string;
  sessionGeneration: number;
  sequenceId: number;
  captureId: string;
  locationMode: NonNullable<DriverLocationPayload["locationMode"]>;
};

function requireFiniteNumber(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`invalid_${field}`);
  }
  return value;
}

/** Horodatage ISO figé une seule fois (enqueue). */
export function resolveFrozenRecordedAt(payload: DriverLocationPayload): string {
  const candidates = [payload.recordedAt, payload.timestamp, payload.sentAt];
  for (const raw of candidates) {
    if (typeof raw === "string" && raw.trim().length > 0) {
      return raw.trim();
    }
  }
  return new Date().toISOString();
}

/**
 * Construit le payload wire immuable stocké en SQLite.
 * Toute nouvelle position GPS doit passer par un nouvel event_id (nouvel enqueue).
 */
export function freezeTrackingLocationPayload(input: {
  eventId: string;
  sequenceId: number;
  trackingSessionId: string;
  sessionGeneration: number;
  captureId: string;
  locationMode: NonNullable<DriverLocationPayload["locationMode"]>;
  missionId: number | null;
  payload: DriverLocationPayload;
  /** Instant d'enqueue (sent_at) — distinct du GNSS si besoin. */
  enqueuedAtIso?: string;
}): FrozenTrackingLocationPayload {
  const recordedAt = resolveFrozenRecordedAt(input.payload);
  const sentAt =
    (typeof input.enqueuedAtIso === "string" && input.enqueuedAtIso.trim()) ||
    recordedAt;

  const frozen: FrozenTrackingLocationPayload = {
    latitude: requireFiniteNumber(input.payload.latitude, "latitude"),
    longitude: requireFiniteNumber(input.payload.longitude, "longitude"),
    accuracy: input.payload.accuracy,
    heading: input.payload.heading,
    speed: input.payload.speed,
    timestamp: recordedAt,
    recordedAt,
    sentAt,
    isBackground: input.payload.isBackground === true,
    missionId: input.missionId ?? input.payload.missionId ?? null,
    locationMode: input.locationMode,
    trackingEventId: input.eventId,
    trackingSessionId: input.trackingSessionId,
    sessionGeneration: input.sessionGeneration,
    sequenceId: input.sequenceId,
    captureId: input.captureId,
    trackingGenerationId: input.payload.trackingGenerationId ?? null,
    missionContextVersion: input.payload.missionContextVersion ?? null,
    trackingIdentityId: input.payload.trackingIdentityId ?? null,
  };

  return Object.freeze(frozen);
}

/** Corps HTTP dérivé du payload figé — déterministe pour retries. */
export function buildDriverLocationHttpBody(
  payload: DriverLocationPayload
): Record<string, unknown> {
  const recordedAt = payload.recordedAt ?? payload.timestamp;
  if (typeof recordedAt !== "string" || recordedAt.trim().length === 0) {
    throw new Error("missing_recorded_at");
  }
  const timestamp = payload.timestamp ?? recordedAt;
  const sentAt = payload.sentAt ?? recordedAt;
  return {
    latitude: payload.latitude,
    longitude: payload.longitude,
    accuracy: payload.accuracy ?? null,
    heading: payload.heading ?? null,
    speed: payload.speed ?? null,
    is_background: payload.isBackground ?? false,
    mission_id: payload.missionId ?? null,
    timestamp,
    recorded_at: recordedAt,
    sent_at: sentAt,
    location_mode: payload.locationMode ?? "availability_presence",
    tracking_event_id: payload.trackingEventId ?? null,
    tracking_session_id: payload.trackingSessionId ?? null,
    session_generation: payload.sessionGeneration ?? null,
    sequence_id: payload.sequenceId ?? null,
    capture_id: payload.captureId ?? null,
  };
}
