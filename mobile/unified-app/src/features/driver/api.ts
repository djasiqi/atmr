import { AxiosError } from "axios";
import { apiClient } from "../../core/api/client";
import { resolveDriverStatusForUx } from "./statusDictionary";
import {
  HttpCircuitBreakerScope,
  onHttpRequestFailure,
  onHttpRequestSuccess,
  shouldAllowHttpRequest,
} from "../../core/network/httpCircuitBreaker";
import {
  DriverLocationPayload,
  DriverLocationAck,
  DriverMission,
  DriverMissionDetail,
  DriverPushRegistrationPayload,
  DriverStatusTransitionPayload,
} from "./types";
import { normalizeDriverProfilePayload } from "./domain/driverAvailability";
import {
  readDriverProfileCache,
  writeDriverProfileCache,
} from "./services/driverProfileCache";

/** Correspond au JSON renvoyé par PUT /driver/me/bookings/:id/status. */
export type DriverStatusUpdateResult = {
  booking_id?: number;
  status?: string;
  server_time?: string;
  unchanged?: boolean;
  mission_milestone?: string;
  error?: string;
  [key: string]: unknown;
};

export type DriverApiError = {
  status: number | null;
  code: string;
  message: string;
  retryable?: boolean;
  retry_after_seconds?: number;
};

export type DriverChatMessage = {
  id: number | string;
  sender_id?: number | string | null;
  receiver_id?: number | string | null;
  content: string;
  sender_role?: string;
  sender_name?: string | null;
  timestamp: string;
  image_url?: string | null;
  pdf_url?: string | null;
  pdf_filename?: string | null;
  audio_url?: string | null;
};

export type DriverEtaSnapshot = {
  mission_id: number;
  eta_minutes: number | null;
  eta_updated_at: string | null;
  has_gps?: boolean;
  driver_lat?: number | null;
  driver_lon?: number | null;
};

export type DriverProfile = {
  id?: number | string;
  first_name?: string | null;
  last_name?: string | null;
  full_name?: string | null;
  email?: string | null;
  phone?: string | null;
  photo_url?: string | null;
  is_available?: boolean | number | string | null;
  [key: string]: unknown;
};

export type DriverRouteSnapshot = {
  route?: unknown;
  points?: unknown[];
  [key: string]: unknown;
};

export type DriverCompletedTrip = {
  id: number | string;
  pickup_location?: string | null;
  dropoff_location?: string | null;
  status?: string | null;
  [key: string]: unknown;
};

function normalizeError(error: unknown): DriverApiError {
  const e = error as AxiosError<{
    error?: string;
    code?: string;
    message?: string;
    error_code?: string;
    error_message?: string;
    retryable?: boolean;
  }>;
  return {
    status: e.response?.status ?? null,
    code: e.response?.data?.error_code ?? e.response?.data?.code ?? "UNKNOWN_ERROR",
    message:
      e.response?.data?.error_message ??
      e.response?.data?.message ??
      e.response?.data?.error ??
      e.message ??
      "Erreur inconnue",
    retryable:
      typeof e.response?.data?.retryable === "boolean"
        ? e.response?.data?.retryable
        : undefined,
    retry_after_seconds:
      typeof e.response?.data?.retry_after_seconds === "number"
        ? e.response.data.retry_after_seconds
        : undefined,
  };
}

function toFailureReason(error: unknown): string {
  const axiosError = error as AxiosError;
  if (axiosError.code) return axiosError.code;
  if (axiosError.response?.status) return `http_${axiosError.response.status}`;
  return "unknown_error";
}

async function runWithCircuitBreaker<T>(
  scope: HttpCircuitBreakerScope,
  operation: () => Promise<T>
): Promise<T> {
  if (!shouldAllowHttpRequest(scope)) {
    throw {
      status: null,
      code: "HTTP_CIRCUIT_BREAKER_OPEN",
      message: `Circuit breaker HTTP ouvert (${scope})`,
      retryable: true,
    } satisfies DriverApiError;
  }
  try {
    const result = await operation();
    onHttpRequestSuccess(scope);
    return result;
  } catch (error) {
    onHttpRequestFailure(scope, toFailureReason(error));
    throw normalizeError(error);
  }
}

function asArray(value: unknown): DriverMission[] {
  if (Array.isArray(value)) {
    return value as DriverMission[];
  }
  if (value && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    if (Array.isArray(obj.items)) return obj.items as DriverMission[];
    if (Array.isArray(obj.results)) return obj.results as DriverMission[];
    if (Array.isArray(obj.data)) return obj.data as DriverMission[];
    if (Array.isArray(obj.bookings)) return obj.bookings as DriverMission[];
  }
  return [];
}

export async function getDriverMissions(): Promise<DriverMission[]> {
  return runWithCircuitBreaker("mission_sync", async () => {
    const { data } = await apiClient.get("/driver/me/bookings");
    return asArray(data);
  });
}

export async function getDriverMissionsSince(sinceIso: string): Promise<DriverMission[]> {
  return runWithCircuitBreaker("mission_sync", async () => {
    const { data } = await apiClient.get("/driver/me/bookings/since", {
      params: { since: sinceIso, include_terminal: true },
    });
    return asArray(data);
  });
}

export async function getDriverMissionDetail(missionId: number): Promise<DriverMissionDetail> {
  return runWithCircuitBreaker("mission_sync", async () => {
    const { data } = await apiClient.get(`/driver/me/bookings/${missionId}`);
    return data as DriverMissionDetail;
  });
}

/** Corps PUT /driver/me/bookings/:id/status (contrat Flask `cancel_reason`). */
export function buildDriverStatusUpdateBody(payload: DriverStatusTransitionPayload) {
  const { targetStatus, reason } = payload;
  if (targetStatus === "CANCELLED" || targetStatus === "FAILED") {
    const normalizedReason = String(reason ?? "").trim().toUpperCase();
    if (normalizedReason === "RELEASE") {
      return { status: targetStatus, cancel_reason: "RELEASE" };
    }
    const reasonText = String(reason ?? "").trim();
    return {
      status: targetStatus,
      cancel_reason: normalizedReason === "FAILED" ? "FAILED" : "CANCEL",
      ...(reasonText && normalizedReason !== "FAILED" ? { reason_text: reasonText } : {}),
    };
  }
  return { status: targetStatus };
}

export async function updateDriverMissionStatus(
  payload: DriverStatusTransitionPayload
): Promise<DriverStatusUpdateResult> {
  return runWithCircuitBreaker("mission_transition", async () => {
    const { data } = await apiClient.put<DriverStatusUpdateResult>(
      `/driver/me/bookings/${payload.missionId}/status`,
      buildDriverStatusUpdateBody(payload),
      {
        headers: {
          "X-Idempotency-Key": payload.idempotencyKey,
        },
      }
    );
    return (data ?? {}) as DriverStatusUpdateResult;
  });
}

/** Valide ingested_event_ids / retry_event_ids (fail-closed si mal formés). */
export function asAckStringArray(value: unknown): string[] | null {
  if (value == null) return null;
  if (!Array.isArray(value)) {
    throw new Error("ack_event_ids_invalid");
  }
  const values = value.filter(
    (item): item is string => typeof item === "string" && item.trim().length > 0
  );
  if (values.length !== value.length) {
    throw new Error("ack_event_ids_invalid");
  }
  return values;
}

export async function sendDriverLocation(payload: DriverLocationPayload): Promise<DriverLocationAck> {
  return runWithCircuitBreaker("tracking_http", async () => {
    const { data } = await apiClient.put("/driver/me/location", {
      latitude: payload.latitude,
      longitude: payload.longitude,
      accuracy: payload.accuracy ?? null,
      heading: payload.heading ?? null,
      speed: payload.speed ?? null,
      is_background: payload.isBackground ?? false,
      mission_id: payload.missionId ?? null,
      timestamp: payload.timestamp ?? new Date().toISOString(),
      location_mode: payload.locationMode ?? "availability_presence",
      tracking_event_id: payload.trackingEventId ?? null,
      tracking_session_id: payload.trackingSessionId ?? null,
      session_generation: payload.sessionGeneration ?? null,
      sequence_id: payload.sequenceId ?? null,
    }, {
      headers: {
        "X-Allow-Offline-Attempt": "1",
        ...(payload.trackingEventId
          ? { "X-Location-Event-Id": payload.trackingEventId }
          : {}),
      },
    });
    const ackBody = data as {
      ack_status?: unknown;
      accept_status?: unknown;
      accept_reason?: unknown;
      durability?: unknown;
      queued?: unknown;
      tracking_event_id?: unknown;
      location_event_id?: unknown;
      trace_id?: unknown;
      ingested_event_ids?: unknown;
      retry_event_ids?: unknown;
    };
    const ingestedEventIds = asAckStringArray(ackBody.ingested_event_ids);
    const retryEventIds = asAckStringArray(ackBody.retry_event_ids);
    const durabilityRaw =
      typeof ackBody.durability === "string" ? ackBody.durability : null;
    const durability =
      durabilityRaw === "persisted_sync" || durabilityRaw === "queued_async"
        ? durabilityRaw
        : null;
    const locationEventId =
      typeof ackBody.location_event_id === "string"
        ? ackBody.location_event_id
        : typeof ackBody.tracking_event_id === "string"
          ? ackBody.tracking_event_id
          : null;
    const acceptStatus =
      typeof ackBody.accept_status === "string" ? ackBody.accept_status : null;

    // 202 / queued_async → non final (conservation SQLite + watermark)
    if (
      durability === "queued_async" ||
      acceptStatus === "accepted_async" ||
      ackBody.queued === true ||
      ackBody.ack_status === "ingested_non_persisted"
    ) {
      return {
        ack_status: "ingested_non_persisted",
        durability: "queued_async",
        accept_reason:
          typeof ackBody.accept_reason === "string" ? ackBody.accept_reason : "queued_kafka",
        tracking_event_id: locationEventId,
        location_event_id: locationEventId,
        trace_id: typeof ackBody.trace_id === "string" ? String(ackBody.trace_id) : null,
        ingested_event_ids: ingestedEventIds,
        retry_event_ids: retryEventIds,
      };
    }

    // Sync durable explicite — ne JAMAIS inventer persisted_sync
    if (durability === "persisted_sync" && ackBody.ack_status === "persisted") {
      return {
        ack_status: "persisted",
        durability: "persisted_sync",
        accept_reason:
          typeof ackBody.accept_reason === "string" ? String(ackBody.accept_reason) : null,
        tracking_event_id: locationEventId,
        location_event_id: locationEventId,
        trace_id: typeof ackBody.trace_id === "string" ? String(ackBody.trace_id) : null,
        ingested_event_ids: ingestedEventIds,
        retry_event_ids: retryEventIds,
      };
    }

    // Phase 0A : absence / valeur inconnue → erreur (retry), jamais « accepted » artificiel
    // 200 sans durability : ne pas tombstoner (compat backend ancien)
    const rawStatus = ackBody.ack_status;
    if (rawStatus == null || rawStatus === "") {
      // Ancien backend sans ack_status ni durability : conserver en file (retry)
      throw new Error("ack_status_missing");
    }
    const status = String(rawStatus);
    const ackStatus: DriverLocationAck["ack_status"] =
      status === "accepted" ||
      status === "queued" ||
      status === "duplicate" ||
      status === "stale" ||
      status === "ignored" ||
      status === "rejected" ||
      status === "ingested" ||
      status === "ingested_non_persisted" ||
      status === "partially_ingested" ||
      status === "persisted"
        ? (status as DriverLocationAck["ack_status"])
        : (() => {
            throw new Error(`ack_status_unknown:${status}`);
          })();
    // "accepted" sans durability → ne pas traiter comme persisted_sync
    return {
      ack_status: ackStatus === "accepted" ? "ingested" : ackStatus,
      durability: null,
      accept_reason: typeof ackBody.accept_reason === "string"
        ? String(ackBody.accept_reason)
        : null,
      tracking_event_id: locationEventId,
      location_event_id: locationEventId,
      trace_id: typeof ackBody.trace_id === "string" ? String(ackBody.trace_id) : null,
      ingested_event_ids: ingestedEventIds,
      retry_event_ids: retryEventIds,
    };
  });
}

export async function registerDriverPushToken(
  payload: DriverPushRegistrationPayload
): Promise<void> {
  try {
    await apiClient.post("/driver/save-push-token", {
      token: payload.token,
      driverId: payload.driverId,
      device_id: payload.deviceId,
      platform: payload.platform,
      provider: payload.provider,
      client_auth_surface: "driver",
    });
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function quickAcceptDriverMission(missionId: number): Promise<void> {
  try {
    await apiClient.post(`/driver/me/bookings/${missionId}/quick-accept`);
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function quickRejectDriverMission(missionId: number): Promise<void> {
  try {
    await apiClient.post(`/driver/me/bookings/${missionId}/quick-reject`);
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function quickStartDriverMission(missionId: number): Promise<void> {
  await updateDriverMissionStatus({
    missionId,
    targetStatus: "IN_PROGRESS",
    idempotencyKey: `push-start-${missionId}-${Date.now()}`,
  });
}

export async function quickCompleteDriverMission(missionId: number): Promise<void> {
  await updateDriverMissionStatus({
    missionId,
    targetStatus: "COMPLETED",
    idempotencyKey: `push-complete-${missionId}-${Date.now()}`,
  });
}

export async function updateDriverAvailability(isAvailable: boolean): Promise<void> {
  try {
    await apiClient.put("/driver/me/availability", {
      is_available: isAvailable,
    });
  } catch (error) {
    throw normalizeError(error);
  }
}

function normalizeMessages(value: unknown): DriverChatMessage[] {
  if (!Array.isArray(value)) return [];
  const messages: DriverChatMessage[] = [];
  value.forEach((entry) => {
    if (!entry || typeof entry !== "object") return;
    const raw = entry as Record<string, unknown>;
    const id = raw.id;
    const content = typeof raw.content === "string" ? raw.content : "";
    const timestamp =
      typeof raw.timestamp === "string" && raw.timestamp.length > 0
        ? raw.timestamp
        : new Date().toISOString();
    messages.push({
      id: typeof id === "string" || typeof id === "number" ? id : `${timestamp}-${content}`,
      sender_id:
        typeof raw.sender_id === "string" || typeof raw.sender_id === "number"
          ? raw.sender_id
          : null,
      receiver_id:
        typeof raw.receiver_id === "string" || typeof raw.receiver_id === "number"
          ? raw.receiver_id
          : null,
      content,
      sender_role: typeof raw.sender_role === "string" ? raw.sender_role : undefined,
      sender_name: typeof raw.sender_name === "string" ? raw.sender_name : null,
      timestamp,
      image_url: typeof raw.image_url === "string" ? raw.image_url : null,
      pdf_url: typeof raw.pdf_url === "string" ? raw.pdf_url : null,
      pdf_filename: typeof raw.pdf_filename === "string" ? raw.pdf_filename : null,
      audio_url: typeof raw.audio_url === "string" ? raw.audio_url : null,
    });
  });
  return messages;
}

export async function getDriverMessages(
  companyId: number,
  options?: { before?: string; limit?: number }
): Promise<DriverChatMessage[]> {
  try {
    const { data } = await apiClient.get(`/messages/${companyId}`, {
      params: {
        before: options?.before,
        limit: options?.limit ?? 40,
      },
    });
    return normalizeMessages(data);
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function getDriverMissionEta(
  missionId: number,
  options?: { missionStatus?: string | null }
): Promise<DriverEtaSnapshot> {
  try {
    const { data } = await apiClient.get("/driver/me/bookings/eta");
    const payload = (data ?? {}) as Record<string, unknown>;
    const bookings = Array.isArray(payload.bookings)
      ? (payload.bookings as Record<string, unknown>[])
      : [];
    const booking = bookings.find((item) => Number(item.id) === missionId);

    if (!booking) {
      const etaMinutesRaw = payload.eta_minutes ?? payload.eta ?? null;
      const etaMinutes =
        typeof etaMinutesRaw === "number" ? etaMinutesRaw : Number(etaMinutesRaw);
      return {
        mission_id: missionId,
        eta_minutes: Number.isFinite(etaMinutes) ? etaMinutes : null,
        eta_updated_at:
          typeof payload.eta_updated_at === "string" && payload.eta_updated_at.length > 0
            ? payload.eta_updated_at
            : new Date().toISOString(),
        has_gps: payload.has_gps === true,
      };
    }

    const statusKey = resolveDriverStatusForUx(options?.missionStatus ?? null);
    const etaSecondsRaw =
      statusKey === "IN_PROGRESS"
        ? booking.eta_to_dropoff_seconds
        : booking.eta_to_pickup_seconds;
    const etaSeconds =
      typeof etaSecondsRaw === "number" ? etaSecondsRaw : Number(etaSecondsRaw);
    const etaMinutes = Number.isFinite(etaSeconds) && etaSeconds > 0
      ? Math.max(1, Math.round(etaSeconds / 60))
      : null;

    const driverPos = payload.driver_position as Record<string, unknown> | undefined;
    const driverLatRaw = driverPos?.lat;
    const driverLonRaw = driverPos?.lon ?? driverPos?.lng;
    const driverLat =
      typeof driverLatRaw === "number" ? driverLatRaw : Number(driverLatRaw);
    const driverLon =
      typeof driverLonRaw === "number" ? driverLonRaw : Number(driverLonRaw);

    return {
      mission_id: missionId,
      eta_minutes: etaMinutes,
      eta_updated_at: new Date().toISOString(),
      has_gps: payload.has_gps === true,
      driver_lat: Number.isFinite(driverLat) ? driverLat : null,
      driver_lon: Number.isFinite(driverLon) ? driverLon : null,
    };
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function getDriverProfile(): Promise<DriverProfile> {
  const cacheHit = await readDriverProfileCache({ allowStale: false });
  if (cacheHit.status === "hit" && cacheHit.profile) {
    return cacheHit.profile;
  }
  try {
    const { data } = await apiClient.get("/driver/me/profile");
    if (data && typeof data === "object") {
      const profile = normalizeDriverProfilePayload(data);
      await writeDriverProfileCache(profile);
      return profile;
    }
    return {};
  } catch (error) {
    const staleCache = await readDriverProfileCache({ allowStale: true });
    if (staleCache.profile) {
      return staleCache.profile;
    }
    throw normalizeError(error);
  }
}

export async function updateDriverProfile(payload: Partial<DriverProfile>): Promise<DriverProfile> {
  try {
    const { data } = await apiClient.put("/driver/me/profile", payload);
    if (data && typeof data === "object") {
      const profile = data as DriverProfile;
      await writeDriverProfileCache(profile);
      return profile;
    }
    return {};
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function updateDriverPhoto(
  payload: string | { photoBase64: string; mimeType?: string }
): Promise<DriverProfile> {
  try {
    const body =
      typeof payload === "string"
        ? { photo: payload }
        : {
            photo: payload.photoBase64,
            mime_type: payload.mimeType,
          };
    const { data } = await apiClient.put("/driver/me/photo", body);
    if (data && typeof data === "object") {
      const payloadData = data as { profile?: DriverProfile } & DriverProfile;
      const profile =
        payloadData.profile && typeof payloadData.profile === "object"
          ? payloadData.profile
          : payloadData;
      await writeDriverProfileCache(profile);
      return profile;
    }
    return {};
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function getDriverRoute(): Promise<DriverRouteSnapshot> {
  try {
    const { data } = await apiClient.get("/driver/me/route");
    if (data && typeof data === "object") {
      return data as DriverRouteSnapshot;
    }
    return {};
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function getDriverBookingsAll(): Promise<DriverMission[]> {
  try {
    const { data } = await apiClient.get("/driver/me/bookings/all");
    return asArray(data);
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function getDriverCompanyBookingsToday(): Promise<DriverMission[]> {
  try {
    const { data } = await apiClient.get("/driver/me/company-bookings/today");
    return asArray(data);
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function getDriverCompletedTrips(driverId: number): Promise<DriverCompletedTrip[]> {
  try {
    const { data } = await apiClient.get(`/drivers/${driverId}/completed-trips`);
    if (Array.isArray(data)) return data as DriverCompletedTrip[];
    if (data && typeof data === "object") {
      const payload = data as Record<string, unknown>;
      if (Array.isArray(payload.items)) return payload.items as DriverCompletedTrip[];
      if (Array.isArray(payload.results)) return payload.results as DriverCompletedTrip[];
      if (Array.isArray(payload.data)) return payload.data as DriverCompletedTrip[];
      if (Array.isArray(payload.bookings)) return payload.bookings as DriverCompletedTrip[];
    }
    return [];
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function triggerDriverTestPush(): Promise<void> {
  try {
    await apiClient.post("/driver/me/test-push");
  } catch (error) {
    throw normalizeError(error);
  }
}

