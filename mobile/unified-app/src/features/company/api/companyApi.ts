import { AxiosError } from "axios";
import { apiClient } from "../../../core/api/client";
import type {
  CompanyDispatchRealtimeDashboard,
  CompanyDispatchMission,
  CompanyDispatchMissionListResponse,
  CompanyDriverLiveLocation,
  CompanyDriverLiveLocationResponse,
  CompanyOptimizerStatusResponse,
} from "./contracts";
import {
  validateCompanyDriverLocationsResponse,
  validateCompanyDispatchRealtimeDashboard,
  validateCompanyMissionListResponse,
  validateCompanyOptimizerStatusResponse,
} from "./contracts";
import { emitCompanyDispatchTelemetry } from "../telemetry/companyTelemetry";
import { mergeCompanyDispatchDelaySources } from "../utils/dispatchWebAlignment";
import { filterMissionsByDispatchListChip } from "../utils/rideListStatusFilter";

type CompanyRequestOptions = {
  contextId: string;
};

const GENERIC_RESTX_CONFLICT =
  "A conflict happened while processing the request. The resource might have been modified while the request was being processed.";

/**
 * Extrait un libellé lisible depuis les erreurs API dispatch (Axios),
 * y compris quand Flask-RESTX renvoie le message 409 générique.
 */
export function getDispatchApiErrorMessage(error: unknown, fallback: string): string {
  if (!(error instanceof Error) || !(error as AxiosError).isAxiosError) {
    return error instanceof Error ? error.message : fallback;
  }
  const ax = error as AxiosError<unknown>;
  const st = ax.response?.status;
  if (st === 401) {
    return "Session expirée ou non autorisée. Reconnectez-vous.";
  }
  if (st === 403) {
    return "Accès refusé pour cette ressource ou cette action.";
  }
  if (st === 404) {
    return "Ressource introuvable sur le serveur. Actualisez la page ou contactez le support.";
  }
  const d = ax.response?.data;
  if (d && typeof d === "object") {
    const msg = (d as Record<string, unknown>).message;
    if (typeof msg === "string" && msg.trim().length > 0) {
      if (msg === GENERIC_RESTX_CONFLICT || (st === 409 && /conflict happened while processing/i.test(msg))) {
        return "Conflit d’assignation (créneau, disponibilité chauffeur, ou course modifiée entretemps). Actualisez la liste, puis réessayez ou choisissez un autre chauffeur.";
      }
      return msg.trim();
    }
  }
  if (st === 409) {
    return "Conflit d’assignation. Actualisez la liste des courses, puis réessayez ou choisissez un autre chauffeur.";
  }
  if (typeof d === "string" && d.trim().length > 0) {
    return d.trim();
  }
  return error.message && !error.message.startsWith("Request failed with status code")
    ? error.message
    : fallback;
}

function shouldTryFallback(error: unknown, domain?: string): boolean {
  const axiosError = error as AxiosError | undefined;
  const status = axiosError?.response?.status;
  const body = axiosError?.response?.data as { error?: unknown; message?: unknown } | undefined;
  const errorCode = typeof body?.error === "string" ? body.error.toLowerCase() : "";
  const message = typeof body?.message === "string" ? body.message.toLowerCase() : "";
  if (
    domain === "dispatch_ride_create" &&
    status === 400 &&
    (errorCode === "invalid_json" || message.includes("json"))
  ) {
    return true;
  }
  return status === 404 || status === 405 || status === 501;
}

async function requestWithFallback<T>(
  requests: (() => Promise<{ data: T }>)[],
  trace?: { domain: string; contextId?: string }
): Promise<{ data: T }> {
  let lastError: unknown = null;
  for (let index = 0; index < requests.length; index += 1) {
    try {
      if (index > 0) {
        emitCompanyDispatchTelemetry(
          "company.dispatch.contract_fallback",
          {
            source: "companyApi.requestWithFallback",
            domain: trace?.domain ?? "unknown",
            context_id: trace?.contextId ?? null,
            attempt: index + 1,
          },
          { allowWhenDisabled: true }
        );
      }
      return await requests[index]();
    } catch (error) {
      lastError = error;
      const status =
        typeof (error as AxiosError)?.response?.status === "number"
          ? (error as AxiosError).response?.status
          : null;
      if (index === requests.length - 1 || !shouldTryFallback(error, trace?.domain)) {
        if (status === 401 || status === 403) {
          emitCompanyDispatchTelemetry(
            "company.dispatch.auth_failure",
            {
              source: "companyApi.requestWithFallback",
              domain: trace?.domain ?? "unknown",
              context_id: trace?.contextId ?? null,
              status,
            },
            { allowWhenDisabled: true }
          );
          throw error;
        }
        emitCompanyDispatchTelemetry(
          "company.dispatch.contract_failure",
          {
            source: "companyApi.requestWithFallback",
            domain: trace?.domain ?? "unknown",
            context_id: trace?.contextId ?? null,
            status,
          },
          { allowWhenDisabled: true }
        );
        throw error;
      }
    }
  }
  throw lastError instanceof Error ? lastError : new Error("request_fallback_exhausted");
}

/**
 * POST /company_mobile/dispatch/v1/rides/:id/… — aligné sur Flask (`company_mobile_dispatch` ns).
 * Repli sur /dispatch/v1/… seulement si 404/405/501.
 */
export async function postCompanyDispatchRideAction(
  options: CompanyRequestOptions & { missionId: number },
  subPath: "/assign" | "/reassign" | "/cancel" | "/schedule" | string,
  body: unknown,
  domain: string
): Promise<void> {
  const id = options.missionId;
  await requestWithFallback(
    [
      () => apiClient.post(`/company_mobile/dispatch/v1/rides/${id}${subPath}`, body, withContextHeaders(options)),
      () => apiClient.post(`/dispatch/v1/rides/${id}${subPath}`, body, withContextHeaders(options)),
    ],
    { domain, contextId: options.contextId }
  );
}

export const COMPANY_BACKEND_ENDPOINTS = {
  missions: "/dispatch/v1/rides",
  realtimeDashboard: "/dispatch/v1/dashboard/realtime",
  dispatchStatus: "/dispatch/v1/status",
  optimizerStatus: "/dispatch/v1/status",
  driversLocations: "/companies/me/drivers/locations",
  driversLive: "/companies/me/drivers/live",
} as const;

type RawRide = {
  id?: number | string;
  mission_id?: number | string;
  booking_id?: number | string;
  company_id?: number | string | null;
  status?: string | null;
  updated_at?: string | null;
  pickup_at?: string | null;
  scheduled_at?: string | null;
  pickup_address?: string | null;
  dropoff_address?: string | null;
  pickup_lat?: number | string | null;
  pickup_lng?: number | string | null;
  pickup_lon?: number | string | null;
  dropoff_lat?: number | string | null;
  dropoff_lng?: number | string | null;
  dropoff_lon?: number | string | null;
  distance_km?: number | string | null;
  route_distance_km?: number | string | null;
  duration_min?: number | string | null;
  route_duration_min?: number | string | null;
  duration_seconds?: number | string | null;
  time?: { pickup_at?: string | null };
  client?: { name?: string | null } | null;
  client_name?: string | null;
  identity?: CompanyDispatchMission["identity"];
  trip_flags?: CompanyDispatchMission["trip_flags"];
  scheduling?: CompanyDispatchMission["scheduling"];
  time_confirmed?: boolean | null;
  display_model?: string | null;
  display_model_version?: number | null;
  search_index?: unknown[] | null;
  route?: {
    pickup_address?: string | null;
    dropoff_address?: string | null;
    distance_km?: number | string | null;
    duration_min?: number | string | null;
    route_duration_min?: number | string | null;
    duration_seconds?: number | string | null;
    pickup_lat?: number | string | null;
    pickup_lon?: number | string | null;
    dropoff_lat?: number | string | null;
    dropoff_lon?: number | string | null;
  };
  driver?: {
    id?: string | number | null;
    name?: string | null;
    display_name?: string | null;
    driver_type?: string | null;
    is_emergency?: boolean;
  } | null;
  transfer?: {
    partner_company_name?: string | null;
  } | null;
  assignment_pickup_delay_minutes?: number | string | null;
};

type RawDispatchRidesResponse = {
  items?: RawRide[];
  missions?: RawRide[];
};

type RawOptimizerResponse = {
  optimizer?: {
    active?: boolean;
    running?: boolean;
    last_tick?: string | null;
    next_window_start?: string | null;
  };
};

type RawDriversResponse = {
  locations?: {
    driver_id?: number | string;
    driver_name?: string;
    full_name?: string;
    first_name?: string;
    last_name?: string;
    latitude?: number;
    longitude?: number;
    lat?: number;
    lon?: number;
    lng?: number;
    timestamp?: string;
    recorded_at?: string;
    received_at?: string;
    mission_id?: number | null;
    current_booking_id?: number | null;
  }[];
  drivers?: {
    driver_id?: number | string;
    driver_name?: string;
    full_name?: string;
    first_name?: string;
    last_name?: string;
    latitude?: number;
    longitude?: number;
    lat?: number;
    lon?: number;
    lng?: number;
    timestamp?: string;
    recorded_at?: string;
    received_at?: string;
    mission_id?: number | null;
  }[];
};

/** Ne synthétise jamais « maintenant » — absence → chaîne vide (Phase 0A fraîcheur). */
function toIso(input: unknown): string {
  if (typeof input === "string" && input.length > 0) return input;
  return "";
}

function toFiniteNumber(input: unknown): number | null {
  if (typeof input === "number" && Number.isFinite(input)) return input;
  if (typeof input === "string") {
    const parsed = Number.parseFloat(input);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function parseDistanceKm(input: unknown): number | null {
  if (typeof input === "number" && Number.isFinite(input)) {
    return input > 0 ? input : null;
  }
  if (typeof input === "string") {
    const normalized = input.trim().toLowerCase().replace(",", ".");
    const match = normalized.match(/-?\d+(\.\d+)?/);
    if (!match) return null;
    const parsed = Number.parseFloat(match[0]);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
  }
  return null;
}

function parseDurationMin(input: unknown): number | null {
  if (typeof input === "number" && Number.isFinite(input)) {
    return input > 0 ? input : null;
  }
  if (typeof input === "string") {
    const normalized = input.trim().toLowerCase().replace(",", ".");
    const match = normalized.match(/-?\d+(\.\d+)?/);
    if (!match) return null;
    const parsed = Number.parseFloat(match[0]);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
  }
  return null;
}

function normalizeMission(raw: RawRide): CompanyDispatchMission | null {
  /** Même identifiant que `/company_dispatch/delays` et les routes `/rides/:id` (booking). */
  const missionId = toFiniteNumber(raw.booking_id ?? raw.id ?? raw.mission_id);
  if (missionId == null) return null;
  let status = String(raw.status ?? "pending").toLowerCase();
  // Backend `Booking` : aller-retour terminé, pas une course « à venir » (avant: inconnu -> pending).
  if (status === "return_completed") {
    status = "completed";
  } else if (status === "canceled") {
    status = "cancelled";
  } else if (status === "awaiting_client_payment" || status === "awaiting_payment") {
    status = "pending";
  } else if (status === "onboard" || status === "en_route_dropoff") {
    status = "in_progress";
  } else if (status === "en_route_pickup") {
    status = "en_route";
  }
  const normalizedStatus: CompanyDispatchMission["status"] =
    status === "pending" ||
    status === "proposed" ||
    status === "accepted" ||
    status === "assigned" ||
    status === "en_route" ||
    status === "arrived" ||
    status === "in_progress" ||
    status === "completed" ||
    status === "cancelled"
      ? status
      : "pending";

  const driverId = toFiniteNumber(raw.driver?.id ?? null);
  const companyId = toFiniteNumber(raw.company_id ?? null);
  const clientName =
    (raw.identity &&
      typeof raw.identity === "object" &&
      raw.identity.passenger &&
      typeof raw.identity.passenger === "object" &&
      typeof raw.identity.passenger.name === "string" &&
      raw.identity.passenger.name.trim())
      ? raw.identity.passenger.name.trim()
      : raw.client && typeof raw.client === "object" && typeof raw.client.name === "string" && raw.client.name.trim()
        ? raw.client.name.trim()
        : typeof raw.client_name === "string" && raw.client_name.trim()
          ? raw.client_name.trim()
          : null;
  const driverNameRaw = raw.driver && typeof raw.driver === "object" ? raw.driver : null;
  const driverNameFromApi =
    typeof driverNameRaw?.display_name === "string" && driverNameRaw.display_name.trim()
      ? driverNameRaw.display_name.trim()
      : typeof driverNameRaw?.name === "string" && driverNameRaw.name.trim()
        ? driverNameRaw.name.trim()
        : null;
  const transferRaw = raw.transfer && typeof raw.transfer === "object" ? raw.transfer : null;
  const partnerCompanyName =
    typeof transferRaw?.partner_company_name === "string" && transferRaw.partner_company_name.trim()
      ? transferRaw.partner_company_name.trim()
      : null;
  const assignmentDelay = toFiniteNumber(raw.assignment_pickup_delay_minutes);
  const routeDistance =
    parseDistanceKm(raw.route?.distance_km) ??
    parseDistanceKm(raw.route_distance_km) ??
    parseDistanceKm(raw.distance_km);
  const routeDurationMin =
    parseDurationMin(raw.route?.duration_min) ??
    parseDurationMin(raw.route?.route_duration_min) ??
    parseDurationMin(raw.route_duration_min) ??
    parseDurationMin(raw.duration_min) ??
    (() => {
      const sec =
        toFiniteNumber(raw.route?.duration_seconds) ??
        toFiniteNumber(raw.duration_seconds);
      return sec != null && sec > 0 ? Math.round(sec / 60) : null;
    })();
  const pickupLabel =
    (typeof raw.route?.pickup_address === "string" && raw.route?.pickup_address.trim()) ||
    (typeof raw.pickup_address === "string" && raw.pickup_address.trim()) ||
    null;
  const dropoffLabel =
    (typeof raw.route?.dropoff_address === "string" && raw.route?.dropoff_address.trim()) ||
    (typeof raw.dropoff_address === "string" && raw.dropoff_address.trim()) ||
    null;
  const pickupLat = toFiniteNumber(raw.route?.pickup_lat ?? raw.pickup_lat);
  const pickupLon = toFiniteNumber(raw.route?.pickup_lon ?? raw.pickup_lon ?? raw.pickup_lng);
  const dropoffLat = toFiniteNumber(raw.route?.dropoff_lat ?? raw.dropoff_lat);
  const dropoffLon = toFiniteNumber(raw.route?.dropoff_lon ?? raw.dropoff_lon ?? raw.dropoff_lng);
  const scheduledAt = raw.time?.pickup_at ?? raw.pickup_at ?? raw.scheduled_at ?? null;
  const identityRaw = raw.identity && typeof raw.identity === "object" ? raw.identity : null;
  const tripFlagsRaw = raw.trip_flags && typeof raw.trip_flags === "object" ? raw.trip_flags : null;
  const schedulingRaw = raw.scheduling && typeof raw.scheduling === "object" ? raw.scheduling : null;
  const searchIndexRaw = Array.isArray(raw.search_index) ? raw.search_index.filter((t) => typeof t === "string") : null;
  return {
    mission_id: missionId,
    status: normalizedStatus,
    scheduled_at: scheduledAt,
    client_name: clientName,
    identity: identityRaw,
    trip_flags: tripFlagsRaw,
    scheduling: schedulingRaw,
    time_confirmed:
      typeof raw.time_confirmed === "boolean"
        ? raw.time_confirmed
        : schedulingRaw && typeof schedulingRaw === "object" && "time_confirmed" in schedulingRaw
          ? Boolean((schedulingRaw as { time_confirmed?: boolean }).time_confirmed)
          : null,
    search_index: searchIndexRaw,
    pickup_label: pickupLabel,
    dropoff_label: dropoffLabel,
    pickup_lat: pickupLat,
    pickup_lon: pickupLon,
    dropoff_lat: dropoffLat,
    dropoff_lon: dropoffLon,
    route_distance_km: routeDistance != null && routeDistance > 0 ? routeDistance : null,
    route_duration_min: routeDurationMin != null && routeDurationMin > 0 ? routeDurationMin : null,
    driver_name: driverNameFromApi,
    driver_id: driverId,
    partner_company_name: partnerCompanyName,
    driver_type:
      typeof raw.driver?.driver_type === "string" && raw.driver.driver_type.trim()
        ? raw.driver.driver_type.trim().toUpperCase()
        : raw.driver?.is_emergency
          ? "EMERGENCY"
          : null,
    company_id: companyId,
    updated_at: raw.updated_at ?? null,
    assignment_pickup_delay_minutes:
      assignmentDelay != null && assignmentDelay > 0 ? assignmentDelay : null,
  };
}

function normalizeLocation(
  raw: NonNullable<RawDriversResponse["locations"]>[number] & {
    id?: number | string;
    status?: string | null;
    tracking_display_status?: string | null;
    position_source?: string | null;
    presence_status?: string | null;
    device_health?: CompanyDriverLiveLocation["device_health"];
  }
): CompanyDriverLiveLocation | null {
  const driverId = toFiniteNumber(raw.driver_id ?? raw.id);
  // Roster sans GPS autorisé — jamais de placeholder 0,0.
  const latitude = toFiniteNumber(raw.latitude ?? raw.lat);
  const longitude = toFiniteNumber(raw.longitude ?? raw.lon ?? raw.lng);
  if (driverId == null) return null;
  const firstName =
    typeof raw.first_name === "string" && raw.first_name.trim()
      ? raw.first_name.trim()
      : null;
  const lastName =
    typeof raw.last_name === "string" && raw.last_name.trim()
      ? raw.last_name.trim()
      : null;
  const mergedFromParts = [firstName, lastName].filter(Boolean).join(" ").trim() || null;
  const fullName =
    typeof raw.full_name === "string" && raw.full_name.trim()
      ? raw.full_name.trim()
      : mergedFromParts;
  const driverName =
    typeof raw.driver_name === "string" && raw.driver_name.trim()
      ? raw.driver_name.trim()
      : fullName;
  const timestamp = toIso(raw.timestamp);
  const recordedAt = toIso(raw.recorded_at ?? raw.timestamp);
  const receivedAt = toIso(raw.received_at);
  const lastSeenSeconds = toFiniteNumber(
    (raw as { last_seen_seconds?: unknown }).last_seen_seconds
  );
  const locationStatus =
    typeof (raw as { location_status?: unknown }).location_status === "string"
      ? String((raw as { location_status: string }).location_status)
      : null;
  const isStale =
    typeof (raw as { is_stale?: unknown }).is_stale === "boolean"
      ? Boolean((raw as { is_stale: boolean }).is_stale)
      : locationStatus === "stale" || locationStatus === "offline";
  const positionSource =
    typeof raw.position_source === "string" && raw.position_source.trim()
      ? raw.position_source.trim().toLowerCase()
      : null;
  const trackingDisplay =
    typeof raw.tracking_display_status === "string" && raw.tracking_display_status.trim()
      ? raw.tracking_display_status.trim().toLowerCase()
      : null;
  const presenceStatus =
    typeof raw.presence_status === "string" && raw.presence_status.trim()
      ? raw.presence_status.trim().toLowerCase()
      : null;
  const status =
    typeof raw.status === "string" && raw.status.trim()
      ? raw.status.trim().toLowerCase()
      : null;
  return {
    driver_id: driverId,
    driver_name: driverName,
    full_name: fullName ?? driverName,
    first_name: firstName,
    last_name: lastName,
    mission_id:
      toFiniteNumber(raw.mission_id ?? raw.current_booking_id) ?? null,
    latitude: latitude,
    longitude: longitude,
    timestamp: timestamp || undefined,
    recorded_at: recordedAt || timestamp || null,
    received_at: receivedAt || null,
    last_seen_seconds: lastSeenSeconds,
    location_status: locationStatus as CompanyDriverLiveLocation["location_status"],
    is_stale: isStale,
    accept_status:
      typeof (raw as { accept_status?: unknown }).accept_status === "string"
        ? String((raw as { accept_status: string }).accept_status)
        : null,
    position_source: positionSource,
    tracking_display_status: trackingDisplay,
    presence_status: presenceStatus,
    status: status,
    device_health:
      raw.device_health && typeof raw.device_health === "object"
        ? raw.device_health
        : undefined,
  };
}

function buildHeaders(contextId: string) {
  return {
    "X-Active-Context-Id": contextId,
  };
}

export async function getDispatchMissions(
  options: CompanyRequestOptions & {
    date: string;
    search?: string;
    status?: string;
  }
): Promise<CompanyDispatchMissionListResponse> {
  const response = await requestWithFallback<RawDispatchRidesResponse>([
    () =>
      apiClient.get<RawDispatchRidesResponse>("/company_mobile/dispatch/v1/rides", {
        headers: buildHeaders(options.contextId),
        params: {
          date: options.date,
          q: options.search || undefined,
          status: options.status || undefined,
          page_size: 50,
        },
      }),
    () =>
      apiClient.get<RawDispatchRidesResponse>(COMPANY_BACKEND_ENDPOINTS.missions, {
        headers: buildHeaders(options.contextId),
        params: {
          date: options.date,
          q: options.search || undefined,
          status: options.status || undefined,
          page_size: 50,
        },
      }),
  ], { domain: "dispatch_missions_get", contextId: options.contextId });
  const rawMissions = response.data.items ?? response.data.missions ?? [];
  const missions = rawMissions
    .map(normalizeMission)
    .filter((mission): mission is CompanyDispatchMission => mission !== null);
  const afterChip = filterMissionsByDispatchListChip(
    missions,
    options.status ?? "all"
  );
  const normalized: CompanyDispatchMissionListResponse = {
    context_id: options.contextId,
    missions: afterChip,
    refreshed_at: new Date().toISOString(),
  };
  if (!validateCompanyMissionListResponse(normalized)) {
    throw new Error("Company missions payload contract mismatch");
  }
  return normalized;
}

export async function getRealtimeDashboard(
  options: CompanyRequestOptions & { date: string }
): Promise<CompanyDispatchRealtimeDashboard> {
  const response = await requestWithFallback<CompanyAnyPayload>([
    () =>
      apiClient.get("/company_mobile/dispatch/v1/dashboard/realtime", {
        headers: buildHeaders(options.contextId),
        params: { date: options.date },
      }),
    () =>
      apiClient.get(COMPANY_BACKEND_ENDPOINTS.realtimeDashboard, {
        headers: buildHeaders(options.contextId),
        params: { date: options.date },
      }),
  ], { domain: "dispatch_dashboard_realtime_get", contextId: options.contextId });
  const payload = (response.data ?? {}) as {
    stats?: { delayed_bookings?: number };
    opportunities?: unknown[];
    quality_metrics?: { avg_delay?: number };
    timestamp?: string;
  };

  const delayedFromApi = typeof payload.stats?.delayed_bookings === "number";
  const hasOpportunitiesArray = Array.isArray(payload.opportunities);
  const normalized: CompanyDispatchRealtimeDashboard = {
    context_id: options.contextId,
    refreshed_at: toIso(payload.timestamp),
    delayed_bookings_metrics_available: delayedFromApi,
    delayed_bookings: delayedFromApi ? (payload.stats!.delayed_bookings as number) : 0,
    opportunities_metrics_available: hasOpportunitiesArray,
    opportunities: hasOpportunitiesArray ? (payload.opportunities as unknown[]).length : 0,
    avg_delay_minutes: payload.quality_metrics?.avg_delay ?? 0,
  };
  if (!validateCompanyDispatchRealtimeDashboard(normalized)) {
    throw new Error("Company dashboard payload contract mismatch");
  }
  return normalized;
}

export async function getOptimizerStatus(
  options: CompanyRequestOptions
): Promise<CompanyOptimizerStatusResponse> {
  const response = await requestWithFallback<RawOptimizerResponse>([
    () =>
      apiClient.get<RawOptimizerResponse>("/company_mobile/dispatch/v1/status", {
        headers: buildHeaders(options.contextId),
      }),
    () =>
      apiClient.get<RawOptimizerResponse>(COMPANY_BACKEND_ENDPOINTS.optimizerStatus, {
        headers: buildHeaders(options.contextId),
      }),
  ], { domain: "dispatch_optimizer_status_get", contextId: options.contextId });
  const optimizer = response.data.optimizer ?? {};
  const isRunning = Boolean(optimizer.running ?? optimizer.active);
  const normalized: CompanyOptimizerStatusResponse = {
    context_id: options.contextId,
    refreshed_at: new Date().toISOString(),
    status: {
      optimizer_enabled: Boolean(optimizer.active ?? optimizer.running),
      optimizer_state: isRunning ? "running" : "idle",
      last_run_at: optimizer.last_tick ?? null,
      last_success_at: optimizer.next_window_start ?? null,
      reason: null,
    },
  };
  if (!validateCompanyOptimizerStatusResponse(normalized)) {
    throw new Error("Company optimizer payload contract mismatch");
  }
  return normalized;
}

async function fetchDriverLocationsEndpoint(
  contextId: string,
  endpoint: string
): Promise<CompanyDriverLiveLocationResponse> {
  const response = await apiClient.get<RawDriversResponse>(endpoint, {
    headers: buildHeaders(contextId),
  });
  const rawRows = response.data.locations ?? response.data.drivers ?? [];
  const locations = rawRows
    .map((row) =>
      normalizeLocation({
        ...row,
        driver_id: row.driver_id ?? (row as { id?: number | string }).id,
      })
    )
    .filter((location): location is CompanyDriverLiveLocation => location !== null);
  const normalized: CompanyDriverLiveLocationResponse = {
    context_id: contextId,
    locations,
    refreshed_at: new Date().toISOString(),
  };
  if (!validateCompanyDriverLocationsResponse(normalized)) {
    throw new Error("Company drivers payload contract mismatch");
  }
  return normalized;
}

export async function getDriversLocationsSnapshot(
  options: CompanyRequestOptions
): Promise<CompanyDriverLiveLocationResponse> {
  try {
    return await fetchDriverLocationsEndpoint(
      options.contextId,
      COMPANY_BACKEND_ENDPOINTS.driversLive
    );
  } catch {
    return await fetchDriverLocationsEndpoint(
      options.contextId,
      COMPANY_BACKEND_ENDPOINTS.driversLocations
    );
  }
}

type CompanyAnyPayload = Record<string, unknown>;

export type CompanyScheduleRidePayload = {
  pickup_at: string;
  timezone?: string;
  force_recompute?: boolean;
  note?: string | null;
};

export type CompanyCancelRidePayload = {
  reason_code: string;
  note?: string | null;
  reason?: string | null;
};

export type CompanyMarkUrgentPayload = {
  urgent?: boolean;
  reason_code?: string | null;
  note?: string | null;
  source?: string | null;
  /** Décalage planification (défaut 15) — aligné sur POST company_mobile/dispatch/…/urgent. */
  extra_delay_minutes?: number;
  /** Surcharge de `reason_code` côté API (champ `reason` Flask). */
  reason?: string | null;
};

export type CompanyDispatchStatusResponse = {
  context_id: string;
  refreshed_at: string;
  dispatch_mode: "manual" | "semi_auto" | "fully_auto" | "unknown";
  dispatch_state: "idle" | "running" | "degraded" | "failed" | "unknown";
  source: "scheduler_runtime";
};

function withContextHeaders(options: CompanyRequestOptions) {
  return { headers: buildHeaders(options.contextId) };
}

export async function getCompanyDispatchDelays(
  options: CompanyRequestOptions & { date: string }
): Promise<unknown[]> {
  const params = { date: options.date };
  const headersPayload = withContextHeaders(options);

  let liveRows: unknown[] = [];
  try {
    const live = await apiClient.get<{ delays?: unknown[] }>("/company_dispatch/delays/live", {
      ...headersPayload,
      params,
    });
    if (Array.isArray(live.data?.delays)) {
      liveRows = live.data!.delays!;
    }
  } catch {
    liveRows = [];
  }

  let snapshotRows: unknown[] = [];
  try {
    const response = await apiClient.get<unknown[]>("/company_dispatch/delays", {
      ...headersPayload,
      params,
    });
    if (Array.isArray(response.data)) {
      snapshotRows = response.data;
    }
  } catch {
    snapshotRows = [];
  }

  return mergeCompanyDispatchDelaySources(liveRows, snapshotRows);
}

function stripNullishFields(payload: Record<string, unknown>): Record<string, unknown> {
  const next: Record<string, unknown> = {};
  Object.entries(payload).forEach(([key, value]) => {
    if (value === null || value === undefined) return;
    next[key] = value;
  });
  return next;
}

async function resolveReservationIdFromMission(
  options: CompanyRequestOptions & { missionId: number }
): Promise<number> {
  const fallbackId = options.missionId;
  const findNumericByKeys = (root: unknown, keys: string[]): number | null => {
    const queue: unknown[] = [root];
    const seen = new Set<unknown>();
    while (queue.length > 0) {
      const current = queue.shift();
      if (!current || typeof current !== "object" || seen.has(current)) continue;
      seen.add(current);
      const row = current as Record<string, unknown>;
      for (const key of keys) {
        const parsed = toFiniteNumber(row[key]);
        if (parsed != null) return parsed;
      }
      Object.values(row).forEach((value) => {
        if (value && typeof value === "object") queue.push(value);
      });
    }
    return null;
  };
  try {
    const detail = await getCompanyRideDetail({
      contextId: options.contextId,
      missionId: options.missionId,
    });
    if (!detail || typeof detail !== "object") return fallbackId;
    const payload = detail as Record<string, unknown>;
    const deepBookingId = findNumericByKeys(payload, ["booking_id", "reservation_id", "bookingId", "reservationId"]);
    if (deepBookingId != null) return deepBookingId;
    const directCandidate =
      (payload.summary as Record<string, unknown> | undefined) ??
      (payload.data as Record<string, unknown> | undefined) ??
      (payload.item as Record<string, unknown> | undefined) ??
      (payload.ride as Record<string, unknown> | undefined) ??
      (payload.mission as Record<string, unknown> | undefined) ??
      payload;
    const directBookingId = toFiniteNumber(
      directCandidate?.booking_id ?? directCandidate?.reservation_id ?? directCandidate?.id
    );
    if (directBookingId != null) return directBookingId;
  } catch {
    // Best effort: si la résolution échoue, conserver missionId.
  }
  return fallbackId;
}

export async function runCompanyDispatch(options: CompanyRequestOptions & { date?: string }) {
  const response = await requestWithFallback<CompanyAnyPayload>(
    [
      () =>
        apiClient.post(
          "/company_mobile/dispatch/v1/run",
          { date: options.date ?? undefined },
          withContextHeaders(options)
        ),
      () =>
        apiClient.post(
          "/company_dispatch/run",
          { date: options.date ?? undefined },
          withContextHeaders(options)
        ),
    ],
    { domain: "dispatch_run_post", contextId: options.contextId }
  );
  return response.data as CompanyAnyPayload;
}

export async function runCompanyOptimizer(options: CompanyRequestOptions & { date?: string }) {
  const response = await requestWithFallback<CompanyAnyPayload>(
    [
      () =>
        apiClient.post(
          "/company_mobile/dispatch/v1/optimizer/run",
          { date: options.date ?? undefined },
          withContextHeaders(options)
        ),
      () =>
        apiClient.post(
          "/company_dispatch/optimizer/start",
          { date: options.date ?? undefined },
          withContextHeaders(options)
        ),
    ],
    { domain: "dispatch_optimizer_run_post", contextId: options.contextId }
  );
  return response.data as CompanyAnyPayload;
}

export async function getDispatchStatus(
  options: CompanyRequestOptions & { date?: string }
): Promise<CompanyDispatchStatusResponse> {
  const response = await requestWithFallback<CompanyAnyPayload>([
    () =>
      apiClient.get<CompanyAnyPayload>("/company_mobile/dispatch/v1/status", {
        headers: buildHeaders(options.contextId),
        params: { date: options.date ?? undefined },
      }),
    () =>
      apiClient.get<CompanyAnyPayload>(COMPANY_BACKEND_ENDPOINTS.dispatchStatus, {
        headers: buildHeaders(options.contextId),
        params: { date: options.date ?? undefined },
      }),
  ], { domain: "dispatch_status_get", contextId: options.contextId });
  const payload = (response.data ?? {}) as CompanyAnyPayload;
  const modeCandidate =
    payload.dispatch_mode ?? payload.mode ?? payload.current_mode ?? payload.target_mode;
  const agentCandidate =
    payload.agent && typeof payload.agent === "object"
      ? (payload.agent as Record<string, unknown>)
      : null;
  const osrmCandidate =
    payload.osrm && typeof payload.osrm === "object"
      ? (payload.osrm as Record<string, unknown>)
      : null;
  const stateCandidate = payload.dispatch_state ?? payload.state ?? payload.status;
  const normalizedModeCandidate =
    typeof modeCandidate === "string" ? modeCandidate.toLowerCase() : null;
  const agentModeCandidate =
    typeof agentCandidate?.mode === "string" ? agentCandidate.mode.toLowerCase() : null;
  const effectiveMode = normalizedModeCandidate ?? agentModeCandidate;
  const dispatch_mode: CompanyDispatchStatusResponse["dispatch_mode"] =
    effectiveMode === "manual" ||
    effectiveMode === "semi_auto" ||
    effectiveMode === "fully_auto"
      ? effectiveMode
      : "unknown";
  const agentActive =
    typeof agentCandidate?.active === "boolean" ? agentCandidate.active : null;
  const osrmStatus =
    typeof osrmCandidate?.status === "string" ? osrmCandidate.status.toUpperCase() : null;
  const dispatch_state: CompanyDispatchStatusResponse["dispatch_state"] =
    stateCandidate === "idle" ||
    stateCandidate === "running" ||
    stateCandidate === "degraded" ||
    stateCandidate === "failed"
      ? stateCandidate
      : agentActive === true
        ? "running"
        : agentActive === false
          ? osrmStatus === "DOWN"
            ? "degraded"
            : "idle"
          : osrmStatus === "DOWN"
            ? "degraded"
      : "unknown";
  return {
    context_id: options.contextId,
    refreshed_at: new Date().toISOString(),
    dispatch_mode,
    dispatch_state,
    source: "scheduler_runtime",
  };
}

export async function assignCompanyRide(
  options: CompanyRequestOptions & { missionId: number; driverId: number }
) {
  try {
    const id = options.missionId;
    const reservationId = await resolveReservationIdFromMission(options);
    const reservationCandidates = Array.from(new Set([reservationId, id])).filter((v) =>
      Number.isFinite(v)
    );
    let reservationAssignError: unknown = null;
    for (const candidateId of reservationCandidates) {
      try {
        await apiClient.post(
          `/companies/me/reservations/${candidateId}/assign`,
          { driver_id: options.driverId },
          withContextHeaders(options)
        );
        return;
      } catch (error) {
        reservationAssignError = error;
      }
    }
    await requestWithFallback(
      [
        () =>
          apiClient.post(
            `/company_mobile/dispatch/v1/rides/${id}/assign`,
            { driver_id: options.driverId },
            withContextHeaders(options)
          ),
        () =>
          apiClient.post(
            `/dispatch/v1/rides/${id}/assign`,
            { driver_id: options.driverId },
            withContextHeaders(options)
          ),
      ],
      { domain: "dispatch_assign_post", contextId: options.contextId }
    );
    // Best effort: tenter de refléter l'événement live web même si l'assignation réelle
    // est passée par un endpoint dispatch.
    for (const candidateId of reservationCandidates) {
      try {
        await apiClient.post(
          `/companies/me/reservations/${candidateId}/assign`,
          { driver_id: options.driverId },
          withContextHeaders(options)
        );
        break;
      } catch {
        // no-op
      }
    }
    if (__DEV__ && reservationAssignError) {
      const status = (reservationAssignError as AxiosError | undefined)?.response?.status ?? null;
      console.log("[assignCompanyRide] fallback dispatch used", {
        missionId: id,
        reservationId,
        reservationCandidates,
        reservationAssignStatus: status,
      });
    }
  } catch (e) {
    throw new Error(getDispatchApiErrorMessage(e, "Assignation impossible."));
  }
}

export async function reassignCompanyRide(
  options: CompanyRequestOptions & { missionId: number; driverId: number }
) {
  try {
    const id = options.missionId;
    const reservationId = await resolveReservationIdFromMission(options);
    const reservationCandidates = Array.from(new Set([reservationId, id])).filter((v) =>
      Number.isFinite(v)
    );
    let reservationAssignError: unknown = null;
    for (const candidateId of reservationCandidates) {
      try {
        await apiClient.post(
          `/companies/me/reservations/${candidateId}/assign`,
          { driver_id: options.driverId },
          withContextHeaders(options)
        );
        return;
      } catch (error) {
        reservationAssignError = error;
      }
    }
    await requestWithFallback(
      [
        () =>
          apiClient.post(
            `/company_mobile/dispatch/v1/rides/${id}/reassign`,
            { driver_id: options.driverId },
            withContextHeaders(options)
          ),
        () =>
          apiClient.post(
            `/dispatch/v1/rides/${id}/reassign`,
            { driver_id: options.driverId },
            withContextHeaders(options)
          ),
      ],
      { domain: "dispatch_reassign_post", contextId: options.contextId }
    );
    for (const candidateId of reservationCandidates) {
      try {
        await apiClient.post(
          `/companies/me/reservations/${candidateId}/assign`,
          { driver_id: options.driverId },
          withContextHeaders(options)
        );
        break;
      } catch {
        // no-op
      }
    }
    if (__DEV__ && reservationAssignError) {
      const status = (reservationAssignError as AxiosError | undefined)?.response?.status ?? null;
      console.log("[reassignCompanyRide] fallback dispatch used", {
        missionId: id,
        reservationId,
        reservationCandidates,
        reservationAssignStatus: status,
      });
    }
  } catch (e) {
    throw new Error(getDispatchApiErrorMessage(e, "Réassignation impossible."));
  }
}

export async function cancelCompanyRide(
  options: CompanyRequestOptions &
    ({ missionId: number; reasonCode: string; note?: string | null; reason?: string } | {
      missionId: number;
      reason?: string;
      note?: string | null;
    })
) {
  const reasonCode =
    "reasonCode" in options && typeof options.reasonCode === "string"
      ? options.reasonCode
      : typeof options.reason === "string" && options.reason.trim().length > 0
        ? options.reason
        : "unspecified";
  const note =
    "note" in options && typeof options.note === "string" && options.note.trim().length > 0
      ? options.note.trim()
      : null;
  const id = options.missionId;
  const cancelPayload = {
    reason_code: reasonCode,
    note,
    reason: options.reason ?? reasonCode,
  } satisfies CompanyCancelRidePayload;
  await requestWithFallback(
    [
      () =>
        apiClient.delete(`/companies/me/reservations/${id}`, {
          ...withContextHeaders(options),
          data: cancelPayload,
        }),
      () =>
        apiClient.post(
          `/company_mobile/dispatch/v1/rides/${id}/cancel`,
          cancelPayload,
          withContextHeaders(options)
        ),
      () =>
        apiClient.post(
          `/dispatch/v1/rides/${id}/cancel`,
          cancelPayload,
          withContextHeaders(options)
        ),
    ],
    { domain: "dispatch_cancel_post", contextId: options.contextId }
  );
}

export async function scheduleCompanyRide(
  options: CompanyRequestOptions & { missionId: number; payload: CompanyScheduleRidePayload }
) {
  const id = options.missionId;
  const webSchedulePayload: Record<string, unknown> = {
    scheduled_time: options.payload.pickup_at,
  };
  if (options.payload.timezone) {
    webSchedulePayload.timezone = options.payload.timezone;
  }
  await requestWithFallback(
    [
      () =>
        apiClient.put(
          `/companies/me/reservations/${id}/schedule`,
          webSchedulePayload,
          withContextHeaders(options)
        ),
      () =>
        apiClient.post(
          `/company_mobile/dispatch/v1/rides/${id}/schedule`,
          options.payload,
          withContextHeaders(options)
        ),
      () =>
        apiClient.post(
          `/dispatch/v1/rides/${id}/schedule`,
          options.payload,
          withContextHeaders(options)
        ),
    ],
    { domain: "dispatch_schedule_post", contextId: options.contextId }
  );
}

export async function markCompanyRideUrgent(
  options: CompanyRequestOptions & { missionId: number; payload?: CompanyMarkUrgentPayload }
) {
  const p = options.payload;
  const extra_delay_minutes = typeof p?.extra_delay_minutes === "number" ? p.extra_delay_minutes : 15;
  const body: Record<string, unknown> = {
    urgent: p?.urgent ?? true,
    reason_code: p?.reason_code ?? null,
    note: p?.note ?? null,
    source: p?.source ?? "mobile_unified_company",
    extra_delay_minutes,
    // Backend (company_mobile_dispatch) : champs `extra_delay_minutes` + `reason` (audit).
    reason: p?.reason ?? p?.reason_code ?? null,
  };
  await requestWithFallback(
    [
      () =>
        apiClient.post(
          `/companies/me/reservations/${options.missionId}/dispatch-now`,
          { minutes_offset: extra_delay_minutes },
          withContextHeaders(options)
        ),
      () =>
        apiClient.post(
          `/company_mobile/dispatch/v1/rides/${options.missionId}/urgent`,
          body,
          withContextHeaders(options)
        ),
      () =>
        apiClient.post(
          `/dispatch/v1/rides/${options.missionId}/urgent`,
          body,
          withContextHeaders(options)
        ),
    ],
    { domain: "dispatch_urgent_post", contextId: options.contextId }
  );
}

export async function createCompanyRide(options: CompanyRequestOptions & { payload: CompanyAnyPayload }) {
  const debugTraceId = `create_company_ride_${Date.now()}`;
  const canonicalPayload: CompanyAnyPayload = {
    ...options.payload,
  };
  const isRecurringRequest =
    canonicalPayload.is_recurring === true ||
    (typeof canonicalPayload.recurrence_type === "string" &&
      canonicalPayload.recurrence_type.trim().length > 0);
  const webManualPayload: CompanyAnyPayload = stripNullishFields({
    ...canonicalPayload,
  });
  delete (webManualPayload as Record<string, unknown>).pickup_address;
  delete (webManualPayload as Record<string, unknown>).dropoff_address;
  delete (webManualPayload as Record<string, unknown>).is_return;
  const legacyPayload: CompanyAnyPayload = {
    ...canonicalPayload,
  };
  const pickupAddress = legacyPayload.pickup_address;
  if (pickupAddress && typeof pickupAddress === "object") {
    const pickupLabel =
      (pickupAddress as Record<string, unknown>).label ??
      (pickupAddress as Record<string, unknown>).address ??
      (pickupAddress as Record<string, unknown>).description;
    if (typeof pickupLabel === "string" && pickupLabel.trim().length > 0) {
      legacyPayload.pickup_address = pickupLabel.trim();
    }
  }
  const dropoffAddress = legacyPayload.dropoff_address;
  if (dropoffAddress && typeof dropoffAddress === "object") {
    const dropoffLabel =
      (dropoffAddress as Record<string, unknown>).label ??
      (dropoffAddress as Record<string, unknown>).address ??
      (dropoffAddress as Record<string, unknown>).description;
    if (typeof dropoffLabel === "string" && dropoffLabel.trim().length > 0) {
      legacyPayload.dropoff_address = dropoffLabel.trim();
    }
  }
  if (
    legacyPayload.is_return === true &&
    typeof legacyPayload.return_date === "string" &&
    legacyPayload.return_date.trim().length > 0 &&
    (legacyPayload.return_time == null || String(legacyPayload.return_time).trim().length === 0)
  ) {
    // Date seule : le backend crée le retour avec scheduled_time=null, time_confirmed=false.
    delete legacyPayload.return_time;
  }
  const isRoundTripRequest =
    canonicalPayload.is_return === true || canonicalPayload.is_round_trip === true;
  const cleanedLegacyPayload = stripNullishFields(legacyPayload as Record<string, unknown>);
  const mobileCreateRequest = () =>
    apiClient.post(
      "/company_mobile/dispatch/v1/rides",
      cleanedLegacyPayload,
      withContextHeaders(options)
    );
  const legacyDispatchCreateRequest = () =>
    apiClient.post("/dispatch/v1/rides", cleanedLegacyPayload, withContextHeaders(options));
  const manualCreateRequest = () =>
    apiClient.post(
      "/companies/me/reservations/manual",
      webManualPayload,
      withContextHeaders(options)
    );
  const reservationsCreateRequest = () =>
    apiClient.post(
      "/companies/me/reservations",
      webManualPayload,
      withContextHeaders(options)
    );

  const requests: (() => Promise<{ data: CompanyAnyPayload }>)[] = isRecurringRequest
    ? [
        manualCreateRequest,
        reservationsCreateRequest,
        mobileCreateRequest,
        legacyDispatchCreateRequest,
      ]
    : isRoundTripRequest
      ? [
          manualCreateRequest,
          mobileCreateRequest,
          legacyDispatchCreateRequest,
          reservationsCreateRequest,
        ]
      : [
          mobileCreateRequest,
          legacyDispatchCreateRequest,
          manualCreateRequest,
          reservationsCreateRequest,
        ];
  const requestNames = isRecurringRequest
    ? [
        "/companies/me/reservations/manual",
        "/companies/me/reservations",
        "/company_mobile/dispatch/v1/rides",
        "/dispatch/v1/rides",
      ]
    : isRoundTripRequest
      ? [
          "/companies/me/reservations/manual",
          "/company_mobile/dispatch/v1/rides",
          "/dispatch/v1/rides",
          "/companies/me/reservations",
        ]
      : [
          "/company_mobile/dispatch/v1/rides",
          "/dispatch/v1/rides",
          "/companies/me/reservations/manual",
          "/companies/me/reservations",
        ];
  const tracedRequests = requests.map((request, index) => {
    const endpoint = requestNames[index] ?? `attempt_${index + 1}`;
    return async () => {
      if (__DEV__) {
        console.log("[createCompanyRide] attempt", {
          trace_id: debugTraceId,
          endpoint,
          contextId: options.contextId,
          isRecurringRequest,
          recurrence: {
            is_recurring: canonicalPayload.is_recurring ?? false,
            recurrence_type: canonicalPayload.recurrence_type ?? null,
            recurrence_days: canonicalPayload.recurrence_days ?? null,
            recurrence_end_date: canonicalPayload.recurrence_end_date ?? null,
            occurrences: canonicalPayload.occurrences ?? null,
            recurrence_series_length: canonicalPayload.recurrence_series_length ?? null,
          },
        });
      }
      let result: { data: CompanyAnyPayload };
      try {
        result = await request();
      } catch (error) {
        if (__DEV__) {
          const axiosError = error as AxiosError<{ message?: unknown; error?: unknown }>;
          const status = axiosError?.response?.status ?? null;
          const body = axiosError?.response?.data;
          const message =
            typeof body?.message === "string"
              ? body.message
              : typeof body?.error === "string"
                ? body.error
                : axiosError?.message ?? "unknown_error";
          console.log("[createCompanyRide] attempt failed", {
            trace_id: debugTraceId,
            endpoint,
            status,
            message,
          });
        }
        throw error;
      }
      if (__DEV__) {
        const data = result?.data as Record<string, unknown> | undefined;
        const reservation =
          data?.reservation && typeof data.reservation === "object"
            ? (data.reservation as Record<string, unknown>)
            : null;
        console.log("[createCompanyRide] success", {
          trace_id: debugTraceId,
          endpoint,
          keys: data ? Object.keys(data) : [],
          response_summary: {
            reservation_id: reservation?.id ?? null,
            is_recurring:
              data?.is_recurring ?? reservation?.is_recurring ?? data?.recurring ?? null,
            recurrence_type:
              data?.recurrence_type ?? reservation?.recurrence_type ?? null,
            occurrences:
              data?.occurrences ?? data?.recurrence_series_length ?? reservation?.occurrences ?? null,
            created_count:
              data?.created_count ??
              data?.created_total ??
              data?.series_count ??
              data?.reservations_count ??
              null,
          },
        });
      }
      return result;
    };
  });
  const response = await requestWithFallback<CompanyAnyPayload>(tracedRequests, {
    domain: "dispatch_ride_create",
    contextId: options.contextId,
  });
  return response.data as CompanyAnyPayload;
}

export async function updateCompanyRide(
  options: CompanyRequestOptions & { missionId: number; payload: CompanyAnyPayload }
) {
  const response = await apiClient.put(
    `/company_mobile/dispatch/v1/rides/${options.missionId}`,
    options.payload,
    withContextHeaders(options)
  );
  return response.data as CompanyAnyPayload;
}

export async function getCompanyDispatchSettings(options: CompanyRequestOptions) {
  const response = await apiClient.get("/dispatch/v1/settings", withContextHeaders(options));
  return response.data as CompanyAnyPayload;
}

export async function updateCompanyDispatchSettings(
  options: CompanyRequestOptions & { payload: CompanyAnyPayload }
) {
  const response = await apiClient.put("/dispatch/v1/settings", options.payload, withContextHeaders(options));
  return response.data as CompanyAnyPayload;
}

export async function getCompanyDispatchModes(options: CompanyRequestOptions) {
  const response = await requestWithFallback<CompanyAnyPayload>([
    () => apiClient.get("/company_mobile/dispatch/v1/mode", withContextHeaders(options)),
    () => apiClient.get("/dispatch/v1/modes", withContextHeaders(options)),
    () => apiClient.get("/dispatch/v1/mode", withContextHeaders(options)),
  ], { domain: "dispatch_modes_get", contextId: options.contextId });
  return response.data as CompanyAnyPayload;
}

export async function switchCompanyDispatchMode(
  options: CompanyRequestOptions & { mode: "manual" | "semi_auto" | "fully_auto" }
) {
  const response = await requestWithFallback<CompanyAnyPayload>([
    () =>
      apiClient.put(
        "/company_mobile/dispatch/v1/mode",
        { dispatch_mode: options.mode },
        withContextHeaders(options)
      ),
    () =>
      apiClient.post(
        "/dispatch/v1/modes/switch",
        { mode: options.mode },
        withContextHeaders(options)
      ),
    () =>
      apiClient.put(
        "/dispatch/v1/mode",
        { dispatch_mode: options.mode },
        withContextHeaders(options)
      ),
  ], { domain: "dispatch_modes_switch", contextId: options.contextId });
  return response.data as CompanyAnyPayload;
}

export async function createCompanyIncident(
  options: CompanyRequestOptions & { payload: CompanyAnyPayload }
) {
  const response = await apiClient.post("/dispatch/v1/incidents", options.payload, withContextHeaders(options));
  return response.data as CompanyAnyPayload;
}

export async function getCompanyDispatchMessages(
  options: CompanyRequestOptions & { date?: string; before?: string; limit?: number }
) {
  const response = await requestWithFallback<CompanyAnyPayload>([
    () =>
      apiClient.get("/company_mobile/dispatch/v1/chat/messages", {
        ...withContextHeaders(options),
        params: {
          date: options.date ?? undefined,
          before: options.before ?? undefined,
          limit: options.limit ?? undefined,
        },
      }),
    () =>
      apiClient.get("/dispatch/v1/messages", {
        ...withContextHeaders(options),
        params: {
          date: options.date ?? undefined,
          before: options.before ?? undefined,
          limit: options.limit ?? undefined,
        },
      }),
    () =>
      apiClient.get("/dispatch/v1/chat/messages", {
        ...withContextHeaders(options),
        params: {
          date: options.date ?? undefined,
          before: options.before ?? undefined,
          limit: options.limit ?? undefined,
        },
      }),
  ], { domain: "dispatch_messages_get", contextId: options.contextId });
  return response.data as CompanyAnyPayload;
}

export async function sendCompanyDispatchMessage(
  options: CompanyRequestOptions & { content: string; missionId?: number }
) {
  const response = await requestWithFallback<CompanyAnyPayload>([
    () =>
      apiClient.post(
        "/company_mobile/dispatch/v1/chat/messages",
        {
          content: options.content,
          mission_id: options.missionId ?? null,
        },
        withContextHeaders(options)
      ),
    () =>
      apiClient.post(
        "/dispatch/v1/messages",
        {
          content: options.content,
          mission_id: options.missionId ?? null,
        },
        withContextHeaders(options)
      ),
    () =>
      apiClient.post(
        "/dispatch/v1/chat/messages",
        {
          content: options.content,
          mission_id: options.missionId ?? null,
        },
        withContextHeaders(options)
      ),
  ], { domain: "dispatch_messages_send", contextId: options.contextId });
  return response.data as CompanyAnyPayload;
}

export async function applyCompanyOpportunity(
  options: CompanyRequestOptions & { opportunityId: number; missionId: number; driverId?: number }
) {
  const fallbackRequests: (() => Promise<{ data: CompanyAnyPayload }>)[] = [];
  if (typeof options.driverId === "number" && Number.isFinite(options.driverId)) {
    fallbackRequests.push(() =>
      apiClient.post(
        `/dispatch/v1/rides/${options.missionId}/reassign`,
        { driver_id: options.driverId, reason: "opportunity_fallback" },
        withContextHeaders(options)
      )
    );
  }
  const response = await requestWithFallback<CompanyAnyPayload>([
    () =>
      apiClient.post(
        "/dispatch/v1/opportunities/apply",
        { opportunity_id: options.opportunityId, mission_id: options.missionId },
        withContextHeaders(options)
      ),
    ...fallbackRequests,
  ], { domain: "dispatch_opportunity_apply", contextId: options.contextId });
  return response.data as CompanyAnyPayload;
}

export async function resetCompanyAssignments(options: CompanyRequestOptions & { date?: string }) {
  const response = await requestWithFallback<CompanyAnyPayload>([
    () =>
      apiClient.post(
        "/dispatch/v1/reset-assignments",
        { date: options.date ?? undefined },
        withContextHeaders(options)
      ),
    () =>
      apiClient.post(
        "/dispatch/v1/reset",
        { date: options.date ?? undefined },
        withContextHeaders(options)
      ),
  ], { domain: "dispatch_reset", contextId: options.contextId });
  return response.data as CompanyAnyPayload;
}

export async function searchCompanyAddresses(options: CompanyRequestOptions & { q: string }) {
  const toAddressRows = (payload: CompanyAnyPayload): Record<string, unknown>[] => {
    if (Array.isArray(payload)) {
      return payload.filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === "object");
    }
    if (!payload || typeof payload !== "object") return [];
    const raw = payload as Record<string, unknown>;
    const candidate = [
      raw.items,
      raw.results,
      raw.data,
      raw.clients,
      raw.addresses,
      raw.features,
      raw.predictions,
      raw.suggestions,
    ].find((entry) => Array.isArray(entry));
    if (!Array.isArray(candidate)) return [];
    return candidate.filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === "object");
  };
  const hasGoogleSource = (rows: Record<string, unknown>[]): boolean =>
    rows.some((row) => {
      const rowSource = typeof row.source === "string" ? row.source : null;
      const properties = row.properties;
      const propertySource =
        properties && typeof properties === "object" && typeof (properties as Record<string, unknown>).source === "string"
          ? ((properties as Record<string, unknown>).source as string)
          : null;
      return rowSource === "google_places" || rowSource === "google" || propertySource === "google_places" || propertySource === "google";
    });
  const statusFromError = (error: unknown): number | null =>
    typeof (error as AxiosError)?.response?.status === "number" ? ((error as AxiosError).response?.status as number) : null;

  const requests: (() => Promise<{ data: CompanyAnyPayload }>)[] = [
    () =>
      apiClient.get("/geocode/autocomplete", {
        ...withContextHeaders(options),
        params: { q: options.q, lat: 46.2044, lon: 6.1432, limit: 8 },
      }),
    () =>
      apiClient.get("/company_mobile/dispatch/v1/addresses/search", {
        ...withContextHeaders(options),
        params: { q: options.q },
      }),
  ];

  try {
    let firstNonEmptyPayload: CompanyAnyPayload | null = null;
    let lastError: unknown = null;
    for (let index = 0; index < requests.length; index += 1) {
      try {
        if (index > 0) {
          emitCompanyDispatchTelemetry(
            "company.dispatch.contract_fallback",
            {
              source: "companyApi.searchCompanyAddresses",
              domain: "dispatch_address_search",
              context_id: options.contextId,
              attempt: index + 1,
            },
            { allowWhenDisabled: true }
          );
        }
        const response = await requests[index]();
        const payload = response.data as CompanyAnyPayload;
        const rows = toAddressRows(payload);
        if (rows.length === 0) {
          continue;
        }
        if (hasGoogleSource(rows)) {
          return payload;
        }
        if (firstNonEmptyPayload == null) {
          firstNonEmptyPayload = payload;
        }
      } catch (error) {
        lastError = error;
        const status = statusFromError(error);
        if (status === 401 || status === 403) {
          emitCompanyDispatchTelemetry(
            "company.dispatch.auth_failure",
            {
              source: "companyApi.searchCompanyAddresses",
              domain: "dispatch_address_search",
              context_id: options.contextId,
              status,
            },
            { allowWhenDisabled: true }
          );
          throw error;
        }
        if (index === requests.length - 1) {
          throw error;
        }
      }
    }
    if (firstNonEmptyPayload != null) {
      return firstNonEmptyPayload;
    }
    if (lastError != null) {
      throw lastError;
    }
    return [];
  } catch (error) {
    const status = (error as AxiosError | undefined)?.response?.status;
    if (status === 404) {
      return [];
    }
    throw error;
  }
}

export async function searchCompanyClients(options: CompanyRequestOptions & { q?: string }) {
  const response = await requestWithFallback<CompanyAnyPayload>([
    () =>
      apiClient.get("/companies/me/clients", {
        ...withContextHeaders(options),
        params: {
          q: options.q ?? undefined,
          search: options.q ?? undefined,
        },
      }),
    () =>
      apiClient.get("/company_mobile/dispatch/v1/clients/search", {
        ...withContextHeaders(options),
        params: { q: options.q ?? undefined },
      }),
  ], { domain: "dispatch_client_search", contextId: options.contextId });
  return response.data as CompanyAnyPayload;
}

export async function createCompanyClient(
  options: CompanyRequestOptions & { payload: CompanyAnyPayload }
) {
  const response = await requestWithFallback<CompanyAnyPayload>([
    () =>
      apiClient.post(
        "/dispatch/v1/clients/create",
        options.payload,
        withContextHeaders(options)
      ),
    () =>
      apiClient.post(
        "/companies/me/clients",
        options.payload,
        withContextHeaders(options)
      ),
  ], { domain: "dispatch_client_create", contextId: options.contextId });
  return response.data as CompanyAnyPayload;
}

export async function getCompanyRideDetail(
  options: CompanyRequestOptions & { missionId: number; date?: string }
) {
  const response = await requestWithFallback<CompanyAnyPayload>([
    () =>
      apiClient.get(`/company_mobile/dispatch/v1/rides/${options.missionId}`, withContextHeaders(options)),
    () =>
      apiClient.get(
        `/dispatch/v1/rides/${options.missionId}`,
        withContextHeaders(options)
      ),
    () =>
      apiClient.get("/dispatch/v1/rides", {
        ...withContextHeaders(options),
        params: {
          date: options.date ?? new Date().toISOString().slice(0, 10),
          page_size: 120,
        },
      }),
  ], { domain: "dispatch_ride_detail", contextId: options.contextId });
  return response.data as CompanyAnyPayload;
}

export async function getCompanyAvailableDrivers(options: CompanyRequestOptions) {
  const response = await requestWithFallback<CompanyAnyPayload>(
    [
      () => apiClient.get("/company_mobile/dispatch/v1/drivers/available", withContextHeaders(options)),
      () => apiClient.get("/dispatch/v1/drivers/available", withContextHeaders(options)),
    ],
    { domain: "dispatch_drivers_available", contextId: options.contextId }
  );
  return response.data as CompanyAnyPayload;
}

export async function getCompanyPartnershipsForTransfer(options: CompanyRequestOptions) {
  const response = await requestWithFallback<CompanyAnyPayload>(
    [
      () =>
        apiClient.get("/company_mobile/partnerships/for-transfer", withContextHeaders(options)),
      () => apiClient.get("/partnerships/for-transfer", withContextHeaders(options)),
    ],
    { domain: "partnerships_for_transfer", contextId: options.contextId }
  );
  return { items: normalizePartnershipsForTransfer(response.data) };
}

export type CompanyTransferPartnershipOption = {
  /** Identifiant du partenariat (requis pour POST …/partnerships/:id/transfers). */
  id: number;
  label: string;
  partnerCompanyId: number;
};

export function normalizePartnershipsForTransfer(payload: unknown): CompanyTransferPartnershipOption[] {
  if (!payload || typeof payload !== "object") return [];
  const root = payload as Record<string, unknown>;
  const rows = [root.data, root.items, root.partnerships, payload].find((value) => Array.isArray(value));
  if (!Array.isArray(rows)) return [];

  return rows
    .map((entry) => {
      if (!entry || typeof entry !== "object") return null;
      const raw = entry as Record<string, unknown>;
      const partnershipId = toFiniteNumber(raw.id);
      if (partnershipId == null) return null;
      const partnerCompanyId =
        toFiniteNumber(raw.partner_company_id ?? raw.company_id ?? raw.target_company_id) ??
        partnershipId;
      const labelRaw =
        (typeof raw.partner_company_name === "string" && raw.partner_company_name.trim()) ||
        (typeof raw.company_name === "string" && raw.company_name.trim()) ||
        (typeof raw.name === "string" && raw.name.trim()) ||
        `Entreprise #${partnerCompanyId}`;
      return {
        id: partnershipId,
        label: labelRaw,
        partnerCompanyId,
      };
    })
    .filter((item): item is CompanyTransferPartnershipOption => item != null);
}

export async function transferCompanyRide(
  options: CompanyRequestOptions & { missionId: number; partnershipId: number }
) {
  try {
    const response = await requestWithFallback<CompanyAnyPayload>(
      [
        () =>
          apiClient.post(
            `/company_mobile/partnerships/${options.partnershipId}/transfers`,
            { booking_id: options.missionId },
            withContextHeaders(options)
          ),
        () =>
          apiClient.post(
            `/partnerships/${options.partnershipId}/transfers`,
            { booking_id: options.missionId },
            withContextHeaders(options)
          ),
      ],
      { domain: "partnership_transfer_post", contextId: options.contextId }
    );
    return response.data as CompanyAnyPayload;
  } catch (error) {
    const axiosError = error as AxiosError | undefined;
    if (axiosError?.response?.status === 409) {
      emitCompanyDispatchTelemetry(
        "company.dispatch.transfer_conflict",
        {
          source: "companyApi.transferCompanyRide",
          context_id: options.contextId,
          mission_id: options.missionId,
          partnership_id: options.partnershipId,
        },
        { allowWhenDisabled: true }
      );
      throw new Error(
        "Conflit de transfert detecte (409). Rafraichissez la mission puis recommencez."
      );
    }
    throw new Error(getDispatchApiErrorMessage(error, "Transfert impossible."));
  }
}

export async function getCompanyBillingSettings(options: CompanyRequestOptions) {
  const response = await apiClient.get("/company-settings/billing", withContextHeaders(options));
  return response.data as CompanyAnyPayload;
}

function unwrapCompanyMePayload(data: unknown): CompanyAnyPayload {
  if (data && typeof data === "object" && "data" in data && !("error" in data)) {
    const inner = (data as Record<string, unknown>).data;
    if (inner && typeof inner === "object") return inner as CompanyAnyPayload;
  }
  if (data && typeof data === "object") return data as CompanyAnyPayload;
  return {};
}

/** Profil entreprise (`GET /companies/me`) — aligné sur le portail web. */
export async function getCompanyProfile(options: CompanyRequestOptions): Promise<CompanyAnyPayload> {
  const response = await apiClient.get("/companies/me", withContextHeaders(options));
  return unwrapCompanyMePayload(response.data);
}

export async function simulateCompanyPricing(
  options: CompanyRequestOptions & { payload: CompanyAnyPayload }
) {
  const response = await apiClient.post("/pricing/simulate", options.payload, {
    ...withContextHeaders(options),
    timeout: 6000,
  });
  return response.data as CompanyAnyPayload;
}

export async function updateCompanyBillingSettings(
  options: CompanyRequestOptions & { payload: CompanyAnyPayload }
) {
  const response = await apiClient.put(
    "/company-settings/billing",
    options.payload,
    withContextHeaders(options)
  );
  return response.data as CompanyAnyPayload;
}

export async function getCompanyClients(
  options: CompanyRequestOptions & { q?: string; page?: number; limit?: number }
) {
  const page = options.page ?? 1;
  const perPage = options.limit ?? 100;
  const response = await requestWithFallback<CompanyAnyPayload>([
    () =>
      apiClient.get("/companies/me/clients", {
        ...withContextHeaders(options),
        params: {
          search: options.q?.trim() || undefined,
          page,
          per_page: perPage,
        },
      }),
    () =>
      apiClient.get("/dispatch/v1/clients", {
        ...withContextHeaders(options),
        params: {
          q: options.q ?? undefined,
          page,
        },
      }),
  ], { domain: "company_clients_readonly", contextId: options.contextId });
  return response.data as CompanyAnyPayload;
}

export async function getCompanyClientDetail(
  options: CompanyRequestOptions & { clientId: number }
) {
  try {
    const response = await requestWithFallback<CompanyAnyPayload>([
      () => apiClient.get(`/companies/me/clients/${options.clientId}`, withContextHeaders(options)),
      () => apiClient.get(`/dispatch/v1/clients/${options.clientId}`, withContextHeaders(options)),
    ], { domain: "company_client_detail_readonly", contextId: options.contextId });
    return response.data as CompanyAnyPayload;
  } catch (error) {
    const status = (error as AxiosError | undefined)?.response?.status;
    if (status === 404) {
      return null;
    }
    throw error;
  }
}

export async function getCompanyInvoices(
  options: CompanyRequestOptions & { q?: string; page?: number; limit?: number }
) {
  const response = await requestWithFallback<CompanyAnyPayload>([
    () =>
      apiClient.get("/companies/me/invoices", {
        ...withContextHeaders(options),
        params: {
          q: options.q ?? undefined,
          page: options.page ?? undefined,
          limit: options.limit ?? undefined,
        },
      }),
    () =>
      apiClient.get(`/invoices/companies/${options.contextId.replace("company:", "")}/invoices`, {
        ...withContextHeaders(options),
        params: {
          q: options.q ?? undefined,
          page: options.page ?? undefined,
          limit: options.limit ?? undefined,
        },
      }),
  ], { domain: "company_invoices_readonly", contextId: options.contextId });
  return response.data as CompanyAnyPayload;
}
