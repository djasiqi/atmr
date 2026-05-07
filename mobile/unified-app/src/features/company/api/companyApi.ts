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

function shouldTryFallback(error: unknown): boolean {
  const axiosError = error as AxiosError | undefined;
  const status = axiosError?.response?.status;
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
      if (index === requests.length - 1 || !shouldTryFallback(error)) {
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
async function postCompanyDispatchRideAction(
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
  time?: { pickup_at?: string | null };
  client?: { name?: string | null } | null;
  route?: { pickup_address?: string | null; dropoff_address?: string | null };
  driver?: { id?: string | number | null; name?: string | null; display_name?: string | null } | null;
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
  drivers?: {
    driver_id?: number | string;
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

function toIso(input: unknown): string {
  if (typeof input === "string" && input.length > 0) return input;
  return new Date().toISOString();
}

function toFiniteNumber(input: unknown): number | null {
  if (typeof input === "number" && Number.isFinite(input)) return input;
  if (typeof input === "string") {
    const parsed = Number.parseFloat(input);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function normalizeMission(raw: RawRide): CompanyDispatchMission | null {
  const missionId = toFiniteNumber(raw.mission_id ?? raw.booking_id ?? raw.id);
  if (missionId == null) return null;
  let status = String(raw.status ?? "pending").toLowerCase();
  // Backend `Booking` : aller-retour terminé, pas une course « à venir » (avant: inconnu -> pending).
  if (status === "return_completed") {
    status = "completed";
  } else if (status === "canceled") {
    status = "cancelled";
  } else if (status === "awaiting_client_payment" || status === "awaiting_payment") {
    status = "pending";
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
    raw.client && typeof raw.client === "object" && typeof raw.client.name === "string" && raw.client.name.trim()
      ? raw.client.name.trim()
      : null;
  const driverNameRaw = raw.driver && typeof raw.driver === "object" ? raw.driver : null;
  const driverNameFromApi =
    typeof driverNameRaw?.display_name === "string" && driverNameRaw.display_name.trim()
      ? driverNameRaw.display_name.trim()
      : typeof driverNameRaw?.name === "string" && driverNameRaw.name.trim()
        ? driverNameRaw.name.trim()
        : null;
  return {
    mission_id: missionId,
    status: normalizedStatus,
    scheduled_at: raw.time?.pickup_at ?? null,
    client_name: clientName,
    pickup_label: raw.route?.pickup_address ?? null,
    dropoff_label: raw.route?.dropoff_address ?? null,
    driver_name: driverNameFromApi,
    driver_id: driverId,
    company_id: companyId,
    updated_at: raw.updated_at ?? null,
  };
}

function normalizeLocation(
  raw: NonNullable<RawDriversResponse["locations"]>[number]
): CompanyDriverLiveLocation | null {
  const driverId = toFiniteNumber(raw.driver_id);
  const latitude = toFiniteNumber(raw.latitude ?? raw.lat);
  const longitude = toFiniteNumber(raw.longitude ?? raw.lon ?? raw.lng);
  if (driverId == null || latitude == null || longitude == null) return null;
  return {
    driver_id: driverId,
    mission_id: raw.mission_id ?? null,
    latitude,
    longitude,
    timestamp: toIso(raw.timestamp),
    recorded_at: toIso(raw.recorded_at ?? raw.timestamp),
    received_at: toIso(raw.received_at),
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
          page_size: 120,
        },
      }),
    () =>
      apiClient.get<RawDispatchRidesResponse>(COMPANY_BACKEND_ENDPOINTS.missions, {
        headers: buildHeaders(options.contextId),
        params: {
          date: options.date,
          q: options.search || undefined,
          status: options.status || undefined,
          page_size: 120,
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
    .map(normalizeLocation)
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
      COMPANY_BACKEND_ENDPOINTS.driversLocations
    );
  } catch {
    return fetchDriverLocationsEndpoint(options.contextId, COMPANY_BACKEND_ENDPOINTS.driversLive);
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
    await postCompanyDispatchRideAction(
      options,
      "/assign",
      { driver_id: options.driverId },
      "dispatch_assign_post"
    );
  } catch (e) {
    throw new Error(getDispatchApiErrorMessage(e, "Assignation impossible."));
  }
}

export async function reassignCompanyRide(
  options: CompanyRequestOptions & { missionId: number; driverId: number }
) {
  try {
    await postCompanyDispatchRideAction(
      options,
      "/reassign",
      { driver_id: options.driverId },
      "dispatch_reassign_post"
    );
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
  await postCompanyDispatchRideAction(
    options,
    "/cancel",
    {
      reason_code: reasonCode,
      note,
      reason: options.reason ?? reasonCode,
    } satisfies CompanyCancelRidePayload,
    "dispatch_cancel_post"
  );
}

export async function scheduleCompanyRide(
  options: CompanyRequestOptions & { missionId: number; payload: CompanyScheduleRidePayload }
) {
  await postCompanyDispatchRideAction(options, "/schedule", options.payload, "dispatch_schedule_post");
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
  const response = await apiClient.post("/dispatch/v1/rides", options.payload, withContextHeaders(options));
  return response.data as CompanyAnyPayload;
}

export async function updateCompanyRide(
  options: CompanyRequestOptions & { missionId: number; payload: CompanyAnyPayload }
) {
  const response = await apiClient.put(
    `/dispatch/v1/rides/${options.missionId}`,
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
  try {
    const response = await requestWithFallback<CompanyAnyPayload>(
      [
        () =>
          apiClient.get("/company_mobile/dispatch/v1/addresses/search", {
            ...withContextHeaders(options),
            params: { q: options.q },
          }),
        () =>
          apiClient.get("/geocode/autocomplete", {
            ...withContextHeaders(options),
            params: { q: options.q, limit: 8 },
          }),
        () =>
          apiClient.get("/dispatch/v1/places/autocomplete", {
            ...withContextHeaders(options),
            params: { q: options.q },
          }),
        () =>
          apiClient.get("/dispatch/v1/addresses/search", {
            ...withContextHeaders(options),
            params: { q: options.q },
          }),
      ],
      { domain: "dispatch_address_search", contextId: options.contextId }
    );
    return response.data as CompanyAnyPayload;
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
  const response = await apiClient.get(
    "/dispatch/v1/partnerships/for-transfer",
    withContextHeaders(options)
  );
  return response.data as CompanyAnyPayload;
}

export async function transferCompanyRide(
  options: CompanyRequestOptions & { missionId: number; targetCompanyId: number }
) {
  try {
    const response = await apiClient.post(
      "/dispatch/v1/partnerships/transfers",
      { mission_id: options.missionId, target_company_id: options.targetCompanyId },
      withContextHeaders(options)
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
          target_company_id: options.targetCompanyId,
        },
        { allowWhenDisabled: true }
      );
      throw new Error(
        "Conflit de transfert detecte (409). Rafraichissez la mission puis recommencez."
      );
    }
    throw error;
  }
}

export async function getCompanyBillingSettings(options: CompanyRequestOptions) {
  const response = await apiClient.get("/company-settings/billing", withContextHeaders(options));
  return response.data as CompanyAnyPayload;
}

export async function simulateCompanyPricing(
  options: CompanyRequestOptions & { payload: CompanyAnyPayload }
) {
  const response = await apiClient.post("/pricing/simulate", options.payload, withContextHeaders(options));
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
  const response = await requestWithFallback<CompanyAnyPayload>([
    () =>
      apiClient.get("/companies/me/clients", {
        ...withContextHeaders(options),
        params: {
          q: options.q ?? undefined,
          page: options.page ?? undefined,
          limit: options.limit ?? undefined,
        },
      }),
    () =>
      apiClient.get("/dispatch/v1/clients", {
        ...withContextHeaders(options),
        params: {
          q: options.q ?? undefined,
        },
      }),
  ], { domain: "company_clients_readonly", contextId: options.contextId });
  return response.data as CompanyAnyPayload;
}

export async function getCompanyClientDetail(
  options: CompanyRequestOptions & { clientId: number }
) {
  const response = await requestWithFallback<CompanyAnyPayload>([
    () => apiClient.get(`/companies/me/clients/${options.clientId}`, withContextHeaders(options)),
    () => apiClient.get(`/dispatch/v1/clients/${options.clientId}`, withContextHeaders(options)),
  ], { domain: "company_client_detail_readonly", contextId: options.contextId });
  return response.data as CompanyAnyPayload;
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
