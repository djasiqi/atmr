export type CompanyDispatchMissionStatus =
  | "pending"
  | "proposed"
  | "accepted"
  | "assigned"
  | "en_route"
  | "arrived"
  | "in_progress"
  | "completed"
  | "cancelled";

export type CompanyDispatchMission = {
  mission_id: number;
  status: CompanyDispatchMissionStatus;
  scheduled_at?: string | null;
  /** Nom d’affichage passager (API `identity.passenger.name` ou repli `client.name`). */
  client_name?: string | null;
  /** Blocs canoniques PR1 (optionnels — repli client côté UI). */
  identity?: {
    passenger?: { name?: string | null };
    source?: {
      type?: string | null;
      id?: number | string | null;
      code?: string | null;
      name?: string | null;
    } | null;
    requester?: { id?: number | string | null; name?: string | null } | null;
    ownership?: {
      owner_company_id?: number | null;
      owner_company_name?: string | null;
    } | null;
    execution?: {
      executing_company_id?: number | null;
      executing_company_name?: string | null;
    } | null;
    upstream?: {
      type?: string | null;
      id?: number | string | null;
      code?: string | null;
      name?: string | null;
    } | null;
    origin_channel?: string | null;
  } | null;
  trip_flags?: Record<string, boolean | number | null> | null;
  scheduling?: {
    scheduled_time?: string | null;
    time_defined?: boolean;
    /** INV-2b — existence d'une heure métier (priorité helpers client). */
    time_scheduled?: boolean;
    time_confirmed?: boolean;
    display_time?: string | null;
  } | null;
  search_index?: string[] | null;
  pickup_label?: string | null;
  dropoff_label?: string | null;
  pickup_lat?: number | null;
  pickup_lon?: number | null;
  dropoff_lat?: number | null;
  dropoff_lon?: number | null;
  /** Distance planifiée (km) — résumé dispatch mobile. */
  route_distance_km?: number | null;
  /** Durée planifiée (minutes) — résumé dispatch mobile. */
  route_duration_min?: number | null;
  /** Nom d’affichage chauffeur (API) pour pastilles type operations-app. */
  driver_name?: string | null;
  driver_id?: number | null;
  /** Partenaire exécutant (course transférée) — affichage carte / fiche. */
  partner_company_name?: string | null;
  /** Type chauffeur (ex. REGULAR, EMERGENCY) — résumé dispatch. */
  driver_type?: string | null;
  company_id?: number | null;
  updated_at?: string | null;
  /**
   * Repli retard pickup (≥ 1 min) depuis `assignment.delay_seconds` quand `/delays*` n’a pas encore de ligne ETA.
   */
  assignment_pickup_delay_minutes?: number | null;
};

export type CompanyDispatchMissionListResponse = {
  context_id: string;
  missions: CompanyDispatchMission[];
  refreshed_at: string;
};

export type CompanyDriverLiveLocation = {
  driver_id: number;
  driver_name?: string | null;
  full_name?: string | null;
  first_name?: string | null;
  last_name?: string | null;
  mission_id?: number | null;
  latitude: number;
  longitude: number;
  accuracy?: number | null;
  heading?: number | null;
  speed?: number | null;
  is_background?: boolean;
  timestamp: string;
  recorded_at?: string | null;
  received_at?: string | null;
  last_seen_seconds?: number | null;
  location_status?: "live" | "recent" | "stale" | "offline" | "last_known" | null;
  tracking_display_status?: string | null;
  presence_status?: string | null;
  status?: string | null;
  device_health?: {
    constraint_reason?: string | null;
    battery_optimized?: boolean | null;
    tracking_active?: boolean | null;
  } | null;
  // Marqueur serveur : position acceptée en mode observabilité uniquement (moins prioritaire qu'une
  // position live). Ne doit jamais écraser une position live plus récente sur la carte entreprise.
  accepted_observability_only?: boolean;
};

export type CompanyDriverLiveLocationResponse = {
  context_id: string;
  locations: CompanyDriverLiveLocation[];
  refreshed_at: string;
};

export type CompanyDelayInvalidationEvent = {
  event_id: string;
  mission_id: number;
  invalidated_reason: "delay_update" | "route_change" | "manual_override";
  created_at: string;
  actor_id?: string | null;
};

export type CompanyOptimizerStatus = {
  optimizer_enabled: boolean;
  optimizer_state: "idle" | "running" | "degraded" | "failed";
  last_run_at?: string | null;
  last_success_at?: string | null;
  reason?: string | null;
};

export type CompanyOptimizerStatusResponse = {
  context_id: string;
  status: CompanyOptimizerStatus;
  refreshed_at: string;
};

export type CompanyDispatchRealtimeDashboard = {
  context_id: string;
  refreshed_at: string;
  /** `false` si l’API n’expose pas encore le compteur (éviter un 0 implicite côté UI) */
  delayed_bookings_metrics_available: boolean;
  delayed_bookings: number;
  /** `false` si le tableau `opportunities` n’est pas fourni (distinct d’un tableau vide) */
  opportunities_metrics_available: boolean;
  opportunities: number;
  avg_delay_minutes: number;
};

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isMissionStatus(value: unknown): value is CompanyDispatchMissionStatus {
  return (
    value === "pending" ||
    value === "proposed" ||
    value === "accepted" ||
    value === "assigned" ||
    value === "en_route" ||
    value === "arrived" ||
    value === "in_progress" ||
    value === "completed" ||
    value === "cancelled"
  );
}

function isOptimizerState(value: unknown): value is CompanyOptimizerStatus["optimizer_state"] {
  return value === "idle" || value === "running" || value === "degraded" || value === "failed";
}

export function isCompanyDispatchMission(value: unknown): value is CompanyDispatchMission {
  if (!value || typeof value !== "object") return false;
  const candidate = value as CompanyDispatchMission;
  return isFiniteNumber(candidate.mission_id) && isMissionStatus(candidate.status);
}

export function validateCompanyMissionListResponse(
  value: unknown
): value is CompanyDispatchMissionListResponse {
  if (!value || typeof value !== "object") return false;
  const candidate = value as CompanyDispatchMissionListResponse;
  return (
    isString(candidate.context_id) &&
    Array.isArray(candidate.missions) &&
    candidate.missions.every(isCompanyDispatchMission) &&
    isString(candidate.refreshed_at)
  );
}

export function validateCompanyDriverLocationsResponse(
  value: unknown
): value is CompanyDriverLiveLocationResponse {
  if (!value || typeof value !== "object") return false;
  const candidate = value as CompanyDriverLiveLocationResponse;
  return (
    isString(candidate.context_id) &&
    Array.isArray(candidate.locations) &&
    candidate.locations.every(
      (location) =>
        isFiniteNumber(location.driver_id) &&
        isFiniteNumber(location.latitude) &&
        isFiniteNumber(location.longitude) &&
        isString(location.timestamp)
    ) &&
    isString(candidate.refreshed_at)
  );
}

export function validateCompanyOptimizerStatusResponse(
  value: unknown
): value is CompanyOptimizerStatusResponse {
  if (!value || typeof value !== "object") return false;
  const candidate = value as CompanyOptimizerStatusResponse;
  return (
    isString(candidate.context_id) &&
    !!candidate.status &&
    typeof candidate.status.optimizer_enabled === "boolean" &&
    isOptimizerState(candidate.status.optimizer_state) &&
    isString(candidate.refreshed_at)
  );
}

export function validateCompanyDispatchRealtimeDashboard(
  value: unknown
): value is CompanyDispatchRealtimeDashboard {
  if (!value || typeof value !== "object") return false;
  const candidate = value as CompanyDispatchRealtimeDashboard;
  return (
    isString(candidate.context_id) &&
    isString(candidate.refreshed_at) &&
    typeof candidate.delayed_bookings_metrics_available === "boolean" &&
    isFiniteNumber(candidate.delayed_bookings) &&
    typeof candidate.opportunities_metrics_available === "boolean" &&
    isFiniteNumber(candidate.opportunities) &&
    isFiniteNumber(candidate.avg_delay_minutes)
  );
}
