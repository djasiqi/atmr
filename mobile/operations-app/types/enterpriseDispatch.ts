export interface DispatchStatus {
  osrm: {
    status: "OK" | "WARNING" | "DOWN";
    latency_ms: number | null;
    last_check: string | null;
  };
  agent: {
    mode: "MANUAL" | "SEMI_AUTO" | "FULLY_AUTO";
    active: boolean;
    last_tick: string | null;
  };
  optimizer: {
    active: boolean;
    next_window_start: string | null;
    running?: boolean;
  };
  kpis: {
    date: string;
    total_bookings: number;
    assigned_bookings: number;
    assignment_rate: number;
    at_risk: number;
  };
  dispatch_run?: {
    id: number;
    status: string;
    assignments_count: number;
    day: string | null;
    created_at: string | null;
    started_at: string | null;
    completed_at: string | null;
  } | null;
  celery_state?: string;
  is_running?: boolean;
}

export interface RideSummary {
  id: string;
  time: {
    pickup_at?: string | null;
    drop_eta?: string | null;
    window_start?: string | null;
    window_end?: string | null;
  };
  client: {
    id: string;
    name: string;
    priority: "LOW" | "NORMAL" | "HIGH";
    first_name?: string;
    last_name?: string;
    birth_date?: string;
    phone?: string;
    home_address?: string;
  };
  route: {
    pickup_address: string;
    dropoff_address: string;
    distance_km?: number | null;
  };
  status: "assigned" | "unassigned" | "completed" | "return_completed" | "in_progress" | "en_route" | "cancelled";
  driver: {
    id?: string | null;
    name?: string | null;
    is_emergency: boolean;
  } | null;
  flags: {
    risk_delay: boolean;
    prefs_respected: boolean;
    fairness_score?: number | null;
    override_pending?: boolean;
  };
}

export interface PaginatedRides {
  page: number;
  page_size: number;
  total: number;
  items: RideSummary[];
}

export interface DriverSuggestion {
  driver_id: string;
  driver_name: string;
  score: number;
  fairness_delta: number | null;
  preferred_match: boolean;
  is_emergency: boolean;
  reason: string;
}

export interface RideEvent {
  ts: string;
  event: string;
  actor: string;
  details?: Record<string, unknown>;
  details_formatted?: string;
}

export interface RideConflict {
  type: string;
  message: string;
  blocking: boolean;
}

export interface RideDetail {
  summary: RideSummary;
  suggestions: DriverSuggestion[];
  history: RideEvent[];
  conflicts: RideConflict[];
  notes?: string[];
}

export interface DispatchSettings {
  fairness: {
    max_gap: number;
  };
  emergency: {
    emergency_penalty: number;
  };
  service_times: {
    pickup_service_min: number;
    dropoff_service_min: number;
    min_transition_margin_min: number;
  };
}

export interface DispatchSettingsUpdate {
  fairness?: {
    max_gap?: number;
  };
  emergency?: {
    emergency_penalty?: number;
  };
  service_times?: Partial<DispatchSettings["service_times"]>;
}

export interface AssignRequestPayload {
  driver_id: string;
  reason?: string | null;
  respect_preferences?: boolean;
  allow_emergency?: boolean;
  idempotency_key?: string;
}

export interface AssignResponsePayload {
  ride_id: string;
  driver_id: string;
  scheduled_time: string;
  fairness_delta: number;
  audit_event_id: string;
  message?: string;
}

export interface DispatchRunResponse {
  message: string;
  for_date: string;
  job?: Record<string, any>;
}

export interface ModeResponse {
  mode_before: "manual" | "semi_auto" | "fully_auto";
  mode_after: "manual" | "semi_auto" | "fully_auto";
  effective_at: string;
  requires_approval: boolean;
  audit_event_id: string;
}

export interface IncidentPayload {
  type: string;
  severity: "low" | "medium" | "high" | "critical";
  ride_id?: string | null;
  driver_id?: string | null;
  note?: string | null;
  attachments?: string[];
}

export interface ScheduleRidePayload {
  pickup_at?: string;
  delta_minutes?: number;
}

export interface MarkUrgentPayload {
  extra_delay_minutes?: number;
  reason?: string;
}

export interface DispatchMessage {
  id: number | string;
  sender_id: number | string | null;
  sender_role?: string;
  sender_name?: string | null;
  content: string;
  created_at: string;
}

// ✅ Types pour l'édition et la création de courses
export interface RideEditPayload {
  pickup_address?: string;
  dropoff_address?: string;
  pickup_lat?: number;
  pickup_lon?: number;
  dropoff_lat?: number;
  dropoff_lon?: number;
  scheduled_time?: string;
  // ✅ P1-4 Phase 3.3: Utiliser client_name au lieu de customer_name
  client_name?: string;
  notes?: string;
  priority?: "LOW" | "NORMAL" | "HIGH";
  amount?: number;
  is_return?: boolean;
}

export interface RideCreatePayload {
  client_id?: string;
  // ✅ P1-4 Phase 3.3: Utiliser client_name au lieu de customer_name
  client_name?: string;
  pickup_address: string;
  dropoff_address: string;
  pickup_lat?: number;
  pickup_lon?: number;
  dropoff_lat?: number;
  dropoff_lon?: number;
  scheduled_time?: string;
  notes?: string;
  priority?: "LOW" | "NORMAL" | "HIGH";
  amount?: number;
  is_return?: boolean;
  return_time?: string;
  assign_driver_id?: string;
  wheelchair_client_has?: boolean;
  wheelchair_need?: boolean;
}

export interface AddressSuggestion {
  label?: string;
  address: string;
  lat?: number;
  lon?: number;
  place_id?: string;
}

export interface ClientOption {
  id: string;
  name: string;
  phone?: string;
  email?: string;
  contact_phone?: string;
  contact_email?: string;
  domicile_address?: string | null;
  domicile_zip?: string | null; // ✅ Code postal
  domicile_city?: string | null; // ✅ Ville
  domicile_lat?: number | null;
  domicile_lon?: number | null;
}
