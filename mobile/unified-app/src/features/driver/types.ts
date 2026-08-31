export type DriverMissionStatus =
  | "ASSIGNED"
  | "EN_ROUTE"
  | "ARRIVED"
  | "IN_PROGRESS"
  | "COMPLETED"
  | "CANCELLED"
  | "REASSIGNED"
  | "NO_SHOW"
  | "EXPIRED"
  | "FAILED";

export type DriverTransitionStatus =
  | "EN_ROUTE"
  | "ARRIVED"
  | "IN_PROGRESS"
  | "COMPLETED"
  | "CANCELLED"
  | "FAILED";

export type DriverSocketEventType =
  | "mission_assigned"
  | "mission_updated"
  | "mission_reassigned"
  | "mission_cancelled"
  | "mission_status_changed"
  | "driver_location_required"
  | "eta_changed"
  | "driver_location_batch_ack";

export type DriverMission = {
  id: number;
  status: string;
  pickup_location?: string | null;
  dropoff_location?: string | null;
  scheduled_time?: string | null;
  time_confirmed?: boolean | null;
  scheduling?: {
    time_defined?: boolean;
    time_scheduled?: boolean;
    display_time?: string;
    display_datetime?: string;
  } | null;
  updated_at?: string | null;
  client_name?: string | null;
  /** Identité de lifecycle (P1 MISSION-STATE) : ligne Assignment serveur. */
  assignment_id?: number | null;
  /** Révision monotone serveur — un snapshot plus ancien ne s'applique pas. */
  mission_revision?: number | null;
  [key: string]: unknown;
};

export type DriverMissionDetail = DriverMission;

export type DriverStatusTransitionPayload = {
  missionId: number;
  targetStatus: DriverTransitionStatus;
  idempotencyKey: string;
  reason?: string | null;
};

export type DriverLocationPayload = {
  latitude: number;
  longitude: number;
  accuracy?: number;
  heading?: number;
  speed?: number;
  /** Instant GNSS / fix — figé à l'enqueue ; alias wire `timestamp`. */
  timestamp?: string;
  /** Instant métier hashé serveur — figé à l'enqueue ; wire `recorded_at`. */
  recordedAt?: string;
  /** Instant 1ʳᵉ mise en file — figé à l'enqueue ; wire `sent_at` (hors hash). */
  sentAt?: string;
  isBackground?: boolean;
  missionId?: number | null;
  locationMode?: "mission_live" | "availability_presence" | "observability_only";
  trackingEventId?: string;
  trackingSessionId?: string | null;
  sessionGeneration?: number | null;
  sequenceId?: number | null;
  trackingGenerationId?: string | null;
  missionContextVersion?: number | null;
  trackingIdentityId?: string | null;
  /** Identité stable du fix natif (hash timestamp+lat+lon ou id OS). */
  captureId?: string | null;
};

export type DriverLocationAckStatus =
  | "accepted"
  | "queued"
  | "duplicate"
  | "stale"
  | "ignored"
  | "rejected"
  | "ingested"
  | "ingested_non_persisted"
  | "partially_ingested"
  | "persisted";

export type DriverLocationDurability = "persisted_sync" | "queued_async" | null;

export type DriverLocationAck = {
  ack_status: DriverLocationAckStatus;
  durability?: DriverLocationDurability;
  accept_reason?: string | null;
  tracking_event_id?: string | null;
  location_event_id?: string | null;
  trace_id?: string | null;
  ingested_event_ids?: string[] | null;
  retry_event_ids?: string[] | null;
};

export type DriverSocketEvent = {
  event_id?: string;
  event_type: DriverSocketEventType;
  mission_id: number;
  updated_at?: string;
  event_sequence?: number;
  payload?: Record<string, unknown>;
};

export type DriverPushRegistrationPayload = {
  token: string;
  driverId: number;
  deviceId: string;
  platform?: "ios" | "android";
  provider?: "expo" | "fcm";
};

