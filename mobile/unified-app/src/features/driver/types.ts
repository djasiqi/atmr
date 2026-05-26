export type DriverMissionStatus =
  | "ASSIGNED"
  | "EN_ROUTE"
  | "ARRIVED"
  | "IN_PROGRESS"
  | "COMPLETED"
  | "CANCELLED"
  | "REASSIGNED"
  | "NO_SHOW"
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
  updated_at?: string | null;
  client_name?: string | null;
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
  timestamp?: string;
  isBackground?: boolean;
  missionId?: number | null;
  locationMode?: "mission_live" | "availability_presence" | "observability_only";
  trackingEventId?: string;
};

export type DriverLocationAckStatus =
  | "accepted"
  | "duplicate"
  | "stale"
  | "ignored"
  | "rejected";

export type DriverLocationAck = {
  ack_status: DriverLocationAckStatus;
  accept_reason?: string | null;
  tracking_event_id?: string | null;
  trace_id?: string | null;
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

