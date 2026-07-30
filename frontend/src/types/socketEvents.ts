/**
 * Types TypeScript pour les events Socket.IO
 * Basé sur docs/SOCKET_EVENTS.md
 * 
 * @module socketEvents
 */

// ============================================================================
// Events Client → Server
// ============================================================================

export interface ConnectPayload {
  auth?: {
    token: string;
  };
}

export interface PingPayload {
  timestamp?: number;
}

export interface DriverHeartbeatPayload {
  last_mission_id?: number;
  location?: {
    lat: number;
    lon: number;
  };
  timestamp: number;
}

export interface JoinDriverRoomPayload {
  driver_id?: number;
}

export interface JoinCompanyPayload {
  // Vide, optionnel
}

export interface DriverLocationPayload {
  driver_id?: number;
  latitude: number;
  longitude: number;
  speed?: number;
  heading?: number;
  accuracy?: number;
  timestamp?: number | string;
}

export interface DriverLocationBatchPosition {
  latitude: number;
  longitude: number;
  speed?: number;
  heading?: number;
  accuracy?: number;
  timestamp?: number | string;
}

export interface DriverLocationBatchPayload {
  driver_id?: number;
  positions: DriverLocationBatchPosition[];
}

export interface TeamChatMessagePayload {
  _localId?: string;
  content?: string;
  receiver_id?: number;
  image_url?: string;
  pdf_url?: string;
  pdf_filename?: string;
  pdf_size?: number;
}

export interface TypingStartPayload {
  receiver_id?: number;
}

export interface TypingStopPayload {
  receiver_id?: number;
}

// ============================================================================
// Events Server → Client
// ============================================================================

export interface ConnectedPayload {
  status: "connected";
  client_id: string;
  timestamp: string;
}

export interface UnauthorizedPayload {
  error: string;
}

export interface RateLimitExceededPayload {
  event: "rate_limit_exceeded";
  message: string;
  attempts: number;
  retry_after_seconds?: number;
}

export interface PongPayload {
  timestamp: string;
}

export interface JoinedDriverRoomPayload {
  driver_id: number;
  room: string;
}

export interface JoinedCompanyPayload {
  company_id: number;
  room: string;
  /** Curseur temps réel au moment du join (PR1 dashboard reliability) — `null` si Redis dégradé. */
  subscribed_cursor?: number | null;
  realtime_sequence_health?: 'ok' | 'degraded';
}

/** P3 — invalide le snapshot GET /company_dispatch/delays/live (debounce côté client). */
export interface DelayLiveInvalidatePayload {
  date: string;
  reason: string;
  scope?: string;
  event_id?: string;
  event_type?: string;
  version?: string;
  timestamp?: string;
}

export interface DriverLocationUpdatePayload {
  driver_id: number;
  company_id?: number;
  first_name?: string | null;
  // ✅ Backend émet "lat"/"lon" (format abrégé optimisé)
  // Pour compatibilité, les frontends doivent accepter les deux formats
  lat: number;
  lon: number;
  // Propriétés optionnelles pour compatibilité ascendante
  latitude?: number; // Alias de "lat"
  longitude?: number; // Alias de "lon"
  speed?: number;
  speed_mps?: number;
  heading?: number;
  accuracy?: number;
  accuracy_m?: number;
  ts?: string; // Backend utilise "ts" (pas "timestamp")
  timestamp?: string; // Alias de "ts"
  recorded_at?: string;
  received_at?: string;
  sent_at?: string;
  location_mode?: "mission_live" | "availability_presence" | "passive_last_known";
  location_status?: "live" | "recent" | "stale" | "offline";
  presence_status?: "online" | "degraded" | "offline";
  last_seen_seconds?: number | null;
  mission_id?: number | null;
  source?: string;
}

export interface DriverLiveStateUpdatePayload {
  driver_id: number;
  company_id?: number;
  lat?: number;
  lng?: number;
  status?: "available" | "assigned" | "busy" | "offline";
  mission_status?: string;
  presence_status?: "online" | "degraded" | "offline";
  location_status?: "live" | "recent" | "stale" | "offline";
  location_mode?: "mission_live" | "availability_presence" | "passive_last_known";
  last_seen_seconds?: number | null;
  mission_id?: number | null;
  timestamp?: string;
  recorded_at?: string;
  received_at?: string;
}

export interface TeamChatMessageReceivedPayload {
  id: number;
  sender_id: number;
  sender_role: "driver" | "company";
  receiver_id: number | null;
  content: string | null;
  image_url: string | null;
  pdf_url: string | null;
  pdf_filename: string | null;
  created_at: string;
  _localId?: string;
}

export interface TypingStartReceivedPayload {
  user_id: number;
  receiver_id: number | null;
}

export interface TypingStopReceivedPayload {
  user_id: number;
  receiver_id: number | null;
}

export interface DriverLocationItem {
  driver_id: number;
  // ✅ Backend émet "lat"/"lon" (cohérent avec Redis et Socket.IO)
  lat: number;
  lon: number;
  // Propriétés optionnelles pour compatibilité ascendante
  latitude?: number; // Alias de "lat"
  longitude?: number; // Alias de "lon"
  speed?: number;
  heading?: number;
  accuracy?: number;
  ts?: string;
  timestamp?: string; // Alias de "ts"
  source?: string;
}

export interface DriverLocationsPayload {
  drivers: DriverLocationItem[];
}

export interface ErrorPayload {
  error: string;
}

// ============================================================================
// Map des events pour typage strict
// ============================================================================

// Events Client → Server
export interface SocketEventMapClient {
  connect: ConnectPayload;
  disconnect: void;
  ping: PingPayload;
  "driver:heartbeat": DriverHeartbeatPayload;
  join_driver_room: JoinDriverRoomPayload;
  join_company: JoinCompanyPayload;
  driver_location: DriverLocationPayload;
  driver_location_batch: DriverLocationBatchPayload;
  team_chat_message: TeamChatMessagePayload;
  typing_start: TypingStartPayload;
  typing_stop: TypingStopPayload;
  get_driver_locations: void;
}

// Events Server → Client
export interface SocketEventMapServer {
  connected: ConnectedPayload;
  unauthorized: UnauthorizedPayload;
  rate_limit_exceeded: RateLimitExceededPayload;
  pong: PongPayload;
  joined_driver_room: JoinedDriverRoomPayload;
  joined_company: JoinedCompanyPayload;
  driver_location_update: DriverLocationUpdatePayload;
  driver_live_state_update: DriverLiveStateUpdatePayload;
  team_chat_message: TeamChatMessageReceivedPayload;
  typing_start: TypingStartReceivedPayload;
  typing_stop: TypingStopReceivedPayload;
  driver_locations: DriverLocationsPayload;
  error: ErrorPayload;
}

// Union des deux maps (pour compatibilité)
export type SocketEventMap = SocketEventMapClient & SocketEventMapServer;

// ============================================================================
// Helpers pour typage
// ============================================================================

export type SocketEventNameClient = keyof SocketEventMapClient;
export type SocketEventNameServer = keyof SocketEventMapServer;
export type SocketEventName = SocketEventNameClient | SocketEventNameServer;

export type SocketEventPayload<T extends SocketEventName> = T extends keyof SocketEventMapClient
  ? SocketEventMapClient[T]
  : T extends keyof SocketEventMapServer
  ? SocketEventMapServer[T]
  : never;

