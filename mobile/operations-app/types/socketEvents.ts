/**
 * Types TypeScript pour les events Socket.IO (Mobile)
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
}

export interface DriverLocationUpdatePayload {
  driver_id: number;
  first_name: string | null;
  latitude: number;
  longitude: number;
  timestamp: string;
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
  latitude: number;
  longitude: number;
  timestamp: string;
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

