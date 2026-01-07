/**
 * Types TypeScript pour les événements Socket.IO - ATMR/Lirie
 * 
 * Cette définition de types correspond à la documentation dans docs/SOCKETIO_EVENTS.md
 * 
 * @see docs/SOCKETIO_EVENTS.md
 */

// ✅ Schéma d'enrichissement automatique (ajouté par le backend)
export interface SocketEventEnrichment {
  event_id?: string; // UUID unique pour déduplication
  version?: string; // Version du schéma (actuellement "1.0")
  timestamp?: string; // Timestamp ISO 8601
}

// ============================================================================
// ÉVÉNEMENTS DE CONNEXION
// ============================================================================

export interface ConnectedEvent {
  message: string; // "✅ Chauffeur connecté" | "✅ Entreprise connectée à company_42"
}

export interface UnauthorizedEvent {
  error: string; // "Token invalide." | "Rôle non autorisé" | "Utilisateur non trouvé"
}

// ============================================================================
// ÉVÉNEMENTS DE CHAT
// ============================================================================

export interface TeamChatMessageRequest {
  content: string;
  receiver_id: number;
  image_url?: string;
  pdf_url?: string;
  pdf_filename?: string;
  pdf_size?: number; // bytes
  _localId?: string; // UUID local pour tracking côté client
}

export interface TeamChatMessageResponse extends SocketEventEnrichment {
  id: number;
  content: string;
  sender_id: number;
  sender_role: "driver" | "company";
  receiver_id: number;
  created_at: string; // ISO 8601
  image_url?: string;
  pdf_url?: string;
  pdf_filename?: string;
  pdf_size?: number;
}

export interface TypingStartRequest {
  receiver_id: number;
}

export interface TypingStopRequest {
  receiver_id: number;
}

// ============================================================================
// ÉVÉNEMENTS DE LOCALISATION
// ============================================================================

export interface DriverLocationRequest {
  latitude: number;
  longitude: number;
  speed?: number; // km/h
  heading?: number; // degrés
  accuracy?: number; // mètres
  timestamp?: number | string; // milliseconds ou ISO 8601
  driver_id?: number; // Optionnel (fallback sur JWT)
}

export interface DriverLocationResponse extends SocketEventEnrichment {
  driver_id: number;
  latitude: number;
  longitude: number;
  speed?: number;
  heading?: number;
  accuracy?: number;
  timestamp: string; // ISO 8601
}

export interface DriverLocationBatchRequest {
  locations: Array<{
    latitude: number;
    longitude: number;
    speed?: number;
    heading?: number;
    accuracy?: number;
    timestamp: number; // milliseconds
  }>;
  driver_id?: number;
}

export interface GetDriverLocationsRequest {
  company_id?: number; // Optionnel (déduit depuis JWT)
}

export interface DriverLocationsResponse {
  locations: Array<{
    driver_id: number;
    latitude: number;
    longitude: number;
    speed?: number;
    heading?: number;
    accuracy?: number;
    timestamp: string; // ISO 8601
  }>;
}

// ============================================================================
// ÉVÉNEMENTS DE BOOKING (COURSES)
// ============================================================================

export interface NewBookingEvent extends SocketEventEnrichment {
  id: number;
  status: "ASSIGNED" | "EN_ROUTE" | "IN_PROGRESS" | "COMPLETED" | "RETURN_COMPLETED" | "CANCELED";
  pickup_location?: string;
  dropoff_location?: string;
  scheduled_time?: string; // ISO 8601
  client_name?: string;
  // ✅ P1-4 Phase 3.1: Déprécié - utiliser client.contact_phone à la place
  /** @deprecated Utiliser client.contact_phone à la place */
  client_phone?: string;
  // ... autres champs Booking (voir type Booking dans api.ts)
}

export interface BookingUpdatedEvent extends SocketEventEnrichment {
  id: number;
  status?: "ASSIGNED" | "EN_ROUTE" | "IN_PROGRESS" | "COMPLETED" | "RETURN_COMPLETED" | "CANCELED";
  // ... champs mis à jour
}

export interface BookingCancelledEvent extends SocketEventEnrichment {
  id: number;
}

// ============================================================================
// ÉVÉNEMENTS DE DISPATCH
// ============================================================================

export interface DispatchRunStartedEvent extends SocketEventEnrichment {
  dispatch_run_id: string;
  date: string; // YYYY-MM-DD
}

export interface DispatchRunCompletedEvent extends SocketEventEnrichment {
  dispatch_run_id: string;
  date: string; // YYYY-MM-DD
  assignments_count: number;
}

export interface DispatchRunFailedEvent extends SocketEventEnrichment {
  dispatch_run_id: string;
  date: string; // YYYY-MM-DD
  error: string;
}

export interface DispatchAssignmentCreatedEvent extends SocketEventEnrichment {
  assignment_id: string;
  booking_id: number;
  driver_id: number;
}

export interface DriverAssignmentReceivedEvent extends SocketEventEnrichment {
  assignment_id: string;
  booking_id: number;
}

export interface DispatchAssignmentUpdatedEvent extends SocketEventEnrichment {
  assignment_id: string;
  booking_id: number;
  driver_id: number;
  fields: Record<string, unknown>; // Champs mis à jour
}

export interface DispatchAssignmentCancelledEvent extends SocketEventEnrichment {
  assignment_id: string;
  booking_id: number;
  driver_id: number;
}

export interface DispatchDelayDetectedEvent extends SocketEventEnrichment {
  assignment_id: string;
  booking_id: number;
  driver_id: number;
  driver_name?: string;
  driver_phone?: string;
  driver_vehicle?: string;
  delay_minutes: number;
  has_alternative: boolean;
  is_dropoff: boolean;
  alternative_driver_id?: number;
  alternative_delay_minutes?: number;
}

export interface DriverDelayDetectedEvent extends SocketEventEnrichment {
  assignment_id: string;
  booking_id: number;
  driver_id: number;
  delay_minutes: number;
  is_dropoff: boolean;
}

// ============================================================================
// ÉVÉNEMENTS DE PLANNING
// ============================================================================

export interface PlanningShiftCreatedEvent extends SocketEventEnrichment {
  shift_id: number;
  driver_id: number;
  start_time: string; // ISO 8601
  end_time: string; // ISO 8601
}

export interface PlanningShiftUpdatedEvent extends SocketEventEnrichment {
  shift_id: number;
  driver_id: number;
  start_time: string; // ISO 8601
  end_time: string; // ISO 8601
}

export interface PlanningShiftDeletedEvent extends SocketEventEnrichment {
  shift_id: number;
}

// ============================================================================
// ÉVÉNEMENTS DE ROOMS
// ============================================================================

export interface JoinDriverRoomRequest {
  driver_id?: number; // Optionnel (fallback sur JWT)
}

export interface JoinCompanyRequest {
  company_id?: number; // Optionnel (déduit depuis JWT)
}

export interface JoinedRoomEvent {
  rooms: string[]; // ["driver_101", "company_42"]
}

export interface JoinedCompanyEvent {
  company_id: number;
  room: string; // "company_42"
}

// ============================================================================
// ÉVÉNEMENTS DE HEARTBEAT
// ============================================================================

export interface PongEvent {
  timestamp: string; // ISO 8601
}

export interface DriverHeartbeatRequest {
  last_mission_id?: number;
  location?: {
    lat: number;
    lon: number;
  };
  timestamp: number; // milliseconds
}

export interface DriverHeartbeatAckEvent {
  received_at: string; // ISO 8601
}

// ============================================================================
// ÉVÉNEMENTS D'ALERTES PROACTIVES
// ============================================================================

export interface SubscribeAlertsRequest {
  company_id: string; // "company_42"
  alert_types?: Array<"delay_risk" | "rl_explanation">;
  filters?: {
    risk_levels?: Array<"high" | "medium" | "low">;
    booking_ids?: number[];
  };
}

export interface SubscriptionConfirmedEvent {
  company_id: string;
  room: string; // "company_42"
  alert_types: string[];
  filters: Record<string, unknown>;
}

export interface UnsubscribeAlertsRequest {
  company_id: string;
}

export interface UnsubscriptionConfirmedEvent {
  company_id: string;
}

export interface DelayAlertEvent {
  type: "delay_risk_alert";
  data: {
    booking_id: number;
    risk_level: "high" | "medium" | "low";
    delay_minutes: number;
    driver_id: number;
  };
  timestamp: string; // ISO 8601
  priority: "high" | "medium" | "low";
}

export interface RLExplanationEvent {
  type: "rl_explanation";
  data: {
    booking_id: number;
    driver_id: number;
    explanation: string;
    confidence: number; // 0.0 - 1.0
  };
  timestamp: string; // ISO 8601
}

export interface RequestExplanationRequest {
  booking_id: number;
  driver_id: number;
}

// ============================================================================
// ÉVÉNEMENTS D'ERREUR
// ============================================================================

export interface ErrorEvent {
  error: string;
}

export interface RateLimitExceededEvent {
  event: "rate_limit_exceeded";
  message: string;
  attempts: number;
  retry_after_seconds: number;
}

// ============================================================================
// ÉVÉNEMENTS SPÉCIAUX
// ============================================================================

export interface DriverArrivedAtPickupEvent {
  driver_id: number;
}

export interface DriverArrivedAtDropoffEvent {
  driver_id: number;
}

// ============================================================================
// UNION TYPES POUR TYPAGE STRICT
// ============================================================================

/**
 * Union type de tous les événements Socket.IO émis par le serveur
 */
export type ServerSocketEvent =
  | ConnectedEvent
  | UnauthorizedEvent
  | TeamChatMessageResponse
  | DriverLocationResponse
  | DriverLocationsResponse
  | NewBookingEvent
  | BookingUpdatedEvent
  | BookingCancelledEvent
  | DispatchRunStartedEvent
  | DispatchRunCompletedEvent
  | DispatchRunFailedEvent
  | DispatchAssignmentCreatedEvent
  | DriverAssignmentReceivedEvent
  | DispatchAssignmentUpdatedEvent
  | DispatchAssignmentCancelledEvent
  | DispatchDelayDetectedEvent
  | DriverDelayDetectedEvent
  | PlanningShiftCreatedEvent
  | PlanningShiftUpdatedEvent
  | PlanningShiftDeletedEvent
  | JoinedRoomEvent
  | JoinedCompanyEvent
  | PongEvent
  | DriverHeartbeatAckEvent
  | SubscriptionConfirmedEvent
  | UnsubscriptionConfirmedEvent
  | DelayAlertEvent
  | RLExplanationEvent
  | ErrorEvent
  | RateLimitExceededEvent
  | DriverArrivedAtPickupEvent
  | DriverArrivedAtDropoffEvent;

/**
 * Union type de tous les événements Socket.IO émis par le client
 */
export type ClientSocketEvent =
  | TeamChatMessageRequest
  | TypingStartRequest
  | TypingStopRequest
  | DriverLocationRequest
  | DriverLocationBatchRequest
  | GetDriverLocationsRequest
  | JoinDriverRoomRequest
  | JoinCompanyRequest
  | DriverHeartbeatRequest
  | SubscribeAlertsRequest
  | UnsubscribeAlertsRequest
  | RequestExplanationRequest;

// ============================================================================
// HELPERS POUR DÉDUPLICATION
// ============================================================================

/**
 * Vérifie si un événement a déjà été traité (déduplication)
 */
export function isEventDuplicate(
  eventId: string | undefined | null,
  seenEventIds: Set<string>
): boolean {
  if (!eventId || typeof eventId !== "string") {
    return false; // Pas d'event_id, pas de déduplication possible
  }
  return seenEventIds.has(eventId);
}

/**
 * Ajoute un event_id au Set de déduplication
 */
export function markEventAsSeen(
  eventId: string | undefined | null,
  seenEventIds: Set<string>,
  maxSize: number = 1000
): void {
  if (!eventId || typeof eventId !== "string") {
    return; // Pas d'event_id, ignorer
  }
  
  seenEventIds.add(eventId);
  
  // Limiter la taille du Set (FIFO)
  if (seenEventIds.size > maxSize) {
    const first = seenEventIds.values().next().value;
    if (first && typeof first === "string") {
      seenEventIds.delete(first);
    }
  }
}
