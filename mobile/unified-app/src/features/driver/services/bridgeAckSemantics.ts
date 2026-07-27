import type { DriverLocationAckStatus } from "../types";

export const BRIDGE_CONFIRMED_ACK_STATUSES = new Set<DriverLocationAckStatus>([
  "accepted",
  "duplicate",
  "ingested",
  "persisted",
]);

export const BRIDGE_QUEUED_ACK_STATUS: DriverLocationAckStatus = "queued";

export type BridgeAckApplication = {
  lastAckAt: string | null;
  lastAckIsQueued: boolean;
  lastAckStatus: DriverLocationAckStatus | null;
  lastAckError: string | null;
  lastAckAttemptSeq: number | null;
  lastAckEventId: string | null;
};

/** Applique la sémantique ACK (sans corrélation seq/event — faite par l’appelant). */
export function resolveBridgeAckFields(
  ackStatus: DriverLocationAckStatus,
  ackAt: string
): BridgeAckApplication {
  if (ackStatus === BRIDGE_QUEUED_ACK_STATUS) {
    return {
      lastAckAt: ackAt,
      lastAckIsQueued: true,
      lastAckStatus: ackStatus,
      lastAckError: null,
      lastAckAttemptSeq: null,
      lastAckEventId: null,
    };
  }
  if (BRIDGE_CONFIRMED_ACK_STATUSES.has(ackStatus)) {
    return {
      lastAckAt: ackAt,
      lastAckIsQueued: false,
      lastAckStatus: ackStatus,
      lastAckError: null,
      lastAckAttemptSeq: null,
      lastAckEventId: null,
    };
  }
  return {
    lastAckAt: null,
    lastAckIsQueued: false,
    lastAckStatus: ackStatus,
    lastAckError: `ack_${ackStatus}`,
    lastAckAttemptSeq: null,
    lastAckEventId: null,
  };
}

export function formatBridgeSyncLabel(input: {
  gpsEnabled: boolean;
  isTracking: boolean;
  lastUpdate: number | undefined;
  lastAckAt: number | undefined;
  lastAckIsQueued: boolean;
  lastAckStatus: DriverLocationAckStatus | null | undefined;
  lastAckError: string | null | undefined;
  currentAttemptSeq: number;
  lastAckAttemptSeq: number | null | undefined;
  currentAttemptEventId: string | null | undefined;
  lastAckEventId: string | null | undefined;
  formatSyncTime: (ms: number) => string;
}): string {
  if (!input.gpsEnabled) return "GPS indisponible";
  if (!input.isTracking) return "GPS prêt";

  const seqMatch =
    input.lastAckAttemptSeq != null &&
    input.lastAckAttemptSeq === input.currentAttemptSeq;
  const eventMatch =
    input.lastAckEventId != null &&
    input.currentAttemptEventId != null &&
    input.lastAckEventId === input.currentAttemptEventId;
  const noError = !input.lastAckError;
  const status = input.lastAckStatus ?? null;

  if (seqMatch && eventMatch && noError && status === "queued" && input.lastAckAt != null) {
    return `GPS connecté • Mis en file ${input.formatSyncTime(input.lastAckAt)}`;
  }
  if (
    seqMatch &&
    eventMatch &&
    noError &&
    status != null &&
    BRIDGE_CONFIRMED_ACK_STATUSES.has(status) &&
    input.lastAckAt != null
  ) {
    return `GPS connecté • Confirmé ${input.formatSyncTime(input.lastAckAt)}`;
  }
  if (seqMatch && eventMatch && status === "partially_ingested") {
    return "GPS connecté • Partiellement reçu";
  }
  if (
    (input.lastAckError && seqMatch && eventMatch) ||
    (status != null &&
      seqMatch &&
      eventMatch &&
      (status === "stale" ||
        status === "ignored" ||
        status === "rejected"))
  ) {
    return "GPS connecté • Non confirmé";
  }
  if (input.lastUpdate != null) {
    return `GPS connecté • Envoyé ${input.formatSyncTime(input.lastUpdate)}`;
  }
  return "GPS connecté";
}
