import { DriverMissionStatus, DriverTransitionStatus } from "./types";

type DriverStatusUx = {
  label: string;
  terminal: boolean;
  nextTransitions: DriverTransitionStatus[];
  ctas: string[];
};

export const DRIVER_STATUS_DICTIONARY_VERSION = "1.0.0";

const DRIVER_STATUS_UX_MAP: Record<DriverMissionStatus, DriverStatusUx> = {
  ASSIGNED: {
    label: "Mission assignee",
    terminal: false,
    nextTransitions: ["EN_ROUTE", "CANCELLED"],
    ctas: ["Rejoindre", "Refuser"],
  },
  EN_ROUTE: {
    label: "En route",
    terminal: false,
    nextTransitions: ["ARRIVED", "CANCELLED"],
    ctas: ["Arrive", "Contacter"],
  },
  ARRIVED: {
    label: "Arrive",
    terminal: false,
    nextTransitions: ["IN_PROGRESS", "CANCELLED"],
    ctas: ["Demarrer"],
  },
  IN_PROGRESS: {
    label: "En cours",
    terminal: false,
    nextTransitions: ["COMPLETED", "FAILED"],
    ctas: ["Terminer", "Signaler"],
  },
  COMPLETED: {
    label: "Terminee",
    terminal: true,
    nextTransitions: [],
    ctas: ["Voir"],
  },
  CANCELLED: {
    label: "Annulee",
    terminal: true,
    nextTransitions: [],
    ctas: ["Voir"],
  },
  REASSIGNED: {
    label: "Reassignee",
    terminal: true,
    nextTransitions: [],
    ctas: ["Rafraichir"],
  },
  NO_SHOW: {
    label: "Client absent",
    terminal: true,
    nextTransitions: [],
    ctas: ["Signaler"],
  },
  FAILED: {
    label: "Echec mission",
    terminal: true,
    nextTransitions: [],
    ctas: ["Support"],
  },
};

const UNKNOWN_STATUS_UX: DriverStatusUx = {
  label: "Statut en cours de mise a jour",
  terminal: false,
  nextTransitions: [],
  ctas: ["Rafraichir"],
};

/**
 * `Booking.serialize` côté driver: `status` = `.value.lower()` (snake tourné, ex. `en_route`, `canceled` US, pas `CANCELLED` côté client).
 */
const API_LOWERCASE_TO_UX: Record<string, DriverMissionStatus> = {
  pending: "ASSIGNED",
  awaiting_client_payment: "ASSIGNED",
  accepted: "ASSIGNED",
  assigned: "ASSIGNED",
  arrived: "ARRIVED",
  en_route: "EN_ROUTE",
  in_progress: "IN_PROGRESS",
  completed: "COMPLETED",
  return_completed: "COMPLETED",
  canceled: "CANCELLED",
  cancelled: "CANCELLED",
  no_show: "NO_SHOW",
  reassigned: "REASSIGNED",
  failed: "FAILED",
};

/** SCREAMING_SNAKE inattendu ou alias (ex. CANCELED en majuscule une seule variante) */
const API_UPPER_EXTRA_TO_UX: Record<string, DriverMissionStatus> = {
  PENDING: "ASSIGNED",
  AWAITING_CLIENT_PAYMENT: "ASSIGNED",
  ACCEPTED: "ASSIGNED",
  CANCELED: "CANCELLED",
};

/**
 * Résout toute forme d’échantillon (snake, enum Python lower, déjà SCREAMING) vers une clé UX.
 */
export function resolveDriverStatusForUx(status: string | null | undefined): DriverMissionStatus {
  const raw = (status ?? "").trim();
  if (!raw) {
    return "ASSIGNED";
  }
  const lower = raw.toLowerCase();
  const fromLower = API_LOWERCASE_TO_UX[lower];
  if (fromLower) {
    return fromLower;
  }
  const upper = raw.toUpperCase();
  if (API_UPPER_EXTRA_TO_UX[upper]) {
    return API_UPPER_EXTRA_TO_UX[upper]!;
  }
  if (upper in DRIVER_STATUS_UX_MAP) {
    return upper as DriverMissionStatus;
  }
  console.warn("[driver_status_dictionary_mismatch_event]", {
    received_status: status ?? null,
    dictionary_version: DRIVER_STATUS_DICTIONARY_VERSION,
  });
  return "ASSIGNED";
}

export function normalizeDriverMissionStatus(status: string | null | undefined): DriverMissionStatus {
  return resolveDriverStatusForUx(status);
}

export function getDriverStatusUx(status: string | null | undefined): DriverStatusUx {
  const key = resolveDriverStatusForUx(status);
  return DRIVER_STATUS_UX_MAP[key] ?? UNKNOWN_STATUS_UX;
}
