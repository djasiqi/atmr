/**
 * File locale des signalements chauffeur.
 * Ne conserve ni nom, ni adresse, ni telephone du passager.
 */

export type IncidentHold = "pending" | "unassigned" | "rejected";

export type IncidentFailure = "network" | "temporary" | "unassigned" | "rejected";

export type IncidentSendOutcome = "ok" | IncidentFailure;

export type StoredIncident = {
  clientEventId: string;
  bookingId: number;
  driverUserId: string;
  category: string;
  detail?: string;
  status: "pending" | "rejected";
  createdAt: string;
  hold: IncidentHold;
  failure?: IncidentFailure;
};

export type IncidentStorage = {
  read: () => Promise<string | null>;
  write: (value: string) => Promise<void>;
};

export type FlushReport = {
  transmitted: string[];
  pending: string[];
  unassigned: string[];
  rejected: string[];
  examined: string[];
};

const DETAIL_MAX = 180;

export function createIncidentEventId(): string {
  const cryptoObj = globalThis.crypto;
  if (cryptoObj && typeof cryptoObj.randomUUID === "function") {
    return cryptoObj.randomUUID();
  }
  return `evt_${Date.now().toString(16)}_${Math.random().toString(16).slice(2)}`;
}

export function memoryIncidentStorage(initial = ""): IncidentStorage & { snapshot: () => string } {
  let raw = initial;
  return {
    read: async () => raw || null,
    write: async (value: string) => {
      raw = value;
    },
    snapshot: () => raw,
  };
}

const TEMPORARY_STATUS = new Set([401, 408, 429]);

const REJECTED_STATUS = new Set([400, 404, 409, 422]);

/**
 * Un refus de validation n'est pas une panne réseau.
 * 401, 408, 429 et 5xx restent récupérables, avec la même clé.
 * Les autres 4xx connus du contrat sont définitifs.
 * Un statut inconnu reste incertain : il n'est pas reclassé en refus.
 */
export function classifyIncidentSendError(
  status: number | null,
  code: string | null
): IncidentFailure {
  if (status === null || status === 0) return "network";
  if (status === 403 && code === "chauffeur_non_affecte") return "unassigned";
  if (TEMPORARY_STATUS.has(status) || (status >= 500 && status <= 599)) return "temporary";
  if (status === 403 || REJECTED_STATUS.has(status)) return "rejected";
  return "network";
}

export function incidentRetriesAutomatically(item: StoredIncident): boolean {
  return item.status !== "rejected" && item.hold !== "rejected";
}

function sanitize(input: StoredIncident): StoredIncident {
  const detail = (input.detail ?? "").trim().slice(0, DETAIL_MAX);
  const rejected = input.status === "rejected" || input.hold === "rejected";
  const stored: StoredIncident = {
    clientEventId: input.clientEventId.trim(),
    bookingId: input.bookingId,
    driverUserId: input.driverUserId.trim(),
    category: input.category.trim(),
    status: rejected ? "rejected" : "pending",
    createdAt: input.createdAt,
    hold: rejected ? "rejected" : input.hold === "unassigned" ? "unassigned" : "pending",
  };
  if (detail) stored.detail = detail;
  if (
    input.failure === "network" ||
    input.failure === "temporary" ||
    input.failure === "unassigned" ||
    input.failure === "rejected"
  ) {
    stored.failure = input.failure;
  }
  return stored;
}

async function readAll(storage: IncidentStorage): Promise<StoredIncident[]> {
  const raw = await storage.read();
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((item): item is StoredIncident => {
        if (!item || typeof item !== "object") return false;
        const row = item as Partial<StoredIncident>;
        return (
          typeof row.clientEventId === "string" &&
          typeof row.bookingId === "number" &&
          typeof row.driverUserId === "string" &&
          typeof row.category === "string"
        );
      })
      .map((item) =>
        sanitize({
          ...item,
          status: item.status === "rejected" ? "rejected" : "pending",
          hold: item.hold ?? "pending",
        })
      );
  } catch {
    return [];
  }
}

async function writeAll(storage: IncidentStorage, rows: StoredIncident[]): Promise<void> {
  await storage.write(JSON.stringify(rows.map(sanitize)));
}

export async function enqueueIncident(
  storage: IncidentStorage,
  input: Omit<StoredIncident, "status" | "hold"> & { hold?: IncidentHold }
): Promise<StoredIncident> {
  const stored = sanitize({ ...input, status: "pending", hold: input.hold ?? "pending" });
  const current = await readAll(storage);
  const next = current.filter((item) => item.clientEventId !== stored.clientEventId);
  next.push(stored);
  await writeAll(storage, next);
  return stored;
}

export async function listPendingIncidents(
  storage: IncidentStorage,
  driverUserId: string,
  bookingId?: number
): Promise<StoredIncident[]> {
  const rows = await readAll(storage);
  return rows.filter(
    (item) =>
      item.driverUserId === driverUserId &&
      (bookingId === undefined || item.bookingId === bookingId)
  );
}

export async function flushIncidentOutbox(
  storage: IncidentStorage,
  driverUserId: string,
  send: (item: StoredIncident) => Promise<IncidentSendOutcome>
): Promise<FlushReport> {
  const rows = await readAll(storage);
  const report: FlushReport = {
    transmitted: [],
    pending: [],
    unassigned: [],
    rejected: [],
    examined: [],
  };
  const kept: StoredIncident[] = [];
  for (const item of rows) {
    if (item.driverUserId !== driverUserId) {
      kept.push(item);
      continue;
    }
    if (!incidentRetriesAutomatically(item)) {
      report.examined.push(item.clientEventId);
      kept.push(item);
      continue;
    }
    const outcome = await send(item);
    if (outcome === "ok") {
      report.transmitted.push(item.clientEventId);
      continue;
    }
    if (outcome === "rejected") {
      report.rejected.push(item.clientEventId);
      kept.push({ ...item, status: "rejected", hold: "rejected", failure: "rejected" });
      continue;
    }
    const hold: IncidentHold = outcome === "unassigned" ? "unassigned" : "pending";
    kept.push({ ...item, status: "pending", hold, failure: outcome });
    if (hold === "unassigned") report.unassigned.push(item.clientEventId);
    else report.pending.push(item.clientEventId);
  }
  await writeAll(storage, kept);
  return report;
}
