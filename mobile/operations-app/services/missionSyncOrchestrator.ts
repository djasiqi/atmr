/**
 * Orchestrateur unique pour GET /driver/me/bookings/since (via getAssignedTrips).
 * - Single-flight par `since` (tous les triggers partagent une requête en cours pour la même clé).
 * - Debounce mémoire par `since` + coalescing inter-triggers (fenêtre courte).
 */
import type { Booking } from "./api";
import { getAssignedTrips } from "./api";
import type { MissionSyncTrigger } from "./missionSyncTypes";

const DEBOUNCE_MS_BY_TRIGGER: Partial<Record<MissionSyncTrigger, number>> = {
  socket_connect: 2500,
  foreground: 2000,
  degraded_interval: 1500,
  reconcile_now: 800,
  reconcile_active: 800,
  manual_screen: 400,
  hydrate_empty: 400,
};

const DEFAULT_DEBOUNCE_MS = 1500;

/** Fenêtre min commune après un GET réussi : évite deux GET rapprochés pour triggers différents (même `since`). */
const GLOBAL_INTER_TRIGGER_COALESCE_MS = 450;

function bypassesDebounce(trigger: MissionSyncTrigger): boolean {
  return trigger === "socket_booking_event";
}

function debounceMs(trigger: MissionSyncTrigger): number {
  return DEBOUNCE_MS_BY_TRIGGER[trigger] ?? DEFAULT_DEBOUNCE_MS;
}

function keySince(since: string | undefined): string {
  return since ?? "__full__";
}

const inFlightBySince = new Map<string, Promise<Booking[]>>();
const lastCompletedAtBySince = new Map<string, number>();
const lastResultBySince = new Map<string, Booking[]>();

export type MissionSyncOptions = {
  since?: string;
};

/** Pour tests / observabilité légère — cardinalité fixe. */
export type MissionSyncOutcome =
  | "network"
  | "debounce_cache"
  | "shared_inflight";

export type MissionSyncResult = {
  bookings: Booking[];
  outcome: MissionSyncOutcome;
};

export async function requestMissionSyncWithMeta(
  trigger: MissionSyncTrigger,
  options?: MissionSyncOptions
): Promise<MissionSyncResult> {
  const since = options?.since;
  const sk = keySince(since);

  const existing = inFlightBySince.get(sk);
  if (existing) {
    const bookings = await existing;
    return { bookings: bookings.slice(), outcome: "shared_inflight" };
  }

  const now = Date.now();
  if (!bypassesDebounce(trigger)) {
    const lastAt = lastCompletedAtBySince.get(sk);
    const lastData = lastResultBySince.get(sk);
    if (lastAt !== undefined && lastData !== undefined) {
      const window = Math.max(
        debounceMs(trigger),
        GLOBAL_INTER_TRIGGER_COALESCE_MS
      );
      if (now - lastAt < window) {
        return { bookings: lastData.slice(), outcome: "debounce_cache" };
      }
    }
  }

  const p = (async () => {
    try {
      return await getAssignedTrips({ since, syncTrigger: trigger });
    } finally {
      inFlightBySince.delete(sk);
    }
  })();

  inFlightBySince.set(sk, p);
  try {
    const data = await p;
    lastCompletedAtBySince.set(sk, Date.now());
    lastResultBySince.set(sk, data.slice());
    return { bookings: data, outcome: "network" };
  } catch (e) {
    throw e;
  }
}

export async function requestMissionSync(
  trigger: MissionSyncTrigger,
  options?: MissionSyncOptions
): Promise<Booking[]> {
  const r = await requestMissionSyncWithMeta(trigger, options);
  return r.bookings;
}

export async function runMissionSync(
  trigger: MissionSyncTrigger,
  options?: MissionSyncOptions
): Promise<Booking[]> {
  return requestMissionSync(trigger, options);
}

export function shouldBypassDebounce(trigger: MissionSyncTrigger): boolean {
  return bypassesDebounce(trigger);
}

/** Tests uniquement */
export function __resetMissionSyncOrchestratorForTests(): void {
  inFlightBySince.clear();
  lastCompletedAtBySince.clear();
  lastResultBySince.clear();
}
