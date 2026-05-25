import { COCKPIT_EVENT_BUS_TTL_MS } from "./cockpitThresholds";

export type CockpitEventType =
  | "DRIVER_SELECTED"
  | "DRIVER_CLEARED"
  | "MISSION_FOCUSED"
  | "SEARCH_OPENED"
  | "SEARCH_CLEARED"
  | "FILTERS_OPENED"
  | "FILTERS_CLOSED"
  | "INCIDENT_ACK"
  | "RECENTER_REQUESTED"
  | "MODE_TRANSITION"
  | "GPS_UPDATE";

export type CockpitEvent = {
  type: CockpitEventType;
  atMs: number;
  source: string;
  /** false par défaut — événements critiques jamais coalescés. */
  coalescable?: boolean;
  payload?: Record<string, unknown>;
};

type Listener = (event: CockpitEvent) => void;

/** Bus cockpit typé — scope local, déterministe, rejouable en dev/test. */
export class CockpitEventBus {
  private journal: CockpitEvent[] = [];
  private listeners = new Set<Listener>();
  private maxJournal = 500;

  emit(event: Omit<CockpitEvent, "atMs"> & { atMs?: number }): CockpitEvent {
    const full: CockpitEvent = {
      ...event,
      atMs: event.atMs ?? Date.now(),
    };
    this.journal.push(full);
    const cutoff = full.atMs - COCKPIT_EVENT_BUS_TTL_MS;
    this.journal = this.journal.filter((e) => e.atMs >= cutoff);
    if (this.journal.length > this.maxJournal) {
      this.journal = this.journal.slice(-this.maxJournal);
    }
    for (const listener of this.listeners) {
      listener(full);
    }
    return full;
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  getJournal(): readonly CockpitEvent[] {
    return this.journal;
  }

  replay(fromMs?: number): CockpitEvent[] {
    if (fromMs == null) return [...this.journal];
    return this.journal.filter((e) => e.atMs >= fromMs);
  }

  clear(): void {
    this.journal = [];
  }
}

let sharedBus: CockpitEventBus | null = null;

export function getCockpitEventBus(): CockpitEventBus {
  if (!sharedBus) sharedBus = new CockpitEventBus();
  return sharedBus;
}

export function resetCockpitEventBusForTests(): void {
  sharedBus = null;
}
