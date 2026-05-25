import type { CockpitEvent, CockpitEventType } from "./cockpitEventBus";

export type FrameArbitrationResult<T> = {
  events: CockpitEvent[];
  decision: T;
};

/**
 * Coalesce les événements du frame et applique une seule résolution.
 * Anti-oscillation : pas de re-résolution intra-frame.
 */
export function mergeFrameEvents(events: CockpitEvent[]): CockpitEvent[] {
  const critical = events.filter((e) => e.coalescable === false);
  const coalescable = coalesceEventsByType(events.filter((e) => e.coalescable !== false));
  return [...critical, ...coalescable].sort((a, b) => a.atMs - b.atMs);
}

export function arbitrateCockpitFrame<T>(
  pendingEvents: CockpitEvent[],
  resolve: (events: CockpitEvent[]) => T
): FrameArbitrationResult<T> {
  const events = mergeFrameEvents(pendingEvents);
  const decision = resolve(events);
  return { events, decision };
}

export function coalesceEventsByType(events: CockpitEvent[]): CockpitEvent[] {
  const byType = new Map<CockpitEventType, CockpitEvent>();
  for (const event of events) {
    byType.set(event.type, event);
  }
  return [...byType.values()].sort((a, b) => a.atMs - b.atMs);
}
