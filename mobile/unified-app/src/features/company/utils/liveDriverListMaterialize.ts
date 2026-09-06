import {
  applyLocalLocationFreshness,
  lastSeenAtOf,
  resolveFunctionalLocationStatus,
} from "./localDriverLocationFreshness";

export type LiveDriverFreshnessSource = {
  driver_id: number;
  is_active?: boolean | null;
  recorded_at?: string | null;
  timestamp?: string | null;
  last_seen_seconds?: number | null;
  location_status?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  tracking_display_status?: string | null;
  position_source?: string | null;
};

export type AppliedLiveDriverEntry<T extends LiveDriverFreshnessSource> = {
  source: T;
  applied: T;
};

export function isSameLiveDriverAnchor(
  left: LiveDriverFreshnessSource,
  right: LiveDriverFreshnessSource
): boolean {
  return (
    lastSeenAtOf(left) === lastSeenAtOf(right) &&
    left.latitude === right.latitude &&
    left.longitude === right.longitude
  );
}

/**
 * Recalcule l’état fonctionnel sans recréer les objets immobiles.
 * Un GPS / last_seen_at nouveau rematérialise ce chauffeur.
 * Le tick 5 s ne remplace un objet qu’au franchissement live → recent → stale.
 */
export function rematerializeLiveDrivers<T extends LiveDriverFreshnessSource>(args: {
  sources: readonly T[];
  previousById: ReadonlyMap<number, AppliedLiveDriverEntry<T>>;
  nowMs: number;
  refreshAgeForUnchangedSources: boolean;
}): {
  drivers: T[];
  nextById: Map<number, AppliedLiveDriverEntry<T>>;
  reused: number;
  replaced: number;
} {
  const nextById = new Map<number, AppliedLiveDriverEntry<T>>();
  const drivers: T[] = [];
  let reused = 0;
  let replaced = 0;

  for (const source of args.sources) {
    if (source.is_active === false) continue;

    const prev = args.previousById.get(source.driver_id);
    if (prev && prev.source === source && !args.refreshAgeForUnchangedSources) {
      nextById.set(source.driver_id, prev);
      drivers.push(prev.applied);
      reused += 1;
      continue;
    }

    const nextStatus = resolveFunctionalLocationStatus(source, args.nowMs);
    const sameAnchor = prev != null && isSameLiveDriverAnchor(prev.source, source);

    if (prev && sameAnchor && prev.applied.location_status === nextStatus) {
      const entry = { source, applied: prev.applied };
      nextById.set(source.driver_id, entry);
      drivers.push(prev.applied);
      reused += 1;
      continue;
    }

    const applied =
      prev && sameAnchor
        ? applyLocalLocationFreshness(prev.applied, args.nowMs)
        : applyLocalLocationFreshness(source, args.nowMs);

    if (applied === prev?.applied) {
      reused += 1;
    } else {
      replaced += 1;
    }
    nextById.set(source.driver_id, { source, applied });
    drivers.push(applied);
  }

  drivers.sort((a, b) => a.driver_id - b.driver_id);
  return { drivers, nextById, reused, replaced };
}

export function reuseLiveDriverListIfUnchanged<T>(previous: T[], next: T[]): T[] {
  if (
    previous.length === next.length &&
    previous.every((driver, index) => driver === next[index])
  ) {
    return previous;
  }
  return next;
}
