const MAX_DEDUP_EVENT_IDS = 2000;

const seenEventIds = new Set();
const lastCanonicalByEntity = new Map();
/** Limite fuite mémoire sur sessions longues / nombreux chauffeurs. */
const MAX_CANONICAL_ENTITY_KEYS = 4000;

const parseIsoMs = (value) => {
  if (value == null || value === '') return null;
  const t = Date.parse(String(value));
  return Number.isFinite(t) ? t : null;
};

/**
 * Temps « métier » pour le guard : préférer l’instant de l’événement GPS,
 * pas `received_at` (souvent temps de traitement / Kafka — rejets « stale » erronés).
 */
export const canonicalRealtimeTimeMs = (payload) => {
  if (!payload || typeof payload !== 'object') return null;
  return (
    parseIsoMs(payload.recorded_at) ??
    parseIsoMs(payload.timestamp) ??
    parseIsoMs(payload.ts) ??
    parseIsoMs(payload.received_at)
  );
};

export const getEntityKey = (payload, fallbackPrefix = 'entity') => {
  const id = payload?.driver_id ?? payload?.booking_id ?? payload?.id ?? null;
  if (id == null) return null;
  return `${fallbackPrefix}:${String(id)}`;
};

function evictCanonicalMapIfNeeded() {
  while (lastCanonicalByEntity.size > MAX_CANONICAL_ENTITY_KEYS) {
    const first = lastCanonicalByEntity.keys().next().value;
    if (first === undefined) break;
    lastCanonicalByEntity.delete(first);
  }
}

export const shouldAcceptRealtimeEvent = ({ eventId, entityKey, canonicalTimeMs }) => {
  if (eventId) {
    if (seenEventIds.has(eventId)) {
      return false;
    }
    seenEventIds.add(eventId);
    if (seenEventIds.size > MAX_DEDUP_EVENT_IDS) {
      const first = seenEventIds.values().next().value;
      seenEventIds.delete(first);
    }
  }

  if (!entityKey || canonicalTimeMs == null) {
    return true;
  }

  const previous = lastCanonicalByEntity.get(entityKey);
  if (previous != null && canonicalTimeMs < previous) {
    if (typeof console !== 'undefined' && typeof console.debug === 'function') {
      console.debug('[realtimeEventGuard] rejected stale event', {
        entityKey,
        canonicalTimeMs,
        previous,
      });
    }
    return false;
  }
  lastCanonicalByEntity.delete(entityKey);
  lastCanonicalByEntity.set(entityKey, canonicalTimeMs);
  evictCanonicalMapIfNeeded();
  return true;
};
