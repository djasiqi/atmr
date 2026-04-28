const MAX_DEDUP_EVENT_IDS = 2000;

const seenEventIds = new Set();
const lastCanonicalByEntity = new Map();

const parseIsoMs = (value) => {
  if (value == null || value === '') return null;
  const t = Date.parse(String(value));
  return Number.isFinite(t) ? t : null;
};

export const canonicalRealtimeTimeMs = (payload) => {
  if (!payload || typeof payload !== 'object') return null;
  return parseIsoMs(payload.received_at ?? payload.recorded_at ?? payload.timestamp ?? payload.ts);
};

export const getEntityKey = (payload, fallbackPrefix = 'entity') => {
  const id = payload?.driver_id ?? payload?.booking_id ?? payload?.id ?? null;
  if (id == null) return null;
  return `${fallbackPrefix}:${String(id)}`;
};

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
    return false;
  }
  lastCanonicalByEntity.set(entityKey, canonicalTimeMs);
  return true;
};

