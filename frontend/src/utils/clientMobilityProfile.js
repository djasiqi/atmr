const STORAGE_KEY = 'lirie_client_mobility_profile_v1';

const DEFAULT_PROFILE = {
  needsWheelchair: false,
  needsElectricWheelchair: false,
  needsWalkingAid: false,
  needsDoorToDoorAssistance: false,
  assistanceLevel: '',
  emergencyContact: '',
  notes: '',
};

function safeReadStore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { byEmail: {}, byPublicId: {}, last: null };
    const parsed = JSON.parse(raw);
    return {
      byEmail: parsed?.byEmail || {},
      byPublicId: parsed?.byPublicId || {},
      last: parsed?.last || null,
    };
  } catch (_) {
    return { byEmail: {}, byPublicId: {}, last: null };
  }
}

function safeWriteStore(store) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  } catch (_) {
    // no-op
  }
}

function hasLegacySensitiveFields(profile) {
  if (!profile || typeof profile !== 'object') return false;
  return Boolean(profile.emergencyContact) || Boolean(profile.notes);
}

/** Flags mobilité uniquement — jamais de contact d'urgence ni notes libres. */
function toPersistedProfile(profile = {}) {
  return {
    needsWheelchair: Boolean(profile.needsWheelchair),
    needsElectricWheelchair: Boolean(profile.needsElectricWheelchair),
    needsWalkingAid: Boolean(profile.needsWalkingAid),
    needsDoorToDoorAssistance: Boolean(profile.needsDoorToDoorAssistance),
    assistanceLevel: String(profile.assistanceLevel || '').trim(),
  };
}

function sanitizeStore(store) {
  let dirty = false;
  const next = {
    byEmail: {},
    byPublicId: {},
    last: store.last ? toPersistedProfile(store.last) : null,
  };
  if (hasLegacySensitiveFields(store.last)) dirty = true;
  Object.entries(store.byEmail || {}).forEach(([email, profile]) => {
    if (hasLegacySensitiveFields(profile)) dirty = true;
    next.byEmail[email] = toPersistedProfile(profile);
  });
  Object.entries(store.byPublicId || {}).forEach(([publicId, profile]) => {
    if (hasLegacySensitiveFields(profile)) dirty = true;
    next.byPublicId[publicId] = toPersistedProfile(profile);
  });
  if (dirty) {
    safeWriteStore(next);
  }
  return next;
}

function readSanitizedStore() {
  return sanitizeStore(safeReadStore());
}

function normalizeProfile(profile = {}) {
  return {
    ...toPersistedProfile(profile),
    emergencyContact: '',
    notes: '',
  };
}

export function getDefaultMobilityProfile() {
  return { ...DEFAULT_PROFILE };
}

export function getLastMobilityProfile() {
  const store = readSanitizedStore();
  return store.last ? normalizeProfile(store.last) : getDefaultMobilityProfile();
}

export function saveLastMobilityProfile(profile) {
  const store = readSanitizedStore();
  const persisted = toPersistedProfile(profile);
  store.last = persisted;
  safeWriteStore(store);
  return normalizeProfile(persisted);
}

export function saveMobilityProfileForEmail(email, profile) {
  const normalizedEmail = String(email || '').trim().toLowerCase();
  const persisted = toPersistedProfile(profile);
  const store = readSanitizedStore();
  if (normalizedEmail) {
    store.byEmail[normalizedEmail] = persisted;
  }
  store.last = persisted;
  safeWriteStore(store);
  return normalizeProfile(persisted);
}

export function linkMobilityProfileToUser({ publicId, email }) {
  const normalizedPublicId = String(publicId || '').trim();
  const normalizedEmail = String(email || '').trim().toLowerCase();
  if (!normalizedPublicId) return null;

  const store = readSanitizedStore();
  const fromEmail =
    (normalizedEmail && store.byEmail[normalizedEmail]) ||
    store.byPublicId[normalizedPublicId] ||
    store.last;
  if (!fromEmail) return null;

  const persisted = toPersistedProfile(fromEmail);
  store.byPublicId[normalizedPublicId] = persisted;
  store.last = persisted;
  safeWriteStore(store);
  return normalizeProfile(persisted);
}

export function getMobilityProfileForUser({ publicId, email } = {}) {
  const normalizedPublicId = String(publicId || '').trim();
  const normalizedEmail = String(email || '').trim().toLowerCase();
  const store = readSanitizedStore();
  const profile =
    (normalizedPublicId && store.byPublicId[normalizedPublicId]) ||
    (normalizedEmail && store.byEmail[normalizedEmail]) ||
    store.last;
  return profile ? normalizeProfile(profile) : getDefaultMobilityProfile();
}
