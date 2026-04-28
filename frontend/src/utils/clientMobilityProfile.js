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

function normalizeProfile(profile = {}) {
  return {
    needsWheelchair: Boolean(profile.needsWheelchair),
    needsElectricWheelchair: Boolean(profile.needsElectricWheelchair),
    needsWalkingAid: Boolean(profile.needsWalkingAid),
    needsDoorToDoorAssistance: Boolean(profile.needsDoorToDoorAssistance),
    assistanceLevel: String(profile.assistanceLevel || '').trim(),
    emergencyContact: String(profile.emergencyContact || '').trim(),
    notes: String(profile.notes || '').trim(),
  };
}

export function getDefaultMobilityProfile() {
  return { ...DEFAULT_PROFILE };
}

export function getLastMobilityProfile() {
  const store = safeReadStore();
  return store.last ? normalizeProfile(store.last) : getDefaultMobilityProfile();
}

export function saveLastMobilityProfile(profile) {
  const store = safeReadStore();
  const normalized = normalizeProfile(profile);
  store.last = normalized;
  safeWriteStore(store);
  return normalized;
}

export function saveMobilityProfileForEmail(email, profile) {
  const normalizedEmail = String(email || '').trim().toLowerCase();
  const normalized = normalizeProfile(profile);
  const store = safeReadStore();
  if (normalizedEmail) {
    store.byEmail[normalizedEmail] = normalized;
  }
  store.last = normalized;
  safeWriteStore(store);
  return normalized;
}

export function linkMobilityProfileToUser({ publicId, email }) {
  const normalizedPublicId = String(publicId || '').trim();
  const normalizedEmail = String(email || '').trim().toLowerCase();
  if (!normalizedPublicId) return null;

  const store = safeReadStore();
  const fromEmail =
    (normalizedEmail && store.byEmail[normalizedEmail]) ||
    store.byPublicId[normalizedPublicId] ||
    store.last;
  if (!fromEmail) return null;

  const normalized = normalizeProfile(fromEmail);
  store.byPublicId[normalizedPublicId] = normalized;
  store.last = normalized;
  safeWriteStore(store);
  return normalized;
}

export function getMobilityProfileForUser({ publicId, email } = {}) {
  const normalizedPublicId = String(publicId || '').trim();
  const normalizedEmail = String(email || '').trim().toLowerCase();
  const store = safeReadStore();
  const profile =
    (normalizedPublicId && store.byPublicId[normalizedPublicId]) ||
    (normalizedEmail && store.byEmail[normalizedEmail]) ||
    store.last;
  return profile ? normalizeProfile(profile) : getDefaultMobilityProfile();
}
