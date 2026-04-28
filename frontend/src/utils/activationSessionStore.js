const ACTIVATION_STORE_KEY = 'lirie_pending_activation_sessions';

const readStore = () => {
  try {
    const raw = localStorage.getItem(ACTIVATION_STORE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
};

const writeStore = (store) => {
  localStorage.setItem(ACTIVATION_STORE_KEY, JSON.stringify(store));
};

const normalizeEmail = (email) => String(email || '').trim().toLowerCase();

export const setPendingActivationSession = ({
  email,
  activation_session_id,
  masked_email,
  masked_phone,
}) => {
  const normalizedEmail = normalizeEmail(email);
  if (!normalizedEmail || !activation_session_id) return;
  const store = readStore();
  store[normalizedEmail] = {
    activation_session_id,
    masked_email: masked_email || null,
    masked_phone: masked_phone || null,
    updated_at: new Date().toISOString(),
  };
  writeStore(store);
};

export const getPendingActivationByEmail = (email) => {
  const normalizedEmail = normalizeEmail(email);
  if (!normalizedEmail) return null;
  const store = readStore();
  return store[normalizedEmail] || null;
};

export const removePendingActivationByEmail = (email) => {
  const normalizedEmail = normalizeEmail(email);
  if (!normalizedEmail) return;
  const store = readStore();
  if (!store[normalizedEmail]) return;
  delete store[normalizedEmail];
  writeStore(store);
};

export const removePendingActivationBySessionId = (activationSessionId) => {
  if (!activationSessionId) return;
  const store = readStore();
  const nextStore = { ...store };
  let changed = false;
  Object.keys(nextStore).forEach((emailKey) => {
    if (nextStore[emailKey]?.activation_session_id === activationSessionId) {
      delete nextStore[emailKey];
      changed = true;
    }
  });
  if (changed) {
    writeStore(nextStore);
  }
};
