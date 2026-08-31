const AUTH_ENV_STORAGE_KEY = 'lirie_auth_env';
const APP_ENV_KEY = 'app';
const DEMO_ENV_KEY = 'demo';

const safeGet = (key) => {
  try {
    return localStorage.getItem(key);
  } catch (_) {
    return null;
  }
};

const safeSet = (key, value) => {
  try {
    localStorage.setItem(key, value);
  } catch (_) {
    // no-op
  }
};

const safeRemove = (key) => {
  try {
    localStorage.removeItem(key);
  } catch (_) {
    // no-op
  }
};

export const getAuthEnv = () => (safeGet(AUTH_ENV_STORAGE_KEY) === DEMO_ENV_KEY ? DEMO_ENV_KEY : APP_ENV_KEY);

export const setAuthEnv = (env) => {
  const normalized = env === DEMO_ENV_KEY ? DEMO_ENV_KEY : APP_ENV_KEY;
  safeSet(AUTH_ENV_STORAGE_KEY, normalized);
  return normalized;
};

export const normalizeAuthRole = (rawRole) => {
  const normalized = String(rawRole || '').trim().toLowerCase();
  if (!normalized) return '';
  if (normalized.startsWith('institution')) return 'institution';
  if (
    normalized === 'admin' ||
    normalized.startsWith('company') ||
    normalized.startsWith('transport_company')
  ) {
    return normalized === 'admin' ? 'admin' : 'company';
  }
  if (normalized.startsWith('driver')) return 'driver';
  return normalized;
};

export const setEnvPublicId = (
  publicId,
  env = getAuthEnv(),
  { mirrorLegacy = true } = {}
) => {
  if (publicId == null || publicId === '') {
    safeRemove(`${env}_public_id`);
    if (mirrorLegacy) safeRemove('public_id');
    return null;
  }
  const normalized = String(publicId);
  safeSet(`${env}_public_id`, normalized);
  if (mirrorLegacy) safeSet('public_id', normalized);
  return normalized;
};

const setEnvAccessToken = (
  token,
  env = getAuthEnv(),
  { mirrorLegacy = true } = {}
) => {
  if (!token) {
    safeRemove(`${env}_access_token`);
    if (mirrorLegacy) safeRemove('authToken');
    return null;
  }
  const normalized = String(token);
  safeSet(`${env}_access_token`, normalized);
  if (mirrorLegacy) safeSet('authToken', normalized);
  return normalized;
};

const setEnvRefreshToken = (
  token,
  env = getAuthEnv(),
  { mirrorLegacy = true } = {}
) => {
  if (!token) {
    safeRemove(`${env}_refresh_token`);
    if (mirrorLegacy) safeRemove('refreshToken');
    return null;
  }
  const normalized = String(token);
  safeSet(`${env}_refresh_token`, normalized);
  if (mirrorLegacy) safeSet('refreshToken', normalized);
  return normalized;
};

export const getEnvAccessToken = (env = getAuthEnv(), { allowLegacy = true } = {}) => {
  const scoped = safeGet(`${env}_access_token`);
  if (scoped) return scoped;
  if (!allowLegacy) return null;
  return safeGet('authToken');
};

export const getActiveAccessToken = ({ allowLegacy = true } = {}) =>
  getEnvAccessToken(getAuthEnv(), { allowLegacy });

export const getEnvRefreshToken = (env = getAuthEnv(), { allowLegacy = true } = {}) => {
  const scoped = safeGet(`${env}_refresh_token`);
  if (scoped) return scoped;
  if (!allowLegacy) return null;
  return safeGet('refreshToken');
};

export const getEnvUser = (env = getAuthEnv()) => {
  const raw = safeGet(`${env}_user`) || safeGet('user');
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch (_) {
    return null;
  }
};

export const setEnvUser = (user, env = getAuthEnv(), { mirrorLegacy = true } = {}) => {
  if (!user) {
    safeRemove(`${env}_user`);
    if (mirrorLegacy) safeRemove('user');
    return null;
  }
  try {
    const serialized = JSON.stringify(user);
    safeSet(`${env}_user`, serialized);
    if (mirrorLegacy) safeSet('user', serialized);
    return user;
  } catch (_) {
    return null;
  }
};

export const getActiveUser = (env = getAuthEnv()) => {
  const scoped = getEnvUser(env);
  if (scoped) return scoped;
  // Ordre : admin dédié avant company (évite qu'un admin lise company_user)
  for (const key of ['admin_user', 'company_user', 'driver_user', 'institution_user']) {
    const raw = safeGet(key);
    if (!raw) continue;
    try {
      return JSON.parse(raw);
    } catch (_) {
      // continue
    }
  }
  return null;
};

export const getActivePublicId = (env = getAuthEnv()) => {
  const user = getActiveUser(env);
  const userPublicId = user?.public_id ?? user?.sub ?? null;
  if (userPublicId != null && userPublicId !== '') return String(userPublicId);
  return safeGet(`${env}_public_id`) || safeGet('public_id');
};

export const getDriverPublicId = (env = getAuthEnv()) =>
  safeGet('driver_public_id') || getActivePublicId(env);

export const getCompanyScopedAccessToken = (env = getAuthEnv()) => {
  if (env === DEMO_ENV_KEY) {
    return (
      safeGet('company_access_token') ||
      safeGet('company_authToken') ||
      getEnvAccessToken(DEMO_ENV_KEY, { allowLegacy: false })
    );
  }
  return (
    safeGet('company_access_token') ||
    safeGet('company_authToken') ||
    getEnvAccessToken(APP_ENV_KEY, { allowLegacy: false })
  );
};

export const clearCompanyScopedTokens = () => {
  try {
    localStorage.removeItem('company_access_token');
    localStorage.removeItem('company_authToken');
    localStorage.removeItem('company_refresh_token');
    localStorage.removeItem('company_refreshToken');
  } catch (_) {
    // no-op
  }
};

const clearRoleScopedSession = (scope) => {
  if (!scope) return;
  safeRemove(`${scope}_user`);
  safeRemove(`${scope}_public_id`);
  safeRemove(`${scope}_access_token`);
  safeRemove(`${scope}_refresh_token`);
  if (scope === 'company') {
    safeRemove('company_authToken');
    safeRemove('company_refreshToken');
  }
  if (scope === 'driver') {
    safeRemove('driver_authToken');
    safeRemove('driver_refreshToken');
  }
};

/**
 * Scope de stockage par rôle.
 * IMPORTANT : admin ≠ company — ne jamais partager company_user avec un admin.
 */
const resolveRoleScope = (role) => {
  const normalized = normalizeAuthRole(role);
  if (normalized === 'admin') return 'admin';
  if (normalized === 'company') return 'company';
  if (normalized === 'driver') return 'driver';
  if (normalized === 'institution') return 'institution';
  return null;
};

export const writeAuthSession = ({
  env = getAuthEnv(),
  user = null,
  role = null,
  accessToken = null,
  refreshToken = null,
} = {}) => {
  const normalizedEnv = env === DEMO_ENV_KEY ? DEMO_ENV_KEY : APP_ENV_KEY;
  setAuthEnv(normalizedEnv);
  const canonicalRole = normalizeAuthRole(role ?? user?.role);
  const resolvedScope = resolveRoleScope(canonicalRole);
  // Conserver le rôle métier réel (admin reste admin — jamais réécrit en company)
  const userPayload = user
    ? { ...user, role: canonicalRole || user.role }
    : null;

  setEnvUser(userPayload, normalizedEnv, { mirrorLegacy: true });
  setEnvPublicId(userPayload?.public_id ?? null, normalizedEnv, { mirrorLegacy: true });
  setEnvAccessToken(accessToken, normalizedEnv, { mirrorLegacy: true });
  setEnvRefreshToken(refreshToken, normalizedEnv, { mirrorLegacy: true });

  ['admin', 'company', 'driver', 'institution']
    .filter((scope) => scope !== resolvedScope)
    .forEach((scope) => clearRoleScopedSession(scope));

  if (resolvedScope) {
    setEnvUser(userPayload, resolvedScope, { mirrorLegacy: false });
    setEnvPublicId(userPayload?.public_id ?? null, resolvedScope, { mirrorLegacy: false });
    setEnvAccessToken(accessToken, resolvedScope, { mirrorLegacy: false });
    setEnvRefreshToken(refreshToken, resolvedScope, { mirrorLegacy: false });
  }

  return { env: normalizedEnv, roleScope: resolvedScope, user: userPayload };
};

export const setDemoRecommendedJourney = (
  journey,
  env = getAuthEnv()
) => {
  const normalized = String(journey || '').trim().toLowerCase() || 'generic';
  safeSet(`${env}_demo_recommended_journey`, normalized);
  safeSet('demo_recommended_journey', normalized);
  return normalized;
};

export const removeLegacyGlobalTokens = () => {
  try {
    localStorage.removeItem('authToken');
    localStorage.removeItem('refreshToken');
  } catch (_) {
    // no-op
  }
};

export const hasCompanyScopedAccessToken = (env = getAuthEnv()) => {
  if (env === DEMO_ENV_KEY) {
    return Boolean(getEnvAccessToken(DEMO_ENV_KEY, { allowLegacy: false }));
  }
  return Boolean(
    safeGet('company_access_token') ||
      safeGet('company_authToken') ||
      getEnvAccessToken(APP_ENV_KEY, { allowLegacy: false })
  );
};

export const hasActiveSession = (env = getAuthEnv()) =>
  Boolean(getEnvAccessToken(env, { allowLegacy: true }) || getEnvUser(env));

/**
 * Session web institution (rôle, storage dédié, ou route dashboard).
 * Utilisé pour la politique idle 15 min (institution) sans toucher company/driver.
 */
export const isInstitutionWebSession = (env = getAuthEnv()) => {
  const users = [getEnvUser(env), getActiveUser(env)];
  if (users.some((user) => normalizeAuthRole(user?.role) === 'institution')) {
    return true;
  }
  if (safeGet('institution_user')) {
    return true;
  }
  try {
    const path = typeof window !== 'undefined' ? window.location?.pathname || '' : '';
    if (path.startsWith('/dashboard/institution/')) {
      return true;
    }
  } catch (_) {
    // ignore
  }
  return false;
};

