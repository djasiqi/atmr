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
  const rawCompanyUser = safeGet('company_user');
  if (!rawCompanyUser) return null;
  try {
    return JSON.parse(rawCompanyUser);
  } catch (_) {
    return null;
  }
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

const resolveRoleScope = (role) => {
  const normalized = String(role || '').trim().toLowerCase();
  if (normalized === 'company' || normalized === 'admin') return 'company';
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
  const resolvedRole = resolveRoleScope(role ?? user?.role);
  const userPayload =
    user && resolvedRole ? { ...user, role: resolvedRole } : user ? { ...user } : null;

  setEnvUser(userPayload, normalizedEnv, { mirrorLegacy: true });
  setEnvPublicId(userPayload?.public_id ?? null, normalizedEnv, { mirrorLegacy: true });
  setEnvAccessToken(accessToken, normalizedEnv, { mirrorLegacy: true });
  setEnvRefreshToken(refreshToken, normalizedEnv, { mirrorLegacy: true });

  ['company', 'driver', 'institution']
    .filter((scope) => scope !== resolvedRole)
    .forEach((scope) => clearRoleScopedSession(scope));

  if (resolvedRole) {
    setEnvUser(userPayload, resolvedRole, { mirrorLegacy: false });
    setEnvPublicId(userPayload?.public_id ?? null, resolvedRole, { mirrorLegacy: false });
    setEnvAccessToken(accessToken, resolvedRole, { mirrorLegacy: false });
    setEnvRefreshToken(refreshToken, resolvedRole, { mirrorLegacy: false });
  }

  return { env: normalizedEnv, roleScope: resolvedRole, user: userPayload };
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

