/**
 * missing_token → déconnexion immédiate (pas de toast / logout différé).
 */

jest.mock('../deferredSessionLogout', () => ({
  notifySessionReauthRequired: jest.fn(),
  cancelDeferredLogout: jest.fn(),
  stopSessionIdleGuard: jest.fn(),
}));

jest.mock('../sessionKeepAlive', () => ({
  tryRefreshSessionIfNeeded: jest.fn(() => Promise.resolve({ status: 'skipped' })),
  suspendSessionKeepAlive: jest.fn(),
}));

jest.mock('../userActivityTracker', () => ({
  isUserRecentlyActive: jest.fn(() => true),
  SESSION_WORKING_LOOKBACK_MS: 30 * 60 * 1000,
}));

jest.mock('../authNavigation', () => ({
  requestAuthNavigate: jest.fn(),
}));

jest.mock('../../services/companySocket', () => ({
  disconnectCompanySocket: jest.fn(),
}));

const {
  default: apiClient,
  AUTH_SESSION_EXPIRED,
  isMissingTokenErrorPayload,
} = require('../apiClient');
const { notifySessionReauthRequired } = require('../deferredSessionLogout');
const { requestAuthNavigate } = require('../authNavigation');
const { AUTH_LOGOUT_AT_KEY, AUTH_LOGOUT_REASON_KEY } = require('../sessionLogoutState');

const missingTokenBody = {
  error: 'missing_token',
  message: 'Missing JWT in cookies or headers (Missing cookie "access_token")',
};

describe('isMissingTokenErrorPayload', () => {
  it('détecte error=missing_token', () => {
    expect(isMissingTokenErrorPayload({ error: 'missing_token' })).toBe(true);
  });

  it('détecte le message Flask-JWT Missing JWT', () => {
    expect(isMissingTokenErrorPayload(missingTokenBody)).toBe(true);
  });

  it('ignore les autres 401', () => {
    expect(isMissingTokenErrorPayload({ error: 'token_expired' })).toBe(false);
  });
});

describe('apiClient — missing_token', () => {
  let previousAdapter;

  beforeEach(() => {
    notifySessionReauthRequired.mockClear();
    requestAuthNavigate.mockClear();
    localStorage.setItem('lirie_auth_env', 'app');
    localStorage.setItem(
      'app_user',
      JSON.stringify({ role: 'company', public_id: 'u-test' })
    );
    localStorage.setItem('app_access_token', 'stale-token');
    sessionStorage.clear();

    previousAdapter = apiClient.defaults.adapter;
    apiClient.defaults.adapter = (config) => {
      const url = String(config.url || '');
      if (url.includes('/auth/logout')) {
        return Promise.resolve({
          data: { ok: true },
          status: 200,
          statusText: 'OK',
          headers: {},
          config,
        });
      }
      if (url.includes('/auth/refresh-token')) {
        return Promise.reject({
          isAxiosError: true,
          message: 'Request failed with status code 401',
          response: {
            status: 401,
            data: missingTokenBody,
            headers: {},
            config,
          },
          config,
        });
      }
      return Promise.reject({
        isAxiosError: true,
        message: 'Request failed with status code 401',
        response: {
          status: 401,
          data: missingTokenBody,
          headers: {},
          config,
        },
        config,
      });
    };
  });

  afterEach(() => {
    apiClient.defaults.adapter = previousAdapter;
    localStorage.clear();
    sessionStorage.clear();
  });

  it('401 missing_token après échec refresh → session expirée + redirect login', async () => {
    await expect(apiClient.get('/companies/me')).rejects.toMatchObject({
      code: AUTH_SESSION_EXPIRED,
      isSessionExpired: true,
      meta: { suppressAuthError: true },
    });

    await new Promise((resolve) => setTimeout(resolve, 100));

    expect(notifySessionReauthRequired).not.toHaveBeenCalled();
    expect(requestAuthNavigate).toHaveBeenCalled();
    expect(localStorage.getItem('app_access_token')).toBeNull();
    expect(localStorage.getItem(AUTH_LOGOUT_AT_KEY)).toBeTruthy();
    expect(sessionStorage.getItem(AUTH_LOGOUT_REASON_KEY)).toBe('session_expired');
  });
});
