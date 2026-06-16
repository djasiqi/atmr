/**
 * P1a — séparation TOKEN_NOT_FRESH vs SESSION_EXPIRED (intercepteur apiClient).
 */

jest.mock('../deferredSessionLogout', () => ({
  notifySessionReauthRequired: jest.fn(),
}));

jest.mock('../sessionKeepAlive', () => ({
  tryRefreshSessionIfNeeded: jest.fn(() => Promise.resolve(false)),
}));

jest.mock('../userActivityTracker', () => ({
  isUserRecentlyActive: jest.fn(() => false),
  SESSION_WORKING_LOOKBACK_MS: 30 * 60 * 1000,
}));

const {
  default: apiClient,
  AUTH_TOKEN_NOT_FRESH,
  registerFreshTokenReauthHandler,
} = require('../apiClient');
const { notifySessionReauthRequired } = require('../deferredSessionLogout');

const fresh401Adapter = (config) =>
  Promise.reject({
    isAxiosError: true,
    message: 'Request failed with status code 401',
    response: {
      status: 401,
      data: { error: 'Fresh token required' },
    },
    config,
  });

describe('apiClient — séparation Fresh vs Expired', () => {
  beforeEach(() => {
    notifySessionReauthRequired.mockClear();
    registerFreshTokenReauthHandler(null);
    localStorage.setItem('lirie_auth_env', 'app');
    localStorage.setItem(
      'app_user',
      JSON.stringify({ role: 'company', public_id: 'u-test' })
    );
  });

  afterEach(() => {
    registerFreshTokenReauthHandler(null);
    localStorage.clear();
  });

  it('401 token non fresh → pas notifySessionReauthRequired', async () => {
    await expect(
      apiClient.get('/companies/me', {
        skipAuthRedirect: true,
        adapter: fresh401Adapter,
      })
    ).rejects.toMatchObject({
      code: AUTH_TOKEN_NOT_FRESH,
      isFreshTokenRequired: true,
    });

    expect(notifySessionReauthRequired).not.toHaveBeenCalled();
  });

  it('401 fresh avec handler → retente via retryFn sans deferred logout', async () => {
    let attempts = 0;
    const adapter = (config) => {
      attempts += 1;
      if (attempts === 1) {
        return fresh401Adapter(config);
      }
      return Promise.resolve({
        data: { ok: true },
        status: 200,
        statusText: 'OK',
        headers: {},
        config,
      });
    };

    registerFreshTokenReauthHandler(async ({ retryFn }) => {
      await retryFn();
    });

    const result = await apiClient.get('/companies/me', {
      skipAuthRedirect: true,
      adapter,
    });

    expect(result.data).toEqual({ ok: true });
    expect(notifySessionReauthRequired).not.toHaveBeenCalled();
    expect(attempts).toBeGreaterThanOrEqual(2);
  });
});

/**
 * Échec refresh + inactivité → deferred logout :
 * voir deferredSessionLogout.test.js et sessionKeepAlive.integration.test.js
 */
