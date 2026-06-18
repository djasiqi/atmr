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
    localStorage.setItem('lirie_auth_env', 'app');
    localStorage.setItem(
      'app_user',
      JSON.stringify({ role: 'company', public_id: 'u-test' })
    );
  });

  afterEach(() => {
    localStorage.clear();
  });

  it('401 token non fresh → déclenche une déconnexion différée', async () => {
    await expect(
      apiClient.get('/companies/me', {
        adapter: fresh401Adapter,
      })
    ).rejects.toMatchObject({
      code: AUTH_TOKEN_NOT_FRESH,
      isFreshTokenRequired: true,
    });

    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(notifySessionReauthRequired).toHaveBeenCalled();
  });
});

/**
 * Échec refresh + inactivité → deferred logout :
 * voir deferredSessionLogout.test.js et sessionKeepAlive.integration.test.js
 */
