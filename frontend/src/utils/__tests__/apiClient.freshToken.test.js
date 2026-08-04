/**
 * P1a — séparation TOKEN_NOT_FRESH vs SESSION_EXPIRED (intercepteur apiClient).
 */

jest.mock('../deferredSessionLogout', () => ({
  notifySessionReauthRequired: jest.fn(),
  stopSessionIdleGuard: jest.fn(),
}));

jest.mock('../sessionKeepAlive', () => ({
  tryRefreshSessionIfNeeded: jest.fn(() => Promise.resolve({ status: 'skipped' })),
  suspendSessionKeepAlive: jest.fn(),
}));

const {
  default: apiClient,
  AUTH_TOKEN_NOT_FRESH,
} = require('../apiClient');
const { notifySessionReauthRequired } = require('../deferredSessionLogout');
const { tryRefreshSessionIfNeeded } = require('../sessionKeepAlive');

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

  it('401 token non fresh → reject AUTH_TOKEN_NOT_FRESH sans logout idle', async () => {
    tryRefreshSessionIfNeeded.mockClear();
    await expect(
      apiClient.get('/companies/me', {
        adapter: fresh401Adapter,
      })
    ).rejects.toMatchObject({
      code: AUTH_TOKEN_NOT_FRESH,
      isFreshTokenRequired: true,
    });

    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(tryRefreshSessionIfNeeded).not.toHaveBeenCalled();
    expect(notifySessionReauthRequired).not.toHaveBeenCalled();
  });
});
