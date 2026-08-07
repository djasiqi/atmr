/**
 * ensureUsableAccessToken / ensureWebAuthReady : cookie-only web.
 */

const mockGetAccessToken = jest.fn();
const mockHasActiveSession = jest.fn();
const mockRefreshSessionTokens = jest.fn();
const mockExpireCurrentWebSession = jest.fn();
const mockIsTerminalRefreshFailure = jest.fn((error) => {
  const status = error?.response?.status;
  return status === 401 || status === 400 || status === 403;
});

jest.mock('../../hooks/useAuthToken', () => ({
  getAccessToken: (...args) => mockGetAccessToken(...args),
}));

jest.mock('../webAuthSession', () => ({
  hasActiveSession: (...args) => mockHasActiveSession(...args),
}));

jest.mock('../apiClient', () => ({
  refreshSessionTokens: (...args) => mockRefreshSessionTokens(...args),
  expireCurrentWebSession: (...args) => mockExpireCurrentWebSession(...args),
  isTerminalRefreshFailure: (...args) => mockIsTerminalRefreshFailure(...args),
}));

const {
  ensureUsableAccessToken,
  ensureWebAuthReady,
  resetEnsureUsableAccessTokenForTests,
} = require('../ensureUsableAccessToken');

describe('ensureUsableAccessToken / ensureWebAuthReady', () => {
  beforeEach(() => {
    resetEnsureUsableAccessTokenForTests();
    mockGetAccessToken.mockReset();
    mockHasActiveSession.mockReset();
    mockRefreshSessionTokens.mockReset();
    mockExpireCurrentWebSession.mockReset();
  });

  it('retourne le token existant sans refresh', async () => {
    mockGetAccessToken.mockReturnValue('tok-ok');
    await expect(ensureUsableAccessToken()).resolves.toBe('tok-ok');
    expect(mockRefreshSessionTokens).not.toHaveBeenCalled();
  });

  it('sans session UI → null / false', async () => {
    mockGetAccessToken.mockReturnValue(null);
    mockHasActiveSession.mockReturnValue(false);
    await expect(ensureUsableAccessToken()).resolves.toBeNull();
    await expect(ensureWebAuthReady()).resolves.toBe(false);
    expect(mockRefreshSessionTokens).not.toHaveBeenCalled();
  });

  it('cookie-only : refresh 200 sans JWT JS → ensureWebAuthReady true', async () => {
    mockGetAccessToken.mockReturnValue(null);
    mockHasActiveSession.mockReturnValue(true);
    mockRefreshSessionTokens.mockResolvedValue(true);
    await expect(ensureWebAuthReady()).resolves.toBe(true);
    await expect(ensureUsableAccessToken()).resolves.toBeNull();
    expect(mockRefreshSessionTokens).toHaveBeenCalled();
    expect(mockExpireCurrentWebSession).not.toHaveBeenCalled();
  });

  it('refresh 401 terminal → expire session', async () => {
    mockGetAccessToken.mockReturnValue(null);
    mockHasActiveSession.mockReturnValue(true);
    mockRefreshSessionTokens.mockRejectedValue({ response: { status: 401 } });
    await expect(ensureWebAuthReady()).resolves.toBe(false);
    expect(mockExpireCurrentWebSession).toHaveBeenCalled();
  });

  it('refresh 503 → pas de logout', async () => {
    mockGetAccessToken.mockReturnValue(null);
    mockHasActiveSession.mockReturnValue(true);
    mockRefreshSessionTokens.mockRejectedValue({ response: { status: 503 } });
    mockIsTerminalRefreshFailure.mockReturnValue(false);
    await expect(ensureWebAuthReady()).resolves.toBe(false);
    expect(mockExpireCurrentWebSession).not.toHaveBeenCalled();
  });
});
