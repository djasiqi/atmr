/**
 * ensureUsableAccessToken : refresh si session UI active sans JWT access.
 */

const mockGetAccessToken = jest.fn();
const mockHasActiveSession = jest.fn();
const mockRefreshSessionTokens = jest.fn();

jest.mock('../../hooks/useAuthToken', () => ({
  getAccessToken: (...args) => mockGetAccessToken(...args),
}));

jest.mock('../webAuthSession', () => ({
  hasActiveSession: (...args) => mockHasActiveSession(...args),
}));

jest.mock('../apiClient', () => ({
  refreshSessionTokens: (...args) => mockRefreshSessionTokens(...args),
}));

const {
  ensureUsableAccessToken,
  resetEnsureUsableAccessTokenForTests,
} = require('../ensureUsableAccessToken');

describe('ensureUsableAccessToken', () => {
  beforeEach(() => {
    resetEnsureUsableAccessTokenForTests();
    mockGetAccessToken.mockReset();
    mockHasActiveSession.mockReset();
    mockRefreshSessionTokens.mockReset();
  });

  it('retourne le token existant sans refresh', async () => {
    mockGetAccessToken.mockReturnValue('tok-ok');
    await expect(ensureUsableAccessToken()).resolves.toBe('tok-ok');
    expect(mockRefreshSessionTokens).not.toHaveBeenCalled();
  });

  it('sans session UI → null', async () => {
    mockGetAccessToken.mockReturnValue(null);
    mockHasActiveSession.mockReturnValue(false);
    await expect(ensureUsableAccessToken()).resolves.toBeNull();
    expect(mockRefreshSessionTokens).not.toHaveBeenCalled();
  });

  it('session UI + token absent → refresh puis nouveau token', async () => {
    mockGetAccessToken.mockReturnValueOnce(null).mockReturnValueOnce('tok-fresh');
    mockHasActiveSession.mockReturnValue(true);
    mockRefreshSessionTokens.mockResolvedValue(true);
    await expect(ensureUsableAccessToken()).resolves.toBe('tok-fresh');
    expect(mockRefreshSessionTokens).toHaveBeenCalledTimes(1);
  });

  it('refresh en échec → null', async () => {
    mockGetAccessToken.mockReturnValue(null);
    mockHasActiveSession.mockReturnValue(true);
    mockRefreshSessionTokens.mockRejectedValue(new Error('boom'));
    await expect(ensureUsableAccessToken()).resolves.toBeNull();
  });
});
