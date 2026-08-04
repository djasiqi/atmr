/**
 * Keep-alive : gap succès, backoff échec, skip pendant warning, résultats structurés.
 */

jest.mock('../apiClient', () => ({
  refreshSessionTokens: jest.fn(),
}));

const mockIsSessionIdleWarningActive = jest.fn(() => false);

jest.mock('../deferredSessionLogout', () => ({
  isSessionIdleWarningActive: (...args) => mockIsSessionIdleWarningActive(...args),
}));

jest.mock('../sessionLogoutState', () => ({
  isExplicitLogoutInProgress: jest.fn(() => false),
  isLoginSessionInProgress: jest.fn(() => false),
}));

const { refreshSessionTokens } = require('../apiClient');
const { isExplicitLogoutInProgress, isLoginSessionInProgress } = require('../sessionLogoutState');
const {
  tryRefreshSessionIfNeeded,
  resetSessionKeepAliveForTests,
  resumeSessionKeepAlive,
  classifyRefreshFailure,
  REFRESH_FAILURE_BACKOFF_MS,
} = require('../sessionKeepAlive');
const { recordUserActivity, resetUserActivityTrackerForTests } = require('../userActivityTracker');

describe('sessionKeepAlive refresh results', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    resetSessionKeepAliveForTests();
    resetUserActivityTrackerForTests();
    localStorage.setItem('app_user', JSON.stringify({ public_id: 'u1', role: 'company' }));
    localStorage.setItem('lirie_auth_env', 'app');
    resumeSessionKeepAlive();
    recordUserActivity();
    refreshSessionTokens.mockReset();
    mockIsSessionIdleWarningActive.mockReturnValue(false);
    isExplicitLogoutInProgress.mockReturnValue(false);
    isLoginSessionInProgress.mockReturnValue(false);
  });

  afterEach(() => {
    resetSessionKeepAliveForTests();
    jest.useRealTimers();
    localStorage.clear();
  });

  it('classifie réseau / 5xx / 429 en transient et 401 en terminal', () => {
    expect(classifyRefreshFailure({ message: 'Network Error' })).toBe('transient_failure');
    expect(classifyRefreshFailure({ response: { status: 502 } })).toBe('transient_failure');
    expect(classifyRefreshFailure({ response: { status: 429 } })).toBe('transient_failure');
    expect(classifyRefreshFailure({ response: { status: 401 } })).toBe('terminal_failure');
  });

  it('retourne refreshed après succès', async () => {
    refreshSessionTokens.mockResolvedValue(true);
    const result = await tryRefreshSessionIfNeeded({ force: true });
    expect(result).toEqual({ status: 'refreshed' });
  });

  it('retourne terminal_failure sur 401', async () => {
    refreshSessionTokens.mockRejectedValue({ response: { status: 401 } });
    const result = await tryRefreshSessionIfNeeded({ force: true });
    expect(result.status).toBe('terminal_failure');
  });

  it('retourne transient_failure sur erreur réseau', async () => {
    refreshSessionTokens.mockRejectedValue({ message: 'Network Error' });
    const result = await tryRefreshSessionIfNeeded({ force: true });
    expect(result.status).toBe('transient_failure');
  });

  it('autorise une nouvelle tentative après backoff court suite à échec', async () => {
    refreshSessionTokens.mockRejectedValueOnce({ message: 'Network Error' });
    await tryRefreshSessionIfNeeded({ force: true });

    refreshSessionTokens.mockResolvedValueOnce(true);
    const blocked = await tryRefreshSessionIfNeeded();
    expect(blocked.status).toBe('skipped');
    expect(refreshSessionTokens).toHaveBeenCalledTimes(1);

    jest.advanceTimersByTime(REFRESH_FAILURE_BACKOFF_MS + 1000);
    recordUserActivity();
    const result = await tryRefreshSessionIfNeeded({ force: true });
    expect(result.status).toBe('refreshed');
    expect(refreshSessionTokens).toHaveBeenCalledTimes(2);
  });

  it('skip le refresh automatique si warning idle actif', async () => {
    mockIsSessionIdleWarningActive.mockReturnValue(true);
    refreshSessionTokens.mockResolvedValue(true);
    const result = await tryRefreshSessionIfNeeded();
    expect(result.status).toBe('skipped');
    expect(refreshSessionTokens).not.toHaveBeenCalled();
  });

  it('autorise force:true même si warning idle actif', async () => {
    mockIsSessionIdleWarningActive.mockReturnValue(true);
    refreshSessionTokens.mockResolvedValue(true);
    const result = await tryRefreshSessionIfNeeded({ force: true });
    expect(result.status).toBe('refreshed');
    expect(refreshSessionTokens).toHaveBeenCalled();
  });
});
