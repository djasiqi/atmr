/**
 * Keep-alive : gap succès, backoff échec, skip pendant warning, résultats structurés.
 */

jest.mock('../apiClient', () => ({
  refreshSessionTokens: jest.fn(),
  expireCurrentWebSession: jest.fn(),
}));

jest.mock('../../services/companySocket', () => ({
  getCompanySocketStatusSnapshot: jest.fn(() => ({ connected: true })),
  retryCompanySocket: jest.fn(),
}));

const mockIsSessionIdleWarningActive = jest.fn(() => false);

jest.mock('../deferredSessionLogout', () => ({
  isSessionIdleWarningActive: (...args) => mockIsSessionIdleWarningActive(...args),
  ensureIdleGuardEvaluated: jest.fn(),
}));

jest.mock('../sessionLogoutState', () => ({
  isExplicitLogoutInProgress: jest.fn(() => false),
  isLoginSessionInProgress: jest.fn(() => false),
}));

const { refreshSessionTokens, expireCurrentWebSession } = require('../apiClient');
const { isExplicitLogoutInProgress, isLoginSessionInProgress } = require('../sessionLogoutState');
const {
  tryRefreshSessionIfNeeded,
  resetSessionKeepAliveForTests,
  resumeSessionKeepAlive,
  startSessionKeepAlive,
  scheduleRefreshFromExp,
  cancelScheduledRefresh,
  classifyRefreshFailure,
  noteAuthTokensRenewed,
  ACCESS_REFRESH_AHEAD_MS,
  CLOCK_SKEW_MS,
  REFRESH_FAILURE_BACKOFF_MS,
} = require('../sessionKeepAlive');
const {
  noteAccessExpiryFromResponse,
  clearStoredAccessExpiry,
} = require('../accessExpiry');
const {
  recordUserActivity,
  resetUserActivityTrackerForTests,
  getLastUserActivityAt,
} = require('../userActivityTracker');

describe('sessionKeepAlive refresh results', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    resetSessionKeepAliveForTests();
    resetUserActivityTrackerForTests();
    clearStoredAccessExpiry();
    localStorage.setItem('app_user', JSON.stringify({ public_id: 'u1', role: 'company' }));
    localStorage.setItem('lirie_auth_env', 'app');
    resumeSessionKeepAlive();
    recordUserActivity();
    refreshSessionTokens.mockReset();
    expireCurrentWebSession.mockReset();
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
    const result = await tryRefreshSessionIfNeeded({ force: true, reason: 'access_expiring' });
    expect(result).toEqual({ status: 'refreshed' });
  });

  it('retourne terminal_failure sur 401 et expire la session', async () => {
    refreshSessionTokens.mockRejectedValue({ response: { status: 401 } });
    const result = await tryRefreshSessionIfNeeded({ force: true, reason: 'access_expired' });
    expect(result.status).toBe('terminal_failure');
    expect(expireCurrentWebSession).toHaveBeenCalled();
  });

  it('retourne transient_failure sur erreur réseau sans logout', async () => {
    refreshSessionTokens.mockRejectedValue({ message: 'Network Error' });
    const result = await tryRefreshSessionIfNeeded({ force: true, reason: 'access_expiring' });
    expect(result.status).toBe('transient_failure');
    expect(expireCurrentWebSession).not.toHaveBeenCalled();
  });

  it('503 refresh → aucune déconnexion', async () => {
    refreshSessionTokens.mockRejectedValue({ response: { status: 503 } });
    const result = await tryRefreshSessionIfNeeded({ force: true, reason: 'access_expiring' });
    expect(result.status).toBe('transient_failure');
    expect(expireCurrentWebSession).not.toHaveBeenCalled();
  });

  it('429 refresh → aucune déconnexion', async () => {
    refreshSessionTokens.mockRejectedValue({ response: { status: 429 } });
    const result = await tryRefreshSessionIfNeeded({ force: true, reason: 'access_expiring' });
    expect(result.status).toBe('transient_failure');
    expect(expireCurrentWebSession).not.toHaveBeenCalled();
  });

  it('autorise une nouvelle tentative après backoff court suite à échec', async () => {
    refreshSessionTokens.mockRejectedValueOnce({ message: 'Network Error' });
    await tryRefreshSessionIfNeeded({ force: true, reason: 'interval_fallback' });

    refreshSessionTokens.mockResolvedValueOnce(true);
    const blocked = await tryRefreshSessionIfNeeded();
    expect(blocked.status).toBe('skipped');
    expect(refreshSessionTokens).toHaveBeenCalledTimes(1);

    jest.advanceTimersByTime(REFRESH_FAILURE_BACKOFF_MS + 1000);
    recordUserActivity();
    const result = await tryRefreshSessionIfNeeded({ force: true, reason: 'access_expiring' });
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

  it('idle warning : force access_expiring ne prolonge pas ; idle_stay_connected oui', async () => {
    mockIsSessionIdleWarningActive.mockReturnValue(true);
    refreshSessionTokens.mockResolvedValue(true);
    const blocked = await tryRefreshSessionIfNeeded({
      force: true,
      reason: 'access_expiring',
    });
    expect(blocked.status).toBe('skipped');

    const allowed = await tryRefreshSessionIfNeeded({
      force: true,
      reason: 'idle_stay_connected',
    });
    expect(allowed.status).toBe('refreshed');
  });

  it('login 08:00 → refresh prévu ~08:55 via access_expires_at', () => {
    const loginAt = new Date('2026-08-07T08:00:00.000Z').getTime();
    jest.setSystemTime(loginAt);
    noteAccessExpiryFromResponse({ access_expires_in: 3600 }, loginAt);
    noteAuthTokensRenewed();
    const spy = jest.spyOn(global, 'setTimeout');
    scheduleRefreshFromExp();
    const expectedDelay = 3600 * 1000 - ACCESS_REFRESH_AHEAD_MS - CLOCK_SKEW_MS;
    expect(spy).toHaveBeenCalledWith(expect.any(Function), expectedDelay);
    spy.mockRestore();
    cancelScheduledRefresh();
  });

  it('reload à 08:50 ne décale pas le refresh vers 09:35 (P0-A)', () => {
    const loginAt = new Date('2026-08-07T08:00:00.000Z').getTime();
    jest.setSystemTime(loginAt);
    noteAccessExpiryFromResponse({ access_expires_in: 3600 }, loginAt);

    jest.setSystemTime(new Date('2026-08-07T08:50:00.000Z').getTime());
    const spy = jest.spyOn(global, 'setTimeout');
    // startSessionKeepAlive ne doit PAS poser lastRefreshSuccessAt = now
    const stop = startSessionKeepAlive();
    const expectedDelay =
      new Date('2026-08-07T09:00:00.000Z').getTime() -
      Date.now() -
      ACCESS_REFRESH_AHEAD_MS -
      CLOCK_SKEW_MS;
    expect(spy).toHaveBeenCalledWith(expect.any(Function), expectedDelay);
    expect(expectedDelay).toBeLessThan(10 * 60 * 1000);
    stop();
    spy.mockRestore();
  });

  it('idle 8h : force visibility_resume skip sans prolonger', async () => {
    refreshSessionTokens.mockResolvedValue(true);
    const eightHoursAgo = Date.now() - 8 * 60 * 60 * 1000 - 1000;
    jest.setSystemTime(eightHoursAgo);
    resetUserActivityTrackerForTests();
    jest.setSystemTime(eightHoursAgo + 8 * 60 * 60 * 1000 + 1000);
    const result = await tryRefreshSessionIfNeeded({
      force: true,
      reason: 'visibility_resume',
    });
    expect(result.status).toBe('skipped');
    expect(refreshSessionTokens).not.toHaveBeenCalled();
    expect(getLastUserActivityAt()).toBe(eightHoursAgo);
  });
});
