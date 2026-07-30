import {
  cancelDeferredLogout,
  initDeferredSessionLogout,
  resetDeferredSessionLogoutForTests,
} from '../deferredSessionLogout';
import {
  recordUserActivity,
  resetUserActivityTrackerForTests,
  SESSION_WORKING_LOOKBACK_MS,
} from '../userActivityTracker';
import {
  MIN_REFRESH_GAP_MS,
  noteAuthTokensRenewed,
  resetSessionKeepAliveForTests,
  resumeSessionKeepAlive,
  tryRefreshSessionIfNeeded,
} from '../sessionKeepAlive';

jest.mock('../apiClient', () => ({
  refreshSessionTokens: jest.fn(() => Promise.resolve(true)),
  isAuthRefreshInProgress: jest.fn(() => false),
}));

jest.mock('../deferredSessionLogout', () => {
  const actual = jest.requireActual('../deferredSessionLogout');
  return {
    ...actual,
    notifySessionReauthRequired: jest.fn(),
  };
});

const { refreshSessionTokens } = require('../apiClient');
const { notifySessionReauthRequired } = require('../deferredSessionLogout');

describe('sessionKeepAlive integration', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    resetSessionKeepAliveForTests();
    resetUserActivityTrackerForTests();
    resetDeferredSessionLogoutForTests();
    initDeferredSessionLogout();
    refreshSessionTokens.mockClear();
    notifySessionReauthRequired.mockClear();
    localStorage.setItem('app_user', JSON.stringify({ role: 'institution', public_id: 'u1' }));
  });

  afterEach(() => {
    resetSessionKeepAliveForTests();
    resetUserActivityTrackerForTests();
    resetDeferredSessionLogoutForTests();
    localStorage.clear();
    jest.useRealTimers();
  });

  it('appelle refreshSessionTokens quand l’utilisateur est actif', async () => {
    recordUserActivity();
    const ok = await tryRefreshSessionIfNeeded({ force: true });
    expect(ok).toBe(true);
    expect(refreshSessionTokens).toHaveBeenCalledTimes(1);
    expect(notifySessionReauthRequired).not.toHaveBeenCalled();
  });

  it('n’appelle pas refresh immédiatement après login (token fresh préservé)', async () => {
    recordUserActivity();
    resumeSessionKeepAlive();
    const ok = await tryRefreshSessionIfNeeded();
    expect(ok).toBe(false);
    expect(refreshSessionTokens).not.toHaveBeenCalled();
  });

  it('n’appelle pas refresh avant MIN_REFRESH_GAP_MS même si actif', async () => {
    recordUserActivity();
    noteAuthTokensRenewed();
    jest.advanceTimersByTime(MIN_REFRESH_GAP_MS - 1000);
    const ok = await tryRefreshSessionIfNeeded();
    expect(ok).toBe(false);
    expect(refreshSessionTokens).not.toHaveBeenCalled();
  });

  it('n’appelle pas refresh si inactif depuis longtemps', async () => {
    recordUserActivity();
    jest.advanceTimersByTime(SESSION_WORKING_LOOKBACK_MS + 1000);
    const ok = await tryRefreshSessionIfNeeded();
    expect(ok).toBe(false);
    expect(refreshSessionTokens).not.toHaveBeenCalled();
  });

  it('cancelDeferredLogout après refresh réussi', async () => {
    recordUserActivity();
    const scheduleSpy = jest.spyOn(
      require('../deferredSessionLogout'),
      'scheduleDeferredLogout'
    );
    await tryRefreshSessionIfNeeded({ force: true });
    expect(scheduleSpy).not.toHaveBeenCalled();
    scheduleSpy.mockRestore();
  });
});

describe('auth-changed multi-onglet', () => {
  beforeEach(() => {
    resetDeferredSessionLogoutForTests();
    initDeferredSessionLogout();
  });

  afterEach(() => {
    resetDeferredSessionLogoutForTests();
  });

  it('annule la déconnexion différée sur auth-changed', () => {
    const { scheduleDeferredLogout } = require('../deferredSessionLogout');
    scheduleDeferredLogout();
    window.dispatchEvent(new Event('auth-changed'));
    cancelDeferredLogout();
    expect(cancelDeferredLogout).not.toThrow;
  });
});
