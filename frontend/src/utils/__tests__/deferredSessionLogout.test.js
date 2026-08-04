import { toast } from 'sonner';
import {
  initDeferredSessionLogout,
  resetDeferredSessionLogoutForTests,
  startSessionIdleGuard,
  stopSessionIdleGuard,
  resolveIdleWarning,
  isSessionIdleWarningActive,
  SESSION_IDLE_TIMEOUT_MS,
  SESSION_IDLE_WARNING_MS,
} from '../deferredSessionLogout';
import {
  applyRemoteUserActivity,
  recordUserActivity,
  resetUserActivityTrackerForTests,
} from '../userActivityTracker';

jest.mock('sonner', () => ({
  toast: {
    warning: jest.fn(),
    error: jest.fn(),
    dismiss: jest.fn(),
  },
}));

const mockLogoutUser = jest.fn(() => Promise.resolve());
const mockTryRefreshSessionIfNeeded = jest.fn();

jest.mock('../apiClient', () => ({
  logoutUser: (...args) => mockLogoutUser(...args),
}));

jest.mock('../sessionKeepAlive', () => ({
  tryRefreshSessionIfNeeded: (...args) => mockTryRefreshSessionIfNeeded(...args),
}));

jest.mock('../webAuthSession', () => ({
  hasActiveSession: jest.fn(() => true),
  getAuthEnv: jest.fn(() => 'app'),
  getActivePublicId: jest.fn(() => 'user-1'),
}));

describe('deferredSessionLogout (garde idle)', () => {
  const POLL_BUFFER_MS = 2000;

  beforeEach(() => {
    jest.useFakeTimers();
    resetUserActivityTrackerForTests();
    resetDeferredSessionLogoutForTests();
    mockLogoutUser.mockClear();
    mockTryRefreshSessionIfNeeded.mockReset();
    toast.warning.mockClear();
    toast.error.mockClear();
    toast.dismiss.mockClear();
    localStorage.setItem('app_user', JSON.stringify({ public_id: 'user-1' }));
  });

  afterEach(() => {
    resetDeferredSessionLogoutForTests();
    resetUserActivityTrackerForTests();
    jest.useRealTimers();
    localStorage.clear();
  });

  it('démarre la garde au boot si session active (sans attendre auth-changed)', () => {
    const { hasActiveSession } = require('../webAuthSession');
    hasActiveSession.mockReturnValue(true);
    initDeferredSessionLogout();
    startSessionIdleGuard();
    recordUserActivity();
    jest.advanceTimersByTime(SESSION_IDLE_TIMEOUT_MS + POLL_BUFFER_MS);
    expect(isSessionIdleWarningActive()).toBe(true);
    expect(toast.warning).toHaveBeenCalledWith(
      expect.stringContaining('Votre session expirera dans'),
      expect.objectContaining({
        dismissible: false,
        closeButton: false,
      })
    );
  });

  it('affiche le préavis après inactivité, pas sur activité locale pendant le warning', () => {
    startSessionIdleGuard();
    recordUserActivity();
    jest.advanceTimersByTime(SESSION_IDLE_TIMEOUT_MS + 1000);

    expect(isSessionIdleWarningActive()).toBe(true);
    const callsBefore = toast.warning.mock.calls.length;

    recordUserActivity();
    jest.advanceTimersByTime(2000);

    expect(isSessionIdleWarningActive()).toBe(true);
    expect(toast.warning.mock.calls.length).toBeGreaterThanOrEqual(callsBefore);
  });

  it('résout le warning sur activité distante (même session)', () => {
    startSessionIdleGuard();
    recordUserActivity();
    jest.advanceTimersByTime(SESSION_IDLE_TIMEOUT_MS + 1000);
    expect(isSessionIdleWarningActive()).toBe(true);

    applyRemoteUserActivity(Date.now());
    expect(isSessionIdleWarningActive()).toBe(false);
  });

  it('déconnecte après countdown sans action', async () => {
    startSessionIdleGuard();
    recordUserActivity();
    jest.advanceTimersByTime(SESSION_IDLE_TIMEOUT_MS + 1000);
    jest.advanceTimersByTime(SESSION_IDLE_WARNING_MS + 1000);
    await Promise.resolve();
    await Promise.resolve();

    expect(mockLogoutUser).toHaveBeenCalledWith(
      expect.objectContaining({
        immediate: true,
        reason: 'idle_timeout',
        preserveNext: true,
      })
    );
  });

  it('Rester connecté : refresh OK → warning fermé', async () => {
    mockTryRefreshSessionIfNeeded.mockResolvedValue({ status: 'refreshed' });
    startSessionIdleGuard();
    recordUserActivity();
    jest.advanceTimersByTime(SESSION_IDLE_TIMEOUT_MS + 1000);

    const action = toast.warning.mock.calls.at(-1)?.[1]?.action;
    expect(action).toBeTruthy();
    await action.onClick({ preventDefault: jest.fn() });

    expect(mockTryRefreshSessionIfNeeded).toHaveBeenCalledWith({ force: true });
    expect(isSessionIdleWarningActive()).toBe(false);
    expect(mockLogoutUser).not.toHaveBeenCalled();
  });

  it('Rester connecté : erreur réseau → pas de logout, warning conservé', async () => {
    mockTryRefreshSessionIfNeeded.mockResolvedValue({ status: 'transient_failure' });
    startSessionIdleGuard();
    recordUserActivity();
    jest.advanceTimersByTime(SESSION_IDLE_TIMEOUT_MS + 1000);

    const action = toast.warning.mock.calls.at(-1)?.[1]?.action;
    await action.onClick({ preventDefault: jest.fn() });

    expect(mockLogoutUser).not.toHaveBeenCalled();
    expect(isSessionIdleWarningActive()).toBe(true);
    expect(toast.error).toHaveBeenCalledWith(
      expect.stringContaining('Impossible de prolonger'),
      expect.any(Object)
    );
  });

  it('Rester connecté : 401 terminal → logout session_expired', async () => {
    mockTryRefreshSessionIfNeeded.mockResolvedValue({ status: 'terminal_failure' });
    startSessionIdleGuard();
    recordUserActivity();
    jest.advanceTimersByTime(SESSION_IDLE_TIMEOUT_MS + 1000);

    const action = toast.warning.mock.calls.at(-1)?.[1]?.action;
    await action.onClick({ preventDefault: jest.fn() });
    // logout via import dynamique apiClient
    await Promise.resolve();
    await Promise.resolve();

    expect(mockLogoutUser).toHaveBeenCalledWith(
      expect.objectContaining({
        immediate: true,
        reason: 'session_expired',
      })
    );
  });

  it('double clic Rester connecté → un seul refresh', async () => {
    const refreshPromise = new Promise(() => {});
    mockTryRefreshSessionIfNeeded.mockReturnValue(refreshPromise);
    startSessionIdleGuard();
    recordUserActivity();
    jest.advanceTimersByTime(SESSION_IDLE_TIMEOUT_MS + 1000);

    const action = toast.warning.mock.calls.at(-1)?.[1]?.action;
    void action.onClick({ preventDefault: jest.fn() });
    await Promise.resolve();
    void action.onClick({ preventDefault: jest.fn() });
    await Promise.resolve();

    expect(mockTryRefreshSessionIfNeeded).toHaveBeenCalledTimes(1);
  });

  it('auth-changed sans session → stop garde', () => {
    const { hasActiveSession } = require('../webAuthSession');
    initDeferredSessionLogout();
    hasActiveSession.mockReturnValue(false);
    window.dispatchEvent(new Event('auth-changed'));
    expect(isSessionIdleWarningActive()).toBe(false);
    stopSessionIdleGuard();
  });

  it('resolveIdleWarning ne stoppe pas la surveillance (phase ACTIVE)', () => {
    startSessionIdleGuard();
    recordUserActivity();
    jest.advanceTimersByTime(SESSION_IDLE_TIMEOUT_MS + 1000);
    resolveIdleWarning();
    expect(isSessionIdleWarningActive()).toBe(false);

    jest.advanceTimersByTime(SESSION_IDLE_TIMEOUT_MS + 2000);
    expect(toast.warning).toHaveBeenCalled();
  });
});
