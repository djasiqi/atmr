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
    custom: jest.fn(),
    warning: jest.fn(),
    error: jest.fn(),
    dismiss: jest.fn(),
  },
}));

jest.mock('../../components/auth/SessionIdleWarningToast', () => {
  const React = require('react');
  const MockToast = (props) => React.createElement('div', { 'data-testid': 'idle-toast', ...props });
  return {
    __esModule: true,
    default: MockToast,
    sessionIdleWarningToastStyles: { sonnerHost: 'sonner-host-mock' },
  };
});

const lastIdleToastProps = () => {
  const renderFn = toast.custom.mock.calls.at(-1)?.[0];
  expect(typeof renderFn).toBe('function');
  const element = renderFn();
  return element.props;
};

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
    toast.custom.mockClear();
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
    expect(toast.custom).toHaveBeenCalledWith(
      expect.any(Function),
      expect.objectContaining({
        dismissible: false,
        duration: Infinity,
      })
    );
    const props = lastIdleToastProps();
    expect(props.secondsLeft).toBeGreaterThan(0);
    expect(props.renewing).toBe(false);
  });

  it('activité locale pendant le préavis → prolongement auto (refresh)', async () => {
    mockTryRefreshSessionIfNeeded.mockResolvedValue({ status: 'refreshed' });
    startSessionIdleGuard();
    recordUserActivity();
    jest.advanceTimersByTime(SESSION_IDLE_TIMEOUT_MS + 1000);

    expect(isSessionIdleWarningActive()).toBe(true);

    recordUserActivity();
    // handleStayConnected est async (import dynamique sessionKeepAlive).
    for (let i = 0; i < 20 && isSessionIdleWarningActive(); i += 1) {
      // eslint-disable-next-line no-await-in-loop
      await Promise.resolve();
    }

    expect(mockTryRefreshSessionIfNeeded).toHaveBeenCalledWith({
      force: true,
      reason: 'idle_stay_connected',
    });
    expect(isSessionIdleWarningActive()).toBe(false);
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

    const props = lastIdleToastProps();
    expect(props.onStay).toBeTruthy();
    await props.onStay();

    expect(mockTryRefreshSessionIfNeeded).toHaveBeenCalledWith({
      force: true,
      reason: 'idle_stay_connected',
    });
    expect(isSessionIdleWarningActive()).toBe(false);
    expect(mockLogoutUser).not.toHaveBeenCalled();
  });

  it('Rester connecté : erreur réseau → pas de logout, warning conservé', async () => {
    mockTryRefreshSessionIfNeeded.mockResolvedValue({ status: 'transient_failure' });
    startSessionIdleGuard();
    recordUserActivity();
    jest.advanceTimersByTime(SESSION_IDLE_TIMEOUT_MS + 1000);

    await lastIdleToastProps().onStay();

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

    await lastIdleToastProps().onStay();
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

    void lastIdleToastProps().onStay();
    await Promise.resolve();
    // Pendant RENEWING, le toast est réaffiché avec renewing=true → onStay no-op
    void lastIdleToastProps().onStay();
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
    expect(toast.custom).toHaveBeenCalled();
  });
});
