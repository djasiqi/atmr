import { toast } from 'sonner';
import {
  scheduleDeferredLogout,
  cancelDeferredLogout,
  initDeferredSessionLogout,
  notifySessionReauthRequired,
  SESSION_LOGOUT_COUNTDOWN_MS,
  resetDeferredSessionLogoutForTests,
} from '../deferredSessionLogout';
import {
  recordUserActivity,
  USER_ACTIVE_WINDOW_MS,
} from '../userActivityTracker';

jest.mock('sonner', () => ({
  toast: {
    warning: jest.fn(),
    dismiss: jest.fn(),
  },
}));

jest.mock('../apiClient', () => ({
  logoutUser: jest.fn(() => Promise.resolve()),
}));

describe('deferredSessionLogout', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    resetDeferredSessionLogoutForTests();
    initDeferredSessionLogout();
    toast.warning.mockClear();
    toast.dismiss.mockClear();
  });

  afterEach(() => {
    resetDeferredSessionLogoutForTests();
    jest.useRealTimers();
  });

  it('lance le compte à rebours si l’utilisateur est déjà inactif', () => {
    recordUserActivity();
    jest.advanceTimersByTime(USER_ACTIVE_WINDOW_MS + 1000);
    scheduleDeferredLogout();

    expect(toast.warning).toHaveBeenCalledWith(
      expect.stringContaining('Déconnexion automatique'),
      expect.objectContaining({ id: 'deferred-session-logout' })
    );
  });

  it('reporte la déconnexion tant que l’utilisateur interagit', () => {
    recordUserActivity();
    scheduleDeferredLogout();

    jest.advanceTimersByTime(5000);

    expect(toast.warning).toHaveBeenCalledWith(
      expect.stringContaining('période d\'inactivité'),
      expect.any(Object)
    );
    expect(toast.warning).not.toHaveBeenCalledWith(
      expect.stringContaining('Déconnexion automatique'),
      expect.any(Object)
    );
  });

  it('planifie via notifySessionReauthRequired après init', () => {
    notifySessionReauthRequired();
    expect(toast.warning).toHaveBeenCalledWith(
      expect.stringContaining('période d\'inactivité'),
      expect.any(Object)
    );
  });

  it('reste silencieux si l’utilisateur travaille encore (silentUntilIdle)', () => {
    recordUserActivity();
    scheduleDeferredLogout({ silentUntilIdle: true });
    expect(toast.warning).not.toHaveBeenCalled();
  });

  it('déconnecte après le compte à rebours sans activité', async () => {
    const { logoutUser } = await import('../apiClient');
    recordUserActivity();
    jest.advanceTimersByTime(USER_ACTIVE_WINDOW_MS + 1000);
    scheduleDeferredLogout();

    jest.runAllTimers();
    await Promise.resolve();
    await Promise.resolve();

    expect(logoutUser).toHaveBeenCalled();
  });
});
