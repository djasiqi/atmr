jest.mock('../deferredSessionLogout', () => ({
  cancelDeferredLogout: jest.fn(),
  stopSessionIdleGuard: jest.fn(),
}));

jest.mock('../sessionKeepAlive', () => ({
  suspendSessionKeepAlive: jest.fn(),
}));

jest.mock('../../services/companySocket', () => ({
  disconnectCompanySocket: jest.fn(),
}));

const { default: apiClient, logoutUser, cleanLocalSession } = require('../apiClient');
const { AUTH_NAVIGATE_EVENT } = require('../authNavigation');
const { AUTH_LOGOUT_AT_KEY, AUTH_LOGOUT_REASON_KEY } = require('../sessionLogoutState');

describe('logoutUser', () => {
  let postSpy;
  let deleteSpy;
  let authChangedHandler;
  let navigateHandler;

  beforeEach(() => {
    postSpy = jest.spyOn(apiClient, 'post').mockResolvedValue({ status: 200 });
    deleteSpy = jest.spyOn(apiClient, 'delete').mockResolvedValue({ status: 204 });
    authChangedHandler = jest.fn();
    navigateHandler = jest.fn();
    window.addEventListener('auth-changed', authChangedHandler);
    window.addEventListener(AUTH_NAVIGATE_EVENT, navigateHandler);
    localStorage.setItem('app_user', JSON.stringify({ public_id: 'u-1', role: 'admin' }));
    localStorage.setItem('lirie_auth_env', 'app');
    sessionStorage.clear();
  });

  afterEach(() => {
    postSpy.mockRestore();
    deleteSpy.mockRestore();
    window.removeEventListener('auth-changed', authChangedHandler);
    window.removeEventListener(AUTH_NAVIGATE_EVENT, navigateHandler);
    cleanLocalSession();
    sessionStorage.clear();
    localStorage.clear();
  });

  it('appelle /auth/logout avant de nettoyer la session locale', async () => {
    const callOrder = [];
    postSpy.mockImplementation(async () => {
      callOrder.push('logout');
      return { status: 200 };
    });

    const originalLocation = window.location;
    delete window.location;
    window.location = { pathname: '/dashboard/admin/x', search: '', href: '' };

    await logoutUser({ redirect: false });

    expect(postSpy).toHaveBeenCalledWith(
      '/auth/logout',
      {},
      { skipAuthRedirect: true, skipFreshTokenLogout: true }
    );
    expect(deleteSpy).not.toHaveBeenCalled();
    expect(callOrder).toEqual(['logout']);
    expect(localStorage.getItem('app_user')).toBeNull();
    expect(authChangedHandler).toHaveBeenCalledTimes(1);
    expect(navigateHandler).not.toHaveBeenCalled();
    expect(localStorage.getItem(AUTH_LOGOUT_AT_KEY)).toBeTruthy();

    window.location = originalLocation;
  });

  it('émet auth-changed et une navigation SPA vers login', async () => {
    const originalLocation = window.location;
    delete window.location;
    window.location = { pathname: '/dashboard/admin/x', search: '', href: '/login' };

    await logoutUser({ redirect: true });

    expect(authChangedHandler).toHaveBeenCalledTimes(1);
    expect(navigateHandler).toHaveBeenCalledTimes(1);
    expect(navigateHandler.mock.calls[0][0].detail).toEqual({
      to: '/login',
      replace: true,
    });
    expect(sessionStorage.getItem('lirie_explicit_logout')).toBeTruthy();

    window.location = originalLocation;
  });

  it('nettoie la session locale même si /auth/logout échoue', async () => {
    postSpy.mockRejectedValue({ response: { status: 401 }, message: 'Unauthorized' });

    const originalLocation = window.location;
    delete window.location;
    window.location = { pathname: '/dashboard/company/co-1', search: '', href: '' };

    await logoutUser({ redirect: false });

    expect(localStorage.getItem('app_user')).toBeNull();
    expect(authChangedHandler).toHaveBeenCalledTimes(1);
    expect(deleteSpy).not.toHaveBeenCalled();

    window.location = originalLocation;
  });

  it('immediate : nettoie sans attendre la fin du logout serveur', async () => {
    let resolveLogout;
    postSpy.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveLogout = resolve;
        })
    );

    const originalLocation = window.location;
    delete window.location;
    window.location = { pathname: '/dashboard/admin/x', search: '', href: '' };

    await logoutUser({
      immediate: true,
      reason: 'session_expired',
      preserveNext: true,
      redirect: true,
    });

    expect(localStorage.getItem('app_user')).toBeNull();
    expect(sessionStorage.getItem(AUTH_LOGOUT_REASON_KEY)).toBe('session_expired');
    expect(localStorage.getItem(AUTH_LOGOUT_AT_KEY)).toBeTruthy();
    expect(authChangedHandler).toHaveBeenCalledTimes(1);
    expect(navigateHandler).toHaveBeenCalled();

    resolveLogout({ status: 200 });
    window.location = originalLocation;
  });
});
