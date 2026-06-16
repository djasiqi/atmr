import {
  beginExplicitLogout,
  clearExplicitLogoutMarker,
  endExplicitLogout,
  EXPLICIT_LOGOUT_SESSION_KEY,
  hasRecentExplicitLogout,
  isExplicitLogoutInProgress,
} from '../sessionLogoutState';
import { suspendSessionKeepAlive, tryRefreshSessionIfNeeded } from '../sessionKeepAlive';

jest.mock('../apiClient', () => ({
  refreshSessionTokens: jest.fn(() => Promise.resolve(true)),
}));

const { refreshSessionTokens } = require('../apiClient');

describe('sessionLogoutState', () => {
  afterEach(() => {
    endExplicitLogout();
    clearExplicitLogoutMarker();
    sessionStorage.clear();
    refreshSessionTokens.mockClear();
  });

  it('marque une déconnexion explicite en cours', () => {
    expect(isExplicitLogoutInProgress()).toBe(false);
    beginExplicitLogout();
    expect(isExplicitLogoutInProgress()).toBe(true);
    expect(sessionStorage.getItem(EXPLICIT_LOGOUT_SESSION_KEY)).toBeTruthy();
  });

  it('bloque le keep-alive pendant une déconnexion explicite', async () => {
    localStorage.setItem('app_user', JSON.stringify({ role: 'company', public_id: 'u1' }));
    beginExplicitLogout();
    suspendSessionKeepAlive();

    const ok = await tryRefreshSessionIfNeeded({ force: true });

    expect(ok).toBe(false);
    expect(refreshSessionTokens).not.toHaveBeenCalled();
  });

  it('détecte une déconnexion récente via sessionStorage', () => {
    beginExplicitLogout();
    expect(hasRecentExplicitLogout()).toBe(true);
    clearExplicitLogoutMarker();
    expect(hasRecentExplicitLogout()).toBe(false);
  });
});
