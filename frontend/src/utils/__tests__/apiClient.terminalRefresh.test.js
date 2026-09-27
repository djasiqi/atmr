/**
 * Classification des échecs refresh : 429 / CSRF 403 ne doivent pas déconnecter.
 */

jest.mock('../deferredSessionLogout', () => ({
  notifySessionReauthRequired: jest.fn(),
  stopSessionIdleGuard: jest.fn(),
}));

jest.mock('../sessionKeepAlive', () => ({
  tryRefreshSessionIfNeeded: jest.fn(() => Promise.resolve({ status: 'skipped' })),
  suspendSessionKeepAlive: jest.fn(),
}));

const {
  isTerminalRefreshFailure,
  isCsrfFailurePayload,
} = require('../apiClient');

describe('isTerminalRefreshFailure', () => {
  it('429 → non terminal', () => {
    expect(isTerminalRefreshFailure({ response: { status: 429, data: { error: 'too_many_requests', message: '50 per 1 hour' } } })).toBe(false);
  });

  it('403 CSRF → non terminal (quota csrf-token saturé)', () => {
    expect(
      isTerminalRefreshFailure({
        response: { status: 403, data: { error: 'Token CSRF manquant' } },
      })
    ).toBe(false);
    expect(
      isTerminalRefreshFailure({
        response: { status: 403, data: { error: 'Token CSRF invalide ou expiré' } },
      })
    ).toBe(false);
  });

  it('401 → terminal', () => {
    expect(
      isTerminalRefreshFailure({
        response: { status: 401, data: { error: 'Refresh token invalide' } },
      })
    ).toBe(true);
  });

  it('403 générique (non CSRF) → terminal', () => {
    expect(
      isTerminalRefreshFailure({
        response: { status: 403, data: { error: 'forbidden' } },
      })
    ).toBe(true);
  });

  it('isCsrfFailurePayload détecte les messages CSRF', () => {
    expect(isCsrfFailurePayload({ error: 'Token CSRF manquant' })).toBe(true);
    expect(isCsrfFailurePayload({ message: 'csrf required' })).toBe(true);
    expect(isCsrfFailurePayload({ error: 'forbidden' })).toBe(false);
  });
});
