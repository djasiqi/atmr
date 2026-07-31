import {
  shouldShowQueryError,
  getQueryErrorMessage,
  isRecoverableAuthError,
  isFreshTokenRequiredError,
  isSessionExpiredError,
  isRateLimitError,
  AUTH_TOKEN_NOT_FRESH,
  AUTH_SESSION_EXPIRED,
} from '../queryAuthError';

jest.mock('../apiClient', () => ({
  isAuthRefreshInProgress: jest.fn(() => false),
  isMissingTokenErrorPayload: jest.fn((data) => data?.error === 'missing_token'),
}));

const { isAuthRefreshInProgress } = require('../apiClient');

describe('queryAuthError', () => {
  beforeEach(() => {
    isAuthRefreshInProgress.mockReturnValue(false);
  });

  it('identifie une erreur token non fresh', () => {
    const error = { code: AUTH_TOKEN_NOT_FRESH, isFreshTokenRequired: true };
    expect(isFreshTokenRequiredError(error)).toBe(true);
    expect(shouldShowQueryError(error)).toBe(false);
  });

  it('masque missing_token / session expirée (pas de toast UI)', () => {
    const error = {
      code: AUTH_SESSION_EXPIRED,
      isSessionExpired: true,
      meta: { suppressAuthError: true },
      response: { status: 401, data: { error: 'missing_token' } },
    };
    expect(isSessionExpiredError(error)).toBe(true);
    expect(isRecoverableAuthError(error)).toBe(true);
    expect(shouldShowQueryError(error)).toBe(false);
    expect(getQueryErrorMessage(error)).toBe('');
  });

  it('masque les erreurs auth pendant un refresh', () => {
    isAuthRefreshInProgress.mockReturnValue(true);
    const error = { response: { status: 401 } };
    expect(isRecoverableAuthError(error)).toBe(true);
    expect(shouldShowQueryError(error)).toBe(false);
  });

  it('affiche les erreurs métier normales', () => {
    const error = { message: 'Erreur serveur' };
    expect(shouldShowQueryError(error)).toBe(true);
    expect(getQueryErrorMessage(error)).toBe('Erreur serveur');
  });

  it('identifie un quota API dépassé (429 ou too_many_requests)', () => {
    expect(isRateLimitError({ response: { status: 429 } })).toBe(true);
    expect(
      isRateLimitError({
        response: { status: 429, data: { error: 'too_many_requests' } },
      })
    ).toBe(true);
    expect(isRateLimitError({ status: 429 })).toBe(true);
    expect(isRateLimitError({ response: { status: 500 } })).toBe(false);
    expect(isRateLimitError(null)).toBe(false);
  });
});
