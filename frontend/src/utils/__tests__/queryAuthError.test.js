import {
  shouldShowQueryError,
  getQueryErrorMessage,
  isRecoverableAuthError,
  isFreshTokenRequiredError,
  AUTH_TOKEN_NOT_FRESH,
} from '../queryAuthError';

jest.mock('../apiClient', () => ({
  isAuthRefreshInProgress: jest.fn(() => false),
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
});
