/**
 * Tests useCompanyAuthToken — garde P0 company_dispatch.
 * Cas : user sans token → false | token valide → true | logout → false
 */
import { renderHook, waitFor } from '@testing-library/react';
import useCompanyAuthToken from '../useCompanyAuthToken';
import { jwtDecode } from 'jwt-decode';

jest.mock('jwt-decode');

const COMPANY_USER_KEY = 'company_user';
const COMPANY_ACCESS_TOKEN_KEY = 'company_access_token';

const mockStoredUser = {
  id: 1,
  public_id: 'company-123',
  role: 'company',
  company_id: 1,
};

const mockDecodedToken = {
  sub: 'company-123',
  role: 'company',
  company_id: 1,
  exp: Math.floor(Date.now() / 1000) + 3600, // valide 1h
};

describe('useCompanyAuthToken', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
  });

  it('user en localStorage + token absent → isCompanyAuthReady=false', async () => {
    localStorage.setItem(COMPANY_USER_KEY, JSON.stringify(mockStoredUser));
    // Pas de company_access_token

    const { result } = renderHook(() => useCompanyAuthToken());

    await waitFor(() => {
      expect(result.current.isCompanyAuthReady).toBe(false);
      expect(result.current.user).toBeNull();
    });
  });

  it('user + token valide → isCompanyAuthReady=true', async () => {
    const fakeToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjk5OTk5OTk5OTl9.x';
    localStorage.setItem(COMPANY_ACCESS_TOKEN_KEY, fakeToken);
    localStorage.setItem(COMPANY_USER_KEY, JSON.stringify(mockStoredUser));
    jwtDecode.mockReturnValue(mockDecodedToken);

    const { result } = renderHook(() => useCompanyAuthToken());

    await waitFor(() => {
      expect(result.current.isCompanyAuthReady).toBe(true);
      expect(result.current.user).toBeDefined();
      expect(result.current.user.role).toBe('company');
      expect(result.current.user.companyId).toBe(1);
    });
  });

  it('localStorage vide (logout) → isCompanyAuthReady=false', async () => {
    // Aucun token, aucun user
    const { result } = renderHook(() => useCompanyAuthToken());

    await waitFor(() => {
      expect(result.current.isCompanyAuthReady).toBe(false);
      expect(result.current.user).toBeNull();
    });
  });

  it('token expiré → isCompanyAuthReady=false et nettoie localStorage', async () => {
    const fakeToken = 'expired-token';
    localStorage.setItem(COMPANY_ACCESS_TOKEN_KEY, fakeToken);
    localStorage.setItem(COMPANY_USER_KEY, JSON.stringify(mockStoredUser));
    jwtDecode.mockReturnValue({
      ...mockDecodedToken,
      exp: Math.floor(Date.now() / 1000) - 3600, // expiré
    });

    const { result } = renderHook(() => useCompanyAuthToken());

    await waitFor(() => {
      expect(result.current.isCompanyAuthReady).toBe(false);
      expect(result.current.user).toBeNull();
    });

    expect(localStorage.getItem(COMPANY_ACCESS_TOKEN_KEY)).toBeNull();
  });
});
