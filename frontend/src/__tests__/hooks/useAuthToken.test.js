// frontend/tests/hooks/useAuthToken.test.js
import { renderHook } from '@testing-library/react';
import useAuthToken, { getAccessToken, getRefreshToken } from 'hooks/useAuthToken';
import { jwtDecode } from 'jwt-decode';

// Mock jwtDecode
jest.mock('jwt-decode');

describe('useAuthToken', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
    jest.spyOn(console, 'warn').mockImplementation(() => {});
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    console.warn.mockRestore();
    console.error.mockRestore();
  });

  it('devrait retourner null si aucun token', () => {
    const { result } = renderHook(() => useAuthToken());

    expect(result.current).toBeNull();
  });

  it('devrait décoder le token et retourner les infos utilisateur', () => {
    const mockToken = 'fake.jwt.token';
    const mockDecoded = {
      sub: 'user-123',
      role: 'company',
      company_id: 5,
      exp: Date.now() / 1000 + 3600, // Expire dans 1h
    };

    localStorage.setItem('authToken', mockToken);
    jwtDecode.mockReturnValue(mockDecoded);

    const { result } = renderHook(() => useAuthToken());

    expect(jwtDecode).toHaveBeenCalledWith(mockToken);
    expect(result.current).toEqual({
      ...mockDecoded,
      isCompany: true,
      isDriver: false,
      isClient: false,
      companyId: 5,
      userId: 'user-123',
      public_id: 'user-123',
    });
  });

  it('devrait identifier correctement un chauffeur', () => {
    const mockToken = 'fake.jwt.token';
    const mockDecoded = {
      sub: 'driver-456',
      role: 'driver',
      exp: Date.now() / 1000 + 3600,
    };

    localStorage.setItem('authToken', mockToken);
    jwtDecode.mockReturnValue(mockDecoded);

    const { result } = renderHook(() => useAuthToken());

    expect(result.current.isDriver).toBe(true);
    expect(result.current.isCompany).toBe(false);
    expect(result.current.isClient).toBe(false);
  });

  it('devrait identifier correctement un client', () => {
    const mockToken = 'fake.jwt.token';
    const mockDecoded = {
      sub: 'client-789',
      role: 'client',
      exp: Date.now() / 1000 + 3600,
    };

    localStorage.setItem('authToken', mockToken);
    jwtDecode.mockReturnValue(mockDecoded);

    const { result } = renderHook(() => useAuthToken());

    expect(result.current.isClient).toBe(true);
    expect(result.current.isDriver).toBe(false);
    expect(result.current.isCompany).toBe(false);
  });

  it('devrait retourner null si le token est expiré', () => {
    const mockToken = 'expired.jwt.token';
    const mockDecoded = {
      sub: 'user-123',
      role: 'company',
      exp: Date.now() / 1000 - 3600, // Expiré il y a 1h
    };

    localStorage.setItem('authToken', mockToken);
    jwtDecode.mockReturnValue(mockDecoded);

    const { result } = renderHook(() => useAuthToken());

    expect(result.current).toBeNull();
    expect(console.warn).toHaveBeenCalledWith('🔐 Token expiré');
  });

  it('devrait retourner null si le décodage échoue', () => {
    const mockToken = 'invalid.token';
    localStorage.setItem('authToken', mockToken);
    jwtDecode.mockImplementation(() => {
      throw new Error('Invalid token');
    });

    const { result } = renderHook(() => useAuthToken());

    expect(result.current).toBeNull();
    expect(console.error).toHaveBeenCalledWith(
      '❌ Erreur lors du décodage du token:',
      expect.any(Error)
    );
  });

  describe('getAccessToken', () => {
    it('devrait retourner le token depuis localStorage', () => {
      localStorage.setItem('authToken', 'my-access-token');

      const token = getAccessToken();

      expect(token).toBe('my-access-token');
    });

    it('devrait retourner null si pas de token', () => {
      const token = getAccessToken();

      expect(token).toBeNull();
    });
  });

  describe('getRefreshToken', () => {
    it('devrait retourner le refresh token depuis localStorage', () => {
      localStorage.setItem('refreshToken', 'my-refresh-token');

      const token = getRefreshToken();

      expect(token).toBe('my-refresh-token');
    });

    it('devrait retourner null si pas de refresh token', () => {
      const token = getRefreshToken();

      expect(token).toBeNull();
    });
  });
});
