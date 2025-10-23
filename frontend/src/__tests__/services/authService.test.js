// frontend/tests/services/authService.test.js
import { loginUser, registerUser, logoutUser, resetPassword } from 'services/authService';
import apiClient from 'utils/apiClient';

// Mock apiClient
jest.mock('utils/apiClient');

describe('authService', () => {
  beforeEach(() => {
    // Clear mocks et localStorage avant chaque test
    jest.clearAllMocks();
    localStorage.clear();
  });

  describe('loginUser', () => {
    it('devrait se connecter avec succès et stocker le token', async () => {
      const mockResponse = {
        data: {
          token: 'fake-jwt-token',
          user: {
            public_id: 'user-123',
            first_name: 'Jean',
            last_name: 'Dupont',
            role: 'company',
            force_password_change: false,
          },
        },
      };

      apiClient.post.mockResolvedValue(mockResponse);

      const result = await loginUser({
        email: 'test@example.com',
        password: 'password123',
      });

      expect(apiClient.post).toHaveBeenCalledWith('/auth/login', {
        email: 'test@example.com',
        password: 'password123',
      });

      expect(result).toEqual({ success: true });
      expect(localStorage.getItem('authToken')).toBe('fake-jwt-token');
      expect(localStorage.getItem('public_id')).toBe('user-123');

      const storedUser = JSON.parse(localStorage.getItem('user'));
      expect(storedUser.first_name).toBe('Jean');
    });

    it('devrait retourner redirectToReset si force_password_change est true', async () => {
      const mockResponse = {
        data: {
          token: 'fake-jwt-token',
          user: {
            public_id: 'user-456',
            force_password_change: true,
          },
        },
      };

      apiClient.post.mockResolvedValue(mockResponse);

      const result = await loginUser({
        email: 'newuser@example.com',
        password: 'temp123',
      });

      expect(result).toEqual({ redirectToReset: true });
      expect(localStorage.getItem('authToken')).toBe('fake-jwt-token');
    });

    it('devrait lever une erreur si public_id est manquant', async () => {
      const mockResponse = {
        data: {
          token: 'fake-jwt-token',
          user: {
            first_name: 'Jean',
            // public_id manquant
          },
        },
      };

      apiClient.post.mockResolvedValue(mockResponse);

      await expect(loginUser({ email: 'test@example.com', password: 'pass' })).rejects.toThrow(
        'Public ID manquant'
      );
    });

    it('devrait propager les erreurs API', async () => {
      const mockError = new Error('Invalid credentials');
      apiClient.post.mockRejectedValue(mockError);

      await expect(loginUser({ email: 'wrong@example.com', password: 'wrong' })).rejects.toThrow(
        'Invalid credentials'
      );
    });
  });

  describe('registerUser', () => {
    it('devrait enregistrer un utilisateur avec succès', async () => {
      const mockResponse = {
        data: {
          message: 'User registered successfully',
          user_id: 789,
        },
      };

      apiClient.post.mockResolvedValue(mockResponse);

      const userData = {
        email: 'nouveau@example.com',
        password: 'secure123',
        first_name: 'Marie',
        last_name: 'Martin',
      };

      const result = await registerUser(userData);

      expect(apiClient.post).toHaveBeenCalledWith('/auth/register', userData);
      expect(result).toEqual(mockResponse.data);
    });

    it("devrait gérer les erreurs d'inscription", async () => {
      const mockError = new Error('Email already exists');
      apiClient.post.mockRejectedValue(mockError);

      await expect(registerUser({ email: 'exists@example.com' })).rejects.toThrow(
        'Email already exists'
      );
    });
  });

  describe('logoutUser', () => {
    it('devrait nettoyer le localStorage', () => {
      // Préparer localStorage
      localStorage.setItem('authToken', 'fake-token');
      localStorage.setItem('user', JSON.stringify({ id: 1 }));
      localStorage.setItem('public_id', 'user-123');

      logoutUser();

      expect(localStorage.getItem('authToken')).toBeNull();
      expect(localStorage.getItem('user')).toBeNull();
      expect(localStorage.getItem('public_id')).toBeNull();
    });
  });

  describe('resetPassword', () => {
    it('devrait mettre à jour le mot de passe', async () => {
      const mockResponse = {
        data: {
          message: 'Password updated successfully',
        },
      };

      apiClient.post.mockResolvedValue(mockResponse);

      const result = await resetPassword('newSecurePassword123');

      expect(apiClient.post).toHaveBeenCalledWith('/auth/update-password', {
        new_password: 'newSecurePassword123',
      });
      expect(result).toEqual(mockResponse.data);
    });

    it('devrait gérer les erreurs de mise à jour', async () => {
      const mockError = new Error('Invalid token');
      apiClient.post.mockRejectedValue(mockError);

      await expect(resetPassword('newPass123')).rejects.toThrow('Invalid token');
    });
  });
});
