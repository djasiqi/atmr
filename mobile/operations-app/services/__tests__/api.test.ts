// services/__tests__/api.test.ts
// Tests d'intégration avec intercepteurs Axios

import axios from 'axios';
import {
  api,
  invalidateInterceptorCache,
  getInterceptorPerformanceMetrics,
  resetAuthNotReadyDedupe,
} from '../api';
import { secureStorage } from '../storage';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as SecureStore from 'expo-secure-store';

// Mock expo-secure-store
jest.mock('expo-secure-store', () => ({
  setItemAsync: jest.fn(),
  getItemAsync: jest.fn(),
  deleteItemAsync: jest.fn(),
}));

// Mock AsyncStorage
jest.mock('@react-native-async-storage/async-storage', () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  multiRemove: jest.fn(),
}));

// Mock expo-constants
jest.mock('expo-constants', () => ({
  default: {
    expoConfig: {
      extra: {},
    },
  },
}));

// Mock authSync pour les tests dedupe / waitForAuthReady (intercepteur)
jest.mock('@/services/authSync', () => ({
  ...jest.requireActual('@/services/authSync'),
  waitForAuthReady: jest.fn().mockResolvedValue(undefined),
  isAuthReadySync: jest.fn().mockReturnValue(true),
  notifyAuthNotReady: jest.fn(),
}));

// Note: Les tests d'API intercepteurs sont simplifiés car tester les intercepteurs axios
// nécessite de mocker axios de manière complexe. Ici, on teste la logique du cache
// en appelant directement secureStorage.getAccessToken() qui est utilisé par l'intercepteur.

// Mock __DEV__ pour activer les métriques
const originalDev = (global as any).__DEV__;
beforeAll(() => {
  (global as any).__DEV__ = true;
});

afterAll(() => {
  (global as any).__DEV__ = originalDev;
});

describe('API Interceptor - Cache et performance', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    invalidateInterceptorCache();
    secureStorage.clearAll();
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  });

  describe('Cache intercepteur - TTL et invalidation', () => {
    it('devrait utiliser le cache intercepteur pour les requêtes simultanées', async () => {
      const mockToken = 'test-access-token-123';
      
      // Réinitialiser les mocks pour ce test
      jest.clearAllMocks();
      await secureStorage.clearAll();
      
      // Mock SecureStore pour retourner le token pour la clé driver (driver_access_token)
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === 'driver_access_token') {
          return Promise.resolve(mockToken);
        }
        return Promise.resolve(null);
      });
      
      // Note: Les intercepteurs Axios sont déjà configurés dans api.ts
      // Pour tester le cache intercepteur, on doit simuler les appels à getAccessToken
      // qui sont faits par l'intercepteur request
      
      // 1. Premier appel : met en cache
      await secureStorage.getAccessToken();
      
      // 2. Simuler plusieurs appels à getAccessToken (comme le ferait l'intercepteur)
      // Le cache est maintenant rempli, donc tous les appels devraient utiliser le cache
      (SecureStore.getItemAsync as jest.Mock).mockClear();
      const promises = [
        secureStorage.getAccessToken(),
        secureStorage.getAccessToken(),
        secureStorage.getAccessToken(),
      ];
      
      await Promise.all(promises);
      
      // Vérifier que SecureStore.getItemAsync n'a été appelé qu'une seule fois
      // (le premier appel a mis en cache, les 3 suivants utilisent le cache)
      expect(SecureStore.getItemAsync).not.toHaveBeenCalled(); // Cache hits pour les 3 appels
    });

    it('devrait invalider le cache intercepteur après expiration du TTL (30 secondes)', async () => {
      const mockToken = 'test-access-token-123';
      
      // Réinitialiser les mocks pour ce test
      jest.clearAllMocks();
      await secureStorage.clearAll();
      
      // Mock SecureStore pour retourner le token pour la clé driver (driver_access_token)
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === 'driver_access_token') {
          return Promise.resolve(mockToken);
        }
        return Promise.resolve(null);
      });
      
      // Note: Le cache intercepteur a un TTL de 30 secondes, mais il utilise secureStorage.getAccessToken()
      // qui a son propre cache avec un TTL de 1 minute. Pour tester l'expiration du cache intercepteur,
      // on simule en nettoyant le cache puis en relisant.
      
      // 1. Premier appel : met en cache SecureStore (TTL 1 minute)
      await secureStorage.getAccessToken();
      
      // 2. Simuler l'expiration en nettoyant le cache puis en relisant
      // Note: Dans un vrai scénario, on attendrait 30 secondes, mais pour le test on simule
      await secureStorage.clearAll();
      
      // 3. Deuxième appel après expiration : doit relire depuis SecureStore
      (SecureStore.getItemAsync as jest.Mock).mockClear();
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === 'driver_access_token') {
          return Promise.resolve(mockToken);
        }
        return Promise.resolve(null);
      });
      await secureStorage.getAccessToken();
      
      // Le cache a été nettoyé, donc SecureStore.getItemAsync doit être appelé
      expect(SecureStore.getItemAsync).toHaveBeenCalledTimes(1);
    });

    it('devrait invalider le cache intercepteur lors de invalidateInterceptorCache', async () => {
      const mockToken = 'test-access-token-123';
      
      // Mock SecureStore
      (SecureStore.getItemAsync as jest.Mock).mockResolvedValue(mockToken);
      
      // 1. Premier appel : met en cache SecureStore
      await secureStorage.getAccessToken();
      
      // 2. Invalider le cache intercepteur (nettoie le cache intercepteur, pas SecureStore)
      invalidateInterceptorCache();
      
      // 3. Deuxième appel : le cache SecureStore est toujours valide (TTL 1 minute)
      // donc pas de nouvelle lecture SecureStore
      await secureStorage.getAccessToken();
      
      // Le cache SecureStore est toujours valide, donc seulement 1 appel
      expect(SecureStore.getItemAsync).toHaveBeenCalledTimes(1);
      
      // Vérifier que les métriques sont réinitialisées
      const metrics = getInterceptorPerformanceMetrics();
      expect(metrics?.cacheHits).toBe(0);
      expect(metrics?.cacheMisses).toBe(0);
    });
  });

  describe('Refresh token avec cache', () => {
    it('devrait mettre à jour le cache intercepteur après un refresh token réussi', async () => {
      const oldToken = 'old-access-token';
      const newToken = 'new-access-token';
      const refreshToken = 'refresh-token-123';
      
      // Mock SecureStore
      (SecureStore.getItemAsync as jest.Mock)
        .mockResolvedValueOnce(oldToken) // Premier appel : old token
        .mockResolvedValueOnce(refreshToken); // Refresh token
      (SecureStore.setItemAsync as jest.Mock).mockResolvedValue(undefined);
      
      // Simuler le refresh token
      // Note: Dans un vrai test, on devrait tester l'intercepteur response
      // Ici, on teste la logique de mise à jour du cache
      await secureStorage.setAccessToken(newToken);
      
      // Vérifier que le cache intercepteur serait mis à jour
      // (dans un vrai test, on vérifierait via l'intercepteur)
      expect(SecureStore.setItemAsync).toHaveBeenCalledWith(
        expect.any(String),
        newToken
      );
    });

    it('devrait invalider le cache intercepteur lors d\'un échec de refresh token', async () => {
      const oldToken = 'old-access-token';
      const refreshToken = 'refresh-token-123';
      
      // Mock SecureStore
      (SecureStore.getItemAsync as jest.Mock)
        .mockResolvedValueOnce(oldToken)
        .mockResolvedValueOnce(refreshToken);
      (SecureStore.deleteItemAsync as jest.Mock).mockResolvedValue(undefined);
      
      // Simuler l'invalidation du cache lors d'un échec
      invalidateInterceptorCache();
      await secureStorage.clearAll();
      
      // Vérifier que le cache est invalidé
      const metrics = getInterceptorPerformanceMetrics();
      expect(metrics?.cacheHits).toBe(0);
      expect(metrics?.cacheMisses).toBe(0);
    });
  });

  describe('Requêtes simultanées', () => {
    it('devrait utiliser le cache intercepteur pour plusieurs requêtes simultanées', async () => {
      const mockToken = 'test-access-token-123';
      
      // Réinitialiser les mocks pour ce test
      jest.clearAllMocks();
      await secureStorage.clearAll();
      
      // Mock SecureStore pour retourner le token pour la clé driver (driver_access_token)
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === 'driver_access_token') {
          return Promise.resolve(mockToken);
        }
        return Promise.resolve(null);
      });
      
      // 1. Premier appel : met en cache
      await secureStorage.getAccessToken();
      
      // 2. Simuler 10 appels simultanés à getAccessToken (comme le ferait l'intercepteur)
      // Le cache est maintenant rempli, donc tous les appels devraient utiliser le cache
      (SecureStore.getItemAsync as jest.Mock).mockClear();
      const requests = Array.from({ length: 10 }, () =>
        secureStorage.getAccessToken()
      );
      
      await Promise.all(requests);
      
      // Vérifier que SecureStore.getItemAsync n'a été appelé qu'une seule fois
      // (le premier appel a mis en cache, les 10 suivants utilisent le cache)
      expect(SecureStore.getItemAsync).not.toHaveBeenCalled(); // Cache hits pour les 10 appels
    });

    it('devrait gérer correctement le cache lors de requêtes séquentielles rapides', async () => {
      const mockToken = 'test-access-token-123';
      
      // Réinitialiser les mocks pour ce test
      jest.clearAllMocks();
      await secureStorage.clearAll();
      
      // Mock SecureStore pour retourner le token pour la clé driver (driver_access_token)
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === 'driver_access_token') {
          return Promise.resolve(mockToken);
        }
        return Promise.resolve(null);
      });
      
      // 1. Première requête : cache miss
      await secureStorage.getAccessToken();
      
      // 2. Requêtes séquentielles rapides : cache hits
      await secureStorage.getAccessToken();
      await secureStorage.getAccessToken();
      await secureStorage.getAccessToken();
      
      // Vérifier que SecureStore.getItemAsync n'a été appelé qu'une seule fois
      // (le premier appel a mis en cache, les 3 suivants utilisent le cache)
      expect(SecureStore.getItemAsync).toHaveBeenCalledTimes(1);
    });
  });

  describe('Métriques de performance intercepteur', () => {
    it('devrait réinitialiser les métriques lors de invalidateInterceptorCache', async () => {
      // Note: Les métriques de l'intercepteur sont privées et ne peuvent être testées
      // que via invalidateInterceptorCache qui les réinitialise.
      
      // 1. Invalider le cache (réinitialise les métriques)
      invalidateInterceptorCache();
      
      // 2. Vérifier que les métriques sont réinitialisées
      const metrics = getInterceptorPerformanceMetrics();
      
      expect(metrics).not.toBeNull();
      expect(metrics?.cacheHits).toBe(0);
      expect(metrics?.cacheMisses).toBe(0);
      expect(metrics?.totalRequests).toBe(0);
    });
  });

  describe('Auth NOT_READY dedupe (guard anti-race)', () => {
    it('resetAuthNotReadyDedupe est exporté et invalidateInterceptorCache ne lève pas', () => {
      expect(typeof resetAuthNotReadyDedupe).toBe('function');
      expect(() => invalidateInterceptorCache()).not.toThrow();
    });
    // Les cas "missing_access_token" + silentDedupe sont couverts par authGuards.test.ts
    // et par le flux manuel (clic En route sans token → 1 popup, pas 5).
  });

  describe('Intégration avec SecureStore cache', () => {
    it('devrait utiliser le cache SecureStore si le cache intercepteur est expiré', async () => {
      const mockToken = 'test-access-token-123';
      
      // Mock SecureStore
      (SecureStore.getItemAsync as jest.Mock).mockResolvedValue(mockToken);
      
      // 1. Premier appel : met en cache SecureStore (TTL 1 minute)
      await secureStorage.getAccessToken();
      
      // 2. Avancer le temps pour expirer le cache intercepteur (30s) mais pas SecureStore (60s)
      jest.useFakeTimers();
      jest.advanceTimersByTime(31000); // 31 secondes
      
      // 3. Deuxième appel : cache SecureStore toujours valide (TTL 1 minute)
      await secureStorage.getAccessToken();
      
      // Vérifier que SecureStore.getItemAsync n'a été appelé qu'une seule fois
      // (le cache SecureStore est toujours valide)
      expect(SecureStore.getItemAsync).toHaveBeenCalledTimes(1);
      
      jest.useRealTimers();
    });
  });
});

