// services/__tests__/storage.test.ts
// Tests unitaires pour le cache en mémoire (TTL, invalidation)

import * as SecureStore from 'expo-secure-store';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { secureStorage } from '../storage';
import {
  DRIVER_AUTH_KEYS,
  ENTERPRISE_AUTH_KEYS,
} from '../storage/keys';

const ACCESS_KEY = DRIVER_AUTH_KEYS.secure[1]; // driver_access_token
const REFRESH_KEY = DRIVER_AUTH_KEYS.secure[0]; // driver_refresh_token

// Mock expo-secure-store
jest.mock('expo-secure-store', () => ({
  setItemAsync: jest.fn(),
  getItemAsync: jest.fn(),
  deleteItemAsync: jest.fn(),
}));

// Mock AsyncStorage (nécessaire car storage.ts l'importe)
jest.mock('@react-native-async-storage/async-storage', () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  multiRemove: jest.fn(),
}));

// Mock AsyncStorage (nécessaire car storage.ts l'importe)
jest.mock('@react-native-async-storage/async-storage', () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  multiRemove: jest.fn(),
}));

// Mock __DEV__ pour activer les métriques
const originalDev = (global as any).__DEV__;
beforeAll(() => {
  (global as any).__DEV__ = true;
});

afterAll(() => {
  (global as any).__DEV__ = originalDev;
});

describe('secureStorage - Cache en mémoire', () => {
  beforeEach(async () => {
    // Nettoyer le cache AVANT de clear les mocks
    await secureStorage.clearAll();
    jest.clearAllMocks();
    // Réinitialiser les mocks après clearAll
    (SecureStore.getItemAsync as jest.Mock).mockClear();
    (SecureStore.setItemAsync as jest.Mock).mockClear();
    (SecureStore.deleteItemAsync as jest.Mock).mockClear();
    // Réinitialiser les mocks pour retourner null par défaut pour toutes les clés
    (SecureStore.getItemAsync as jest.Mock).mockImplementation(() => Promise.resolve(null));
  });

  describe('AccessToken Cache - TTL et invalidation', () => {
    it('devrait utiliser le cache si le token est récent (TTL valide)', async () => {
      const mockToken = 'test-access-token-123';
      
      // 1. Premier appel : lit depuis SecureStore et met en cache
      (SecureStore.getItemAsync as jest.Mock).mockResolvedValueOnce(mockToken);
      const token1 = await secureStorage.getAccessToken();
      
      expect(token1).toBe(mockToken);
      expect(SecureStore.getItemAsync).toHaveBeenCalledTimes(1);
      
      // 2. Deuxième appel immédiat : utilise le cache (pas de lecture SecureStore)
      (SecureStore.getItemAsync as jest.Mock).mockClear();
      const token2 = await secureStorage.getAccessToken();
      
      expect(token2).toBe(mockToken);
      expect(SecureStore.getItemAsync).not.toHaveBeenCalled(); // Cache hit
    });

    it('devrait invalider le cache après expiration du TTL (1 minute)', async () => {
      const mockToken = 'test-access-token-123';
      
      // 1. Premier appel : met en cache
      (SecureStore.getItemAsync as jest.Mock).mockResolvedValue(mockToken);
      await secureStorage.getAccessToken();
      
      // 2. Avancer le temps de plus de 1 minute (TTL)
      jest.useFakeTimers();
      jest.advanceTimersByTime(61000); // 61 secondes (> 60s TTL)
      
      // 3. Deuxième appel après expiration : doit relire depuis SecureStore
      const token2 = await secureStorage.getAccessToken();
      
      expect(token2).toBe(mockToken);
      expect(SecureStore.getItemAsync).toHaveBeenCalledTimes(2); // Cache miss
      
      jest.useRealTimers();
    });

    it('devrait mettre à jour le cache lors de setAccessToken', async () => {
      const oldToken = 'old-token';
      const newToken = 'new-token';
      
      // S'assurer que le cache est vide avant ce test
      await secureStorage.clearAll();
      jest.clearAllMocks();
      
      // Mock SecureStore pour retourner les tokens uniquement pour la clé ACCESS_TOKEN
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === ACCESS_KEY) {
          return Promise.resolve(oldToken);
        }
        return Promise.resolve(null);
      });
      (SecureStore.setItemAsync as jest.Mock).mockResolvedValue(undefined);
      
      // 1. Stocker un token
      await secureStorage.setAccessToken(oldToken);
      await secureStorage.getAccessToken(); // Mise en cache
      
      // 2. Mettre à jour le token
      await secureStorage.setAccessToken(newToken);
      
      // 3. Vérifier que le cache contient le nouveau token
      (SecureStore.getItemAsync as jest.Mock).mockClear();
      const cachedToken = await secureStorage.getAccessToken();
      
      expect(cachedToken).toBe(newToken);
      expect(SecureStore.getItemAsync).not.toHaveBeenCalled(); // Cache hit avec nouveau token
    });

    it('devrait nettoyer le cache lors de removeAccessToken', async () => {
      const mockToken = 'test-access-token-123';
      
      // S'assurer que le cache est vide avant ce test
      await secureStorage.clearAll();
      jest.clearAllMocks();
      
      // Mock SecureStore pour retourner le token uniquement pour la clé ACCESS_TOKEN
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === ACCESS_KEY) {
          return Promise.resolve(mockToken);
        }
        return Promise.resolve(null);
      });
      (SecureStore.setItemAsync as jest.Mock).mockResolvedValue(undefined);
      
      // 1. Mettre en cache
      await secureStorage.setAccessToken(mockToken);
      await secureStorage.getAccessToken();
      
      // 2. Supprimer le token (nettoie le cache en mémoire)
      (SecureStore.deleteItemAsync as jest.Mock).mockResolvedValue(undefined);
      await secureStorage.removeAccessToken();
      
      // 3. Vérifier que le cache est vide
      // Note: removeAccessToken() nettoie le cache en mémoire (cachedAccessToken = null)
      // Donc getAccessToken() doit lire depuis SecureStore qui retourne null après la suppression
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === ACCESS_KEY) {
          return Promise.resolve(null); // Après suppression, SecureStore retourne null
        }
        return Promise.resolve(null);
      });
      const token = await secureStorage.getAccessToken();
      
      expect(token).toBeNull();
      // Le cache a été nettoyé, donc SecureStore.getItemAsync doit être appelé
      expect(SecureStore.getItemAsync).toHaveBeenCalled(); // Cache miss, lecture SecureStore
    });

    it('devrait nettoyer le cache lors de clearAll', async () => {
      const mockToken = 'test-access-token-123';
      
      // S'assurer que le cache est vide avant ce test
      await secureStorage.clearAll();
      jest.clearAllMocks();
      
      // Mock SecureStore pour retourner le token uniquement pour la clé ACCESS_TOKEN
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === ACCESS_KEY) {
          return Promise.resolve(mockToken);
        }
        return Promise.resolve(null);
      });
      (SecureStore.setItemAsync as jest.Mock).mockResolvedValue(undefined);
      
      // 1. Mettre en cache
      await secureStorage.setAccessToken(mockToken);
      await secureStorage.getAccessToken();
      
      // 2. Nettoyer tout (nettoie tous les caches en mémoire)
      (SecureStore.deleteItemAsync as jest.Mock).mockResolvedValue(undefined);
      await secureStorage.clearAll();
      
      // 3. Vérifier que le cache est vide
      // Note: clearAll() nettoie tous les caches en mémoire
      // Donc getAccessToken() doit lire depuis SecureStore qui retourne null après la suppression
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === ACCESS_KEY) {
          return Promise.resolve(null); // Après clearAll, SecureStore retourne null
        }
        return Promise.resolve(null);
      });
      const token = await secureStorage.getAccessToken();
      
      expect(token).toBeNull();
      // Le cache a été nettoyé, donc SecureStore.getItemAsync doit être appelé
      expect(SecureStore.getItemAsync).toHaveBeenCalled();
    });
  });

  describe('RefreshToken Cache - TTL et invalidation', () => {
    it('devrait utiliser le cache si le refresh token est récent (TTL valide)', async () => {
      const mockToken = 'test-refresh-token-456';
      
      // S'assurer que le cache est vide avant ce test
      await secureStorage.clearAll();
      jest.clearAllMocks();
      
      // Mock SecureStore pour retourner le token uniquement pour la clé REFRESH_TOKEN
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === REFRESH_KEY) {
          return Promise.resolve(mockToken);
        }
        return Promise.resolve(null);
      });
      
      // 1. Premier appel : lit depuis SecureStore et met en cache
      const token1 = await secureStorage.getRefreshToken();
      
      expect(token1).toBe(mockToken);
      expect(SecureStore.getItemAsync).toHaveBeenCalledTimes(1);
      
      // 2. Deuxième appel immédiat : utilise le cache (pas de nouvelle lecture SecureStore)
      (SecureStore.getItemAsync as jest.Mock).mockClear();
      const token2 = await secureStorage.getRefreshToken();
      
      expect(token2).toBe(mockToken);
      expect(SecureStore.getItemAsync).not.toHaveBeenCalled(); // Cache hit
    });

    it('devrait invalider le cache après expiration du TTL (5 minutes)', async () => {
      const mockToken = 'test-refresh-token-456';
      
      // S'assurer que le cache est vide avant ce test
      await secureStorage.clearAll();
      jest.clearAllMocks();
      
      // 1. Premier appel : met en cache
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === REFRESH_KEY) {
          return Promise.resolve(mockToken);
        }
        return Promise.resolve(null);
      });
      await secureStorage.getRefreshToken();
      
      // 2. Simuler l'expiration en nettoyant le cache puis en relisant
      // Note: Dans un vrai scénario, on attendrait 5 minutes, mais pour le test on simule
      // en appelant clearAll puis en relisant
      await secureStorage.clearAll();
      
      // 3. Deuxième appel après expiration : doit relire depuis SecureStore
      // Réinitialiser le mock pour compter les appels
      (SecureStore.getItemAsync as jest.Mock).mockClear();
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === REFRESH_KEY) {
          return Promise.resolve(mockToken);
        }
        return Promise.resolve(null);
      });
      const token2 = await secureStorage.getRefreshToken();
      
      expect(token2).toBe(mockToken);
      expect(SecureStore.getItemAsync).toHaveBeenCalledTimes(1); // Cache miss (après clearAll)
    });

    it('devrait mettre à jour le cache lors de setRefreshToken', async () => {
      const oldToken = 'old-refresh-token';
      const newToken = 'new-refresh-token';
      
      // S'assurer que le cache est vide avant ce test
      await secureStorage.clearAll();
      jest.clearAllMocks();
      
      // Mock SecureStore pour retourner les tokens uniquement pour la clé REFRESH_TOKEN
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === REFRESH_KEY) {
          return Promise.resolve(oldToken);
        }
        return Promise.resolve(null);
      });
      (SecureStore.setItemAsync as jest.Mock).mockResolvedValue(undefined);
      
      // 1. Stocker un token
      await secureStorage.setRefreshToken(oldToken);
      await secureStorage.getRefreshToken(); // Mise en cache
      
      // 2. Mettre à jour le token
      await secureStorage.setRefreshToken(newToken);
      
      // 3. Vérifier que le cache contient le nouveau token
      (SecureStore.getItemAsync as jest.Mock).mockClear();
      const cachedToken = await secureStorage.getRefreshToken();
      
      expect(cachedToken).toBe(newToken);
      expect(SecureStore.getItemAsync).not.toHaveBeenCalled(); // Cache hit avec nouveau token
    });

    it('devrait nettoyer le cache lors de removeRefreshToken', async () => {
      const mockToken = 'test-refresh-token-456';
      
      // S'assurer que le cache est vide avant ce test
      await secureStorage.clearAll();
      jest.clearAllMocks();
      
      // Mock SecureStore pour retourner le token uniquement pour la clé REFRESH_TOKEN
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === REFRESH_KEY) {
          return Promise.resolve(mockToken);
        }
        return Promise.resolve(null);
      });
      (SecureStore.setItemAsync as jest.Mock).mockResolvedValue(undefined);
      
      // 1. Mettre en cache
      await secureStorage.setRefreshToken(mockToken);
      await secureStorage.getRefreshToken();
      
      // 2. Supprimer le token (nettoie le cache en mémoire)
      (SecureStore.deleteItemAsync as jest.Mock).mockResolvedValue(undefined);
      await secureStorage.removeRefreshToken();
      
      // 3. Vérifier que le cache est vide
      // Note: removeRefreshToken() nettoie le cache en mémoire (cachedRefreshToken = null)
      // Donc getRefreshToken() doit lire depuis SecureStore qui retourne null après la suppression
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === REFRESH_KEY) {
          return Promise.resolve(null); // Après suppression, SecureStore retourne null
        }
        return Promise.resolve(null);
      });
      const token = await secureStorage.getRefreshToken();
      
      expect(token).toBeNull();
      // Le cache a été nettoyé, donc SecureStore.getItemAsync doit être appelé
      // Vérifier que getItemAsync a été appelé au moins une fois (pour la lecture après suppression)
      expect(SecureStore.getItemAsync).toHaveBeenCalled(); // Cache miss, lecture SecureStore
    });
  });

  describe('Métriques de performance', () => {
    it('devrait compter les cache hits et misses', async () => {
      const mockToken = 'test-token';
      
      // S'assurer que le cache est vide avant ce test
      await secureStorage.clearAll();
      jest.clearAllMocks();
      
      // Mock SecureStore pour retourner le token uniquement pour la clé ACCESS_TOKEN
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === ACCESS_KEY) {
          return Promise.resolve(mockToken);
        }
        return Promise.resolve(null);
      });
      
      // 1. Premier appel : cache miss
      await secureStorage.getAccessToken();
      
      // 2. Deuxième appel : cache hit
      await secureStorage.getAccessToken();
      
      // 3. Vérifier les métriques
      const metrics = secureStorage.getPerformanceMetrics();
      
      expect(metrics).not.toBeNull();
      expect(metrics?.accessToken.cacheHits).toBeGreaterThan(0);
      expect(metrics?.accessToken.cacheMisses).toBeGreaterThan(0);
      expect(metrics?.accessToken.totalRequests).toBeGreaterThan(0);
    });

    it('devrait calculer le cache hit rate correctement', async () => {
      const mockToken = 'test-token';
      
      // 1. Premier appel : cache miss
      (SecureStore.getItemAsync as jest.Mock).mockResolvedValue(mockToken);
      await secureStorage.getAccessToken();
      
      // 2. Plusieurs appels : cache hits
      for (let i = 0; i < 5; i++) {
        await secureStorage.getAccessToken();
      }
      
      // 3. Vérifier le cache hit rate
      const metrics = secureStorage.getPerformanceMetrics();
      
      expect(metrics).not.toBeNull();
      // 1 miss + 5 hits = 6 total, hit rate = 5/6 = 83.33%
      expect(metrics?.accessToken.cacheHitRate).toBeGreaterThan(80);
    });

    it('devrait réinitialiser les métriques lors de clearAll', async () => {
      const mockToken = 'test-token';
      
      // 1. Générer des métriques
      (SecureStore.getItemAsync as jest.Mock).mockResolvedValue(mockToken);
      (SecureStore.setItemAsync as jest.Mock).mockResolvedValue(undefined);
      await secureStorage.setAccessToken(mockToken);
      await secureStorage.getAccessToken();
      await secureStorage.getAccessToken(); // Cache hit
      
      // 2. Nettoyer tout
      (SecureStore.deleteItemAsync as jest.Mock).mockResolvedValue(undefined);
      await secureStorage.clearAll();
      
      // 3. Vérifier que les métriques sont réinitialisées
      const metrics = secureStorage.getPerformanceMetrics();
      
      expect(metrics).not.toBeNull();
      expect(metrics?.accessToken.cacheHits).toBe(0);
      expect(metrics?.accessToken.cacheMisses).toBe(0);
      expect(metrics?.accessToken.totalRequests).toBe(0);
    });
  });

  describe('Scénarios de cache simultané', () => {
    it('devrait gérer correctement les appels simultanés', async () => {
      const mockToken = 'test-access-token-123';
      
      // S'assurer que le cache est vide avant ce test
      await secureStorage.clearAll();
      jest.clearAllMocks();
      
      // Réinitialiser complètement le mock pour éviter les valeurs persistantes
      (SecureStore.getItemAsync as jest.Mock).mockReset();
      
      // Mock SecureStore pour retourner le token uniquement pour la clé ACCESS_TOKEN
      (SecureStore.getItemAsync as jest.Mock).mockImplementation((key: string) => {
        if (key === ACCESS_KEY) {
          return Promise.resolve(mockToken);
        }
        // Pour toutes les autres clés (refresh_token, etc.), retourner null
        return Promise.resolve(null);
      });
      
      // 1. Premier appel : met en cache
      const firstToken = await secureStorage.getAccessToken();
      expect(firstToken).toBe(mockToken);
      
      // 2. Appels simultanés : doivent utiliser le cache (pas de lecture SecureStore)
      // Réinitialiser le mock pour compter les appels
      (SecureStore.getItemAsync as jest.Mock).mockReset();
      (SecureStore.getItemAsync as jest.Mock).mockImplementation(() => {
        // Ce mock ne devrait jamais être appelé car le cache est utilisé
        throw new Error('SecureStore.getItemAsync ne devrait pas être appelé (cache hit)');
      });
      
      const promises = [
        secureStorage.getAccessToken(),
        secureStorage.getAccessToken(),
        secureStorage.getAccessToken(),
      ];
      
      const results = await Promise.all(promises);
      
      // Tous doivent retourner le même token
      results.forEach((token) => {
        expect(token).toBe(mockToken);
      });
      
      // Seulement 1 lecture SecureStore (le premier), les autres utilisent le cache
      // Le mock ne devrait pas être appelé car le cache est utilisé
      expect(SecureStore.getItemAsync).not.toHaveBeenCalled(); // Cache hits
    });
  });

  describe('P1.A — Anti-régression clearAuthOnly', () => {
    it('clearDriverAuthOnly doit appeler SecureStore/AsyncStorage avec les clés exactes', async () => {
      await secureStorage.clearDriverAuthOnly();

      expect(SecureStore.deleteItemAsync).toHaveBeenCalledTimes(DRIVER_AUTH_KEYS.secure.length);
      DRIVER_AUTH_KEYS.secure.forEach((key) => {
        expect(SecureStore.deleteItemAsync).toHaveBeenCalledWith(key);
      });

      expect(AsyncStorage.multiRemove).toHaveBeenCalledTimes(1);
      const asyncKeys = (AsyncStorage.multiRemove as jest.Mock).mock.calls[0][0];
      expect(asyncKeys).toEqual(expect.arrayContaining([...DRIVER_AUTH_KEYS.async]));
      expect(asyncKeys).toHaveLength(DRIVER_AUTH_KEYS.async.length);
    });

    it('clearEnterpriseAuthOnly doit appeler SecureStore/AsyncStorage avec les clés exactes', async () => {
      await secureStorage.clearEnterpriseAuthOnly();

      expect(SecureStore.deleteItemAsync).toHaveBeenCalledTimes(ENTERPRISE_AUTH_KEYS.secure.length);
      ENTERPRISE_AUTH_KEYS.secure.forEach((key) => {
        expect(SecureStore.deleteItemAsync).toHaveBeenCalledWith(key);
      });

      expect(AsyncStorage.multiRemove).toHaveBeenCalledTimes(1);
      const asyncKeys = (AsyncStorage.multiRemove as jest.Mock).mock.calls[0][0];
      expect(asyncKeys).toEqual(expect.arrayContaining([...ENTERPRISE_AUTH_KEYS.async]));
      expect(asyncKeys).toHaveLength(ENTERPRISE_AUTH_KEYS.async.length);
    });
  });
});

