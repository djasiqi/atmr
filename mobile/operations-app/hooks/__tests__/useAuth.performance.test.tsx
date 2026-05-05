// hooks/__tests__/useAuth.performance.test.tsx
// Tests de performance pour le temps de démarrage (Phase 2 : Optimisation démarrage)
// Valide que l'optimisation Promise.all réduit effectivement le temps de démarrage

import * as SecureStore from 'expo-secure-store';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { secureStorage } from '@/services/storage';
import { DRIVER_AUTH_SECURE_KEYS } from '@/services/storage/keys';

/** Aligné sur services/storage.ts SECURE_KEYS (pas [1] = backup refresh). */
const ACCESS_TOKEN_KEY = DRIVER_AUTH_SECURE_KEYS[2]; // driver_access_token
const REFRESH_TOKEN_KEY = DRIVER_AUTH_SECURE_KEYS[0]; // driver_refresh_token

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

// Mock __DEV__ pour activer les métriques
const originalDev = (global as any).__DEV__;
beforeAll(() => {
    (global as any).__DEV__ = true;
});

afterAll(() => {
    (global as any).__DEV__ = originalDev;
});

// Constantes pour les tests
const MODE_KEY = 'auth.mode';
const ENTERPRISE_DEVICE_KEY = 'enterprise.device_id';

// Helper pour simuler un délai (pour tester les performances)
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

describe('useAuth - Tests de performance démarrage', () => {
    beforeEach(async () => {
        // Nettoyer le cache avant chaque test
        await secureStorage.clearAll();
        jest.clearAllMocks();
        (SecureStore.getItemAsync as jest.Mock).mockClear();
        (AsyncStorage.getItem as jest.Mock).mockClear();
    });

    describe('Mesure temps de chargement initial (Promise.all)', () => {
        it('devrait charger les tokens en parallèle rapidement avec cache', async () => {
            const mockAccessToken = 'test-access-token-123';
            const mockRefreshToken = 'test-refresh-token-456';
            const mockMode = 'driver';

            // Mock SecureStore pour simuler des lectures rapides (5ms chacune)
            (SecureStore.getItemAsync as jest.Mock).mockImplementation(
                async (key: string) => {
                    await delay(5); // Simuler 5ms de lecture
                    if (key === ACCESS_TOKEN_KEY) return mockAccessToken;
                    if (key === REFRESH_TOKEN_KEY) return mockRefreshToken;
                    return null;
                }
            );

            (AsyncStorage.getItem as jest.Mock).mockImplementation(
                async (key: string) => {
                    await delay(5); // Simuler 5ms de lecture
                    if (key === MODE_KEY) return mockMode;
                    if (key === ENTERPRISE_DEVICE_KEY) return null;
                    return null;
                }
            );

            // 1. Premier appel : met en cache (cache miss)
            await secureStorage.setAccessToken(mockAccessToken);
            await secureStorage.setRefreshToken(mockRefreshToken);
            // setRefreshToken peut lire le primary pour le backup — ne pas compter ces appels
            (SecureStore.getItemAsync as jest.Mock).mockClear();

            // 2. Deuxième appel : utilise le cache (cache hit)
            const startTime = performance.now();
            const [storedMode, storedDevice, refreshToken, accessToken] =
                await Promise.all([
                    AsyncStorage.getItem(MODE_KEY),
                    AsyncStorage.getItem(ENTERPRISE_DEVICE_KEY),
                    secureStorage.getRefreshToken(),
                    secureStorage.getAccessToken(),
                ]);
            const endTime = performance.now();
            const duration = endTime - startTime;

            // Vérifier que les valeurs sont correctes
            expect(accessToken).toBe(mockAccessToken);
            expect(refreshToken).toBe(mockRefreshToken);
            expect(storedMode).toBe(mockMode);

            // ⚡ OPTIMISATION : Avec cache, le temps devrait être < 20ms
            // (les lectures AsyncStorage prennent ~5ms, mais SecureStore utilise le cache)
            expect(duration).toBeLessThan(20);

            // Vérifier que SecureStore.getItemAsync n'a PAS été appelé (cache hit)
            expect(SecureStore.getItemAsync).not.toHaveBeenCalled();
        });

        it('devrait charger les tokens en parallèle même sans cache (première lecture)', async () => {
            const mockAccessToken = 'test-access-token-123';
            const mockRefreshToken = 'test-refresh-token-456';
            const mockMode = 'driver';

            // Mock SecureStore pour simuler des lectures (5ms chacune)
            (SecureStore.getItemAsync as jest.Mock).mockImplementation(
                async (key: string) => {
                    await delay(5);
                    if (key === ACCESS_TOKEN_KEY) return mockAccessToken;
                    if (key === REFRESH_TOKEN_KEY) return mockRefreshToken;
                    return null;
                }
            );

            (AsyncStorage.getItem as jest.Mock).mockImplementation(
                async (key: string) => {
                    await delay(5);
                    if (key === MODE_KEY) return mockMode;
                    if (key === ENTERPRISE_DEVICE_KEY) return null;
                    return null;
                }
            );

            // Pas de cache : première lecture
            const startTime = performance.now();
            const [storedMode, storedDevice, refreshToken, accessToken] =
                await Promise.all([
                    AsyncStorage.getItem(MODE_KEY),
                    AsyncStorage.getItem(ENTERPRISE_DEVICE_KEY),
                    secureStorage.getRefreshToken(),
                    secureStorage.getAccessToken(),
                ]);
            const endTime = performance.now();
            const duration = endTime - startTime;

            // Vérifier que les valeurs sont correctes
            expect(accessToken).toBe(mockAccessToken);
            expect(refreshToken).toBe(mockRefreshToken);
            expect(storedMode).toBe(mockMode);

            // ⚡ OPTIMISATION : En parallèle, le temps devrait être ~5-15ms
            // (toutes les lectures se font en parallèle, donc le temps = max(lectures) ~5ms)
            // Au lieu de séquentiel qui serait ~20ms (5ms * 4 lectures)
            // Tolérance pour variations machines / CI (timers non déterministes)
            expect(duration).toBeLessThan(40);

            // Vérifier que SecureStore.getItemAsync a été appelé 2 fois (access + refresh)
            expect(SecureStore.getItemAsync).toHaveBeenCalledTimes(2);
        });

        it('devrait être plus rapide en parallèle qu\'en séquentiel', async () => {
            const mockAccessToken = 'test-access-token-123';
            const mockRefreshToken = 'test-refresh-token-456';
            const mockMode = 'driver';

            // Mock SecureStore pour simuler des lectures (5ms chacune)
            (SecureStore.getItemAsync as jest.Mock).mockImplementation(
                async (key: string) => {
                    await delay(5);
                    if (key === ACCESS_TOKEN_KEY) return mockAccessToken;
                    if (key === REFRESH_TOKEN_KEY) return mockRefreshToken;
                    return null;
                }
            );

            (AsyncStorage.getItem as jest.Mock).mockImplementation(
                async (key: string) => {
                    await delay(5);
                    if (key === MODE_KEY) return mockMode;
                    if (key === ENTERPRISE_DEVICE_KEY) return null;
                    return null;
                }
            );

            // 1. Test séquentiel
            await secureStorage.clearAll(); // S'assurer qu'il n'y a pas de cache
            const sequentialStart = performance.now();
            const mode1 = await AsyncStorage.getItem(MODE_KEY);
            const device1 = await AsyncStorage.getItem(ENTERPRISE_DEVICE_KEY);
            const refresh1 = await secureStorage.getRefreshToken();
            const access1 = await secureStorage.getAccessToken();
            const sequentialEnd = performance.now();
            const sequentialDuration = sequentialEnd - sequentialStart;

            // 2. Test parallèle
            await secureStorage.clearAll(); // Réinitialiser
            const parallelStart = performance.now();
            const [mode2, device2, refresh2, access2] = await Promise.all([
                AsyncStorage.getItem(MODE_KEY),
                AsyncStorage.getItem(ENTERPRISE_DEVICE_KEY),
                secureStorage.getRefreshToken(),
                secureStorage.getAccessToken(),
            ]);
            const parallelEnd = performance.now();
            const parallelDuration = parallelEnd - parallelStart;

            // Vérifier que les valeurs sont identiques
            expect(access1).toBe(access2);
            expect(refresh1).toBe(refresh2);
            expect(mode1).toBe(mode2);

            // ⚡ OPTIMISATION : Le parallèle devrait être significativement plus rapide
            // Séquentiel : ~20ms (5ms * 4 lectures)
            // Parallèle : ~5-15ms (max des lectures en parallèle)
            expect(parallelDuration).toBeLessThan(sequentialDuration);
            expect(parallelDuration).toBeLessThan(40); // Tolérance CI (timing non déterministe)
        });
    });

    describe('Nombre de lectures SecureStore lors du démarrage', () => {
        it('devrait faire 0 lecture SecureStore si le cache est valide', async () => {
            const mockAccessToken = 'test-access-token-123';
            const mockRefreshToken = 'test-refresh-token-456';

            // Mettre en cache
            await secureStorage.setAccessToken(mockAccessToken);
            await secureStorage.setRefreshToken(mockRefreshToken);

            // Réinitialiser le compteur
            (SecureStore.getItemAsync as jest.Mock).mockClear();

            // Charger les tokens (devrait utiliser le cache)
            await Promise.all([
                secureStorage.getRefreshToken(),
                secureStorage.getAccessToken(),
            ]);

            // Vérifier qu'aucune lecture SecureStore n'a été faite (cache hit)
            expect(SecureStore.getItemAsync).not.toHaveBeenCalled();
        });

        it('devrait faire 2 lectures SecureStore si le cache est vide', async () => {
            const mockAccessToken = 'test-access-token-123';
            const mockRefreshToken = 'test-refresh-token-456';

            // Pas de cache
            await secureStorage.clearAll();

            (SecureStore.getItemAsync as jest.Mock).mockImplementation(
                async (key: string) => {
                    if (key === ACCESS_TOKEN_KEY) return mockAccessToken;
                    if (key === REFRESH_TOKEN_KEY) return mockRefreshToken;
                    return null;
                }
            );

            // Charger les tokens (cache miss)
            await Promise.all([
                secureStorage.getRefreshToken(),
                secureStorage.getAccessToken(),
            ]);

            // Vérifier que SecureStore.getItemAsync a été appelé 2 fois (access + refresh)
            expect(SecureStore.getItemAsync).toHaveBeenCalledTimes(2);
        });

        it('devrait faire 1 lecture SecureStore si seulement un token est en cache', async () => {
            const mockAccessToken = 'test-access-token-123';
            const mockRefreshToken = 'test-refresh-token-456';

            // Nettoyer tout d'abord
            await secureStorage.clearAll();
            jest.clearAllMocks();

            // Mettre seulement access_token en cache (sans refresh_token)
            await secureStorage.setAccessToken(mockAccessToken);
            // Ne pas mettre refresh_token en cache (pas de setRefreshToken)

            // Réinitialiser les mocks pour compter les appels
            (SecureStore.getItemAsync as jest.Mock).mockClear();
            (SecureStore.getItemAsync as jest.Mock).mockImplementation(
                async (key: string) => {
                    if (key === REFRESH_TOKEN_KEY) return mockRefreshToken;
                    // Pour access_token, ne pas retourner de valeur car il est déjà en cache
                    return null;
                }
            );

            // Charger les tokens
            await Promise.all([
                secureStorage.getRefreshToken(), // Cache miss → lecture SecureStore (1 appel)
                secureStorage.getAccessToken(), // Cache hit (pas de lecture SecureStore)
            ]);

            // Vérifier que SecureStore.getItemAsync a été appelé 1 fois (seulement refresh_token)
            // Note: getAccessToken() utilise le cache, donc pas d'appel SecureStore
            expect(SecureStore.getItemAsync).toHaveBeenCalledTimes(1);
            expect(SecureStore.getItemAsync).toHaveBeenCalledWith(
                REFRESH_TOKEN_KEY,
                expect.anything()
            );
        });
    });

    describe('Scénarios de démarrage', () => {
        it('devrait démarrer rapidement avec tokens en cache (cache hit)', async () => {
            const mockAccessToken = 'test-access-token-123';
            const mockRefreshToken = 'test-refresh-token-456';
            const mockMode = 'driver';

            // Mettre en cache
            await secureStorage.setAccessToken(mockAccessToken);
            await secureStorage.setRefreshToken(mockRefreshToken);
            (SecureStore.getItemAsync as jest.Mock).mockClear();

            (AsyncStorage.getItem as jest.Mock).mockResolvedValue(mockMode);

            // Mesurer le temps de démarrage
            const startTime = performance.now();
            const [storedMode, storedDevice, refreshToken, accessToken] =
                await Promise.all([
                    AsyncStorage.getItem(MODE_KEY),
                    AsyncStorage.getItem(ENTERPRISE_DEVICE_KEY),
                    secureStorage.getRefreshToken(),
                    secureStorage.getAccessToken(),
                ]);
            const endTime = performance.now();
            const duration = endTime - startTime;

            // Vérifier les valeurs
            expect(accessToken).toBe(mockAccessToken);
            expect(refreshToken).toBe(mockRefreshToken);
            expect(storedMode).toBe(mockMode);

            // ⚡ OPTIMISATION : Avec cache, < 20ms
            expect(duration).toBeLessThan(20);

            // Aucune lecture SecureStore (cache hit)
            expect(SecureStore.getItemAsync).not.toHaveBeenCalled();
        });

        it('devrait démarrer rapidement sans cache (première lecture)', async () => {
            const mockAccessToken = 'test-access-token-123';
            const mockRefreshToken = 'test-refresh-token-456';
            const mockMode = 'driver';

            // Pas de cache
            await secureStorage.clearAll();

            (SecureStore.getItemAsync as jest.Mock).mockImplementation(
                async (key: string) => {
                    await delay(5); // Simuler 5ms de lecture
                    if (key === ACCESS_TOKEN_KEY) return mockAccessToken;
                    if (key === REFRESH_TOKEN_KEY) return mockRefreshToken;
                    return null;
                }
            );

            (AsyncStorage.getItem as jest.Mock).mockImplementation(
                async (key: string) => {
                    await delay(5);
                    if (key === MODE_KEY) return mockMode;
                    return null;
                }
            );

            // Mesurer le temps de démarrage
            const startTime = performance.now();
            const [storedMode, storedDevice, refreshToken, accessToken] =
                await Promise.all([
                    AsyncStorage.getItem(MODE_KEY),
                    AsyncStorage.getItem(ENTERPRISE_DEVICE_KEY),
                    secureStorage.getRefreshToken(),
                    secureStorage.getAccessToken(),
                ]);
            const endTime = performance.now();
            const duration = endTime - startTime;

            // Vérifier les valeurs
            expect(accessToken).toBe(mockAccessToken);
            expect(refreshToken).toBe(mockRefreshToken);
            expect(storedMode).toBe(mockMode);

            // ⚡ OPTIMISATION : en parallèle (tolérance large pour CI / machines lentes)
            expect(duration).toBeLessThan(40);

            // 2 lectures SecureStore (access + refresh)
            expect(SecureStore.getItemAsync).toHaveBeenCalledTimes(2);
        });

        it('devrait démarrer rapidement sans tokens (première connexion)', async () => {
            // Pas de tokens
            await secureStorage.clearAll();

            (SecureStore.getItemAsync as jest.Mock).mockResolvedValue(null);
            (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);

            // Mesurer le temps de démarrage
            const startTime = performance.now();
            const [storedMode, storedDevice, refreshToken, accessToken] =
                await Promise.all([
                    AsyncStorage.getItem(MODE_KEY),
                    AsyncStorage.getItem(ENTERPRISE_DEVICE_KEY),
                    secureStorage.getRefreshToken(),
                    secureStorage.getAccessToken(),
                ]);
            const endTime = performance.now();
            const duration = endTime - startTime;

            // Vérifier que les tokens sont null
            expect(accessToken).toBeNull();
            expect(refreshToken).toBeNull();

            // ⚡ OPTIMISATION : Même sans tokens, < 15ms en parallèle
            expect(duration).toBeLessThan(15);

            // refresh: primary + backup fallback ; access: 1 lecture
            expect(SecureStore.getItemAsync).toHaveBeenCalledTimes(3);
        });

        it('devrait nettoyer rapidement avec refresh token invalide', async () => {
            const invalidRefreshToken = 'invalid-refresh-token';

            // Mettre un refresh token en cache
            await secureStorage.setRefreshToken(invalidRefreshToken);

            // Simuler que le refresh token est invalide (retourne null)
            (SecureStore.getItemAsync as jest.Mock).mockImplementation(
                async (key: string) => {
                    if (key === REFRESH_TOKEN_KEY) return invalidRefreshToken;
                    return null;
                }
            );

            // Mesurer le temps de lecture
            const startTime = performance.now();
            const refreshToken = await secureStorage.getRefreshToken();
            const endTime = performance.now();
            const duration = endTime - startTime;

            // Vérifier que le token est récupéré
            expect(refreshToken).toBe(invalidRefreshToken);

            // ⚡ OPTIMISATION : Lecture rapide même si le token est invalide
            expect(duration).toBeLessThan(10);

            // Nettoyer (simule le comportement de useAuth lors d'un refresh token invalide)
            const clearStart = performance.now();
            await secureStorage.clearAll();
            const clearEnd = performance.now();
            const clearDuration = clearEnd - clearStart;

            // Nettoyage rapide
            expect(clearDuration).toBeLessThan(10);
        });
    });

    describe('Métriques de performance', () => {
        it('devrait mesurer le cache hit rate après plusieurs lectures', async () => {
            const mockAccessToken = 'test-access-token-123';
            const mockRefreshToken = 'test-refresh-token-456';

            // Mettre en cache
            await secureStorage.setAccessToken(mockAccessToken);
            await secureStorage.setRefreshToken(mockRefreshToken);

            // Faire plusieurs lectures (devraient utiliser le cache)
            for (let i = 0; i < 5; i++) {
                await Promise.all([
                    secureStorage.getRefreshToken(),
                    secureStorage.getAccessToken(),
                ]);
            }

            // Vérifier les métriques
            const metrics = secureStorage.getPerformanceMetrics();
            expect(metrics).not.toBeNull();
            expect(metrics?.accessToken.cacheHits).toBeGreaterThan(0);
            expect(metrics?.accessToken.cacheHitRate).toBeGreaterThan(80); // > 80% cache hit rate
        });

        it('devrait mesurer le temps moyen de lecture', async () => {
            const mockAccessToken = 'test-access-token-123';

            // Pas de cache initial
            await secureStorage.clearAll();

            (SecureStore.getItemAsync as jest.Mock).mockImplementation(
                async (key: string) => {
                    await delay(5); // Simuler 5ms de lecture
                    if (key === ACCESS_TOKEN_KEY) return mockAccessToken;
                    return null;
                }
            );

            // Faire plusieurs lectures
            for (let i = 0; i < 3; i++) {
                await secureStorage.getAccessToken();
            }

            // Vérifier les métriques
            const metrics = secureStorage.getPerformanceMetrics();
            expect(metrics).not.toBeNull();
            expect(metrics?.accessToken.totalRequests).toBe(3);
            expect(metrics?.accessToken.avgReadTime).toBeGreaterThan(0);
        });
    });
});

